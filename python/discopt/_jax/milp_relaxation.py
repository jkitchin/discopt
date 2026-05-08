"""
MILP Relaxation Builder for AMP (Adaptive Multivariate Partitioning).

Builds a linear programming relaxation of the original MINLP by:
  1. Replacing bilinear terms x_i*x_j with auxiliary variables w_ij and
     adding standard McCormick envelope constraints.
  2. Replacing monomial terms x_i^n with auxiliary variables s_i and adding
     piecewise tangent-cut underestimators plus partition-activated secant
     overestimators when the variable is discretized.
  3. Linearizing the original objective and constraints.

The LP relaxation gives a valid lower bound:
  LP_opt ≤ global NLP_opt

As the partition becomes finer (more intervals in disc_state), more tangent and
local secant cuts are added for monomials, tightening the lower bound.

Theory: Nagarajan et al., JOGO 2018, Section 4 (piecewise McCormick relaxation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt._jax.discretization import DiscretizationState
from discopt._jax.embedding import EmbeddingMap, build_embedding_map
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.term_classifier import (
    NonlinearTerms,
    _compute_var_offset,
    _get_flat_index,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    ObjectiveSense,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result and model wrappers
# ---------------------------------------------------------------------------


@dataclass
class MilpRelaxationResult:
    """Result of solving a MILP relaxation."""

    status: str  # "optimal", "infeasible", "error", "time_limit"
    objective: Optional[float] = None
    x: Optional[np.ndarray] = None


class MilpRelaxationModel:
    """Wrapper around a MILP that exposes a .solve() method.

    Stores the LP data and delegates solving to solve_milp (HiGHS).
    """

    def __init__(
        self,
        c: np.ndarray,
        A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
        b_ub: Optional[np.ndarray],
        bounds: list[tuple[float, float]],
        obj_offset: float = 0.0,
        integrality: Optional[np.ndarray] = None,
        objective_bound_valid: bool = True,
    ):
        self._c = c
        self._A_ub = A_ub
        self._b_ub = b_ub
        self._bounds = bounds
        self._obj_offset = obj_offset
        self._integrality = integrality
        self._objective_bound_valid = objective_bound_valid

    def solve(
        self,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
    ) -> MilpRelaxationResult:
        from discopt.solvers import SolveStatus
        from discopt.solvers.milp_highs import solve_milp

        result = solve_milp(
            c=self._c,
            A_ub=self._A_ub,
            b_ub=self._b_ub,
            bounds=self._bounds,
            integrality=self._integrality,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )

        # Map SolveStatus enum to string
        status_map = {
            SolveStatus.OPTIMAL: "optimal",
            SolveStatus.INFEASIBLE: "infeasible",
            SolveStatus.UNBOUNDED: "unbounded",
            SolveStatus.TIME_LIMIT: "time_limit",
            SolveStatus.ITERATION_LIMIT: "iteration_limit",
            SolveStatus.ERROR: "error",
        }
        status_str = status_map.get(result.status, str(result.status))

        obj = None
        if result.objective is not None and self._objective_bound_valid:
            obj = float(result.objective) + self._obj_offset

        return MilpRelaxationResult(status=status_str, objective=obj, x=result.x)


@dataclass
class UnivariateRelaxation:
    """Lifted outer relaxation for a supported univariate operator."""

    expr_id: int
    func_name: str
    aux_col: int
    arg_coeff: np.ndarray
    arg_const: float
    arg_lb: float
    arg_ub: float


# ---------------------------------------------------------------------------
# Helpers: variable bounds
# ---------------------------------------------------------------------------


def _piecewise_product_bounds(
    a_k: float,
    b_k: float,
    y_lb: float,
    y_ub: float,
) -> tuple[list[float], float, float]:
    """Return interval corner products and their min/max values."""
    corners = [a_k * y_lb, a_k * y_ub, b_k * y_lb, b_k * y_ub]
    return corners, min(corners), max(corners)


def _compute_piecewise_big_m(corners: list[float]) -> float:
    """Scale Big-M with the interval magnitude instead of adding a flat constant."""
    max_corner = max(abs(float(c)) for c in corners)
    return max_corner * (1.0 + 1e-4) + max(1e-6, 1e-4 * max_corner)


def _linear_expr_bounds(
    coeff: np.ndarray,
    const: float,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float]:
    """Return interval bounds for an affine expression over variable bounds."""
    lower = float(const)
    upper = float(const)
    for c_i, lb_i, ub_i in zip(coeff, lb, ub):
        c = float(c_i)
        if c >= 0.0:
            lower += c * float(lb_i)
            upper += c * float(ub_i)
        else:
            lower += c * float(ub_i)
            upper += c * float(lb_i)
    return lower, upper


def _normalize_convhull_formulation(formulation: str) -> str:
    """Normalize accepted bilinear convex-hull mode names."""
    aliases = {
        "disaggregated": "disaggregated",
        "piecewise": "disaggregated",
        "sos2": "sos2",
        "facet": "facet",
        "lambda": "sos2",
    }
    try:
        return aliases[formulation]
    except KeyError as err:
        raise ValueError(
            f"Unsupported convhull_formulation: {formulation!r}. "
            "Choose from 'disaggregated', 'sos2', 'facet', or 'lambda'."
        ) from err


def _sorted_unique_points(points: list[float]) -> list[float]:
    """Return sorted points with near-duplicates removed."""
    unique: list[float] = []
    for point in sorted(float(p) for p in points):
        if not unique or abs(point - unique[-1]) > 1e-12:
            unique.append(point)
    return unique


def _power_tangent_line(t: float, n: int) -> tuple[float, float]:
    """Return slope/intercept for the tangent to x**n at x=t."""
    slope = float(n * (t ** (n - 1)))
    intercept = float((t**n) - slope * t)
    return slope, intercept


def _power_secant_line(lb: float, ub: float, n: int) -> tuple[float, float]:
    """Return slope/intercept for the secant through (lb, lb**n) and (ub, ub**n)."""
    if abs(ub - lb) <= 1e-12:
        return 0.0, float(lb**n)
    slope = float((ub**n - lb**n) / (ub - lb))
    intercept = float(lb**n - slope * lb)
    return slope, intercept


def _power_is_convex_on_box(n: int, lb: float) -> bool:
    """Return True when x**n is convex on the current box."""
    return n % 2 == 0 or lb >= 0.0


def _monomial_breakpoints(
    var_idx: int,
    lb_i: float,
    ub_i: float,
    disc_state: DiscretizationState,
) -> list[float]:
    """Return refinement-aware monomial cut points, including zero when needed."""
    if var_idx in disc_state.partitions and len(disc_state.partitions[var_idx]) >= 2:
        points = [float(p) for p in disc_state.partitions[var_idx]]
    else:
        points = [lb_i, ub_i]
    if lb_i < 0.0 < ub_i:
        points.append(0.0)
    return _sorted_unique_points(points)


def _odd_mixed_tangent_is_valid(
    t: float,
    lb: float,
    ub: float,
    n: int,
    kind: str,
) -> bool:
    """Check whether the tangent at t is a global under/over-estimator on [lb, ub]."""
    slope, intercept = _power_tangent_line(t, n)
    critical_points = [lb, ub, t]
    mirrored = -t
    if lb <= mirrored <= ub:
        critical_points.append(mirrored)

    diffs = [float(x**n - (slope * x + intercept)) for x in _sorted_unique_points(critical_points)]
    tol = 1e-10
    if kind == "under":
        return all(diff >= -tol for diff in diffs)
    if kind == "over":
        return all(diff <= tol for diff in diffs)
    raise ValueError(f"Unknown tangent validity kind: {kind}")


def _choose_trilinear_pair(
    term: tuple[int, int, int],
    partitioned_vars: set[int],
) -> tuple[tuple[int, int], int]:
    """Choose a deterministic trilinear decomposition pair.

    Prefer a pair that includes as many currently partitioned original variables as
    possible so the first or second lifted bilinear term can reuse the stronger
    piecewise relaxation machinery already present for bilinear terms.
    """
    i, j, k = tuple(sorted(term))
    candidates = [((i, j), k), ((i, k), j), ((j, k), i)]
    candidates.sort()
    return max(
        candidates,
        key=lambda item: (
            sum(v in partitioned_vars for v in item[0]),
            item[0][0] in partitioned_vars or item[0][1] in partitioned_vars,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers: expression decomposition
# ---------------------------------------------------------------------------


def _decompose_product(expr: Expression, model: Model) -> tuple[float, list[int]] | None:
    """Decompose a product expression into (scalar, [flat_var_idx, ...]).

    Returns None if expr contains non-constant, non-variable leaves.
    Constants are accumulated into the scalar; variable references are
    appended to the index list.
    """
    scalar: list[float] = [1.0]
    var_indices: list[int] = []

    def visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, Constant):
            scalar[0] *= float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            var_indices.append(flat)
            return True
        return False

    if visit(expr):
        return scalar[0], var_indices
    return None


def _collect_distinct_multilinear_products(model: Model) -> list[tuple[int, ...]]:
    """Return distinct-variable product terms with four or more factors."""
    terms: set[tuple[int, ...]] = set()

    def visit(expr: Expression) -> None:
        if isinstance(expr, BinaryOp):
            if expr.op == "*":
                decomp = _decompose_product(expr, model)
                if decomp is not None:
                    _, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if len(unique) >= 4 and len(unique) == len(indices):
                        terms.add(tuple(sorted(unique)))
                        return
            visit(expr.left)
            visit(expr.right)
            return

        if isinstance(expr, UnaryOp):
            visit(expr.operand)
            return

        if isinstance(expr, SumExpression):
            visit(expr.operand)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)

    return sorted(terms)


def _linearize_affine_expr(expr: Expression, model: Model, n_vars: int) -> tuple[np.ndarray, float]:
    """Linearize an affine expression over original variables.

    Raises ValueError when the expression contains nonlinear structure.  This is
    intentionally narrower than _linearize_expr because univariate operator
    relaxations are only soundly supported here for affine arguments.
    """
    coeff = np.zeros(n_vars, dtype=np.float64)
    const_acc: list[float] = [0.0]

    def visit(e: Expression, scale: float) -> None:
        if isinstance(e, Constant):
            const_acc[0] += scale * float(e.value)
            return

        if isinstance(e, Variable):
            offset = _compute_var_offset(e, model)
            if e.size == 1:
                coeff[offset] += scale
                return
            raise ValueError(f"Cannot use array variable as scalar affine argument: {e}")

        if isinstance(e, IndexExpression):
            flat = _get_flat_index(e, model)
            if flat is None:
                raise ValueError(f"Cannot linearize IndexExpression: {e}")
            coeff[flat] += scale
            return

        if isinstance(e, UnaryOp) and e.op == "neg":
            visit(e.operand, -scale)
            return

        if isinstance(e, BinaryOp):
            if e.op == "+":
                visit(e.left, scale)
                visit(e.right, scale)
                return
            if e.op == "-":
                visit(e.left, scale)
                visit(e.right, -scale)
                return
            if e.op == "*":
                if isinstance(e.left, Constant):
                    visit(e.right, scale * float(e.left.value))
                    return
                if isinstance(e.right, Constant):
                    visit(e.left, scale * float(e.right.value))
                    return
                raise ValueError(f"Non-affine product in univariate argument: {e}")
            if e.op == "/":
                if isinstance(e.right, Constant):
                    visit(e.left, scale / float(e.right.value))
                    return
                raise ValueError(f"Non-affine division in univariate argument: {e}")
            if e.op == "**":
                if isinstance(e.right, Constant):
                    exp = float(e.right.value)
                    if exp == 1.0:
                        visit(e.left, scale)
                        return
                    if exp == 0.0:
                        const_acc[0] += scale
                        return
                raise ValueError(f"Non-affine power in univariate argument: {e}")

        if isinstance(e, SumExpression):
            op = e.operand
            if isinstance(op, Variable):
                offset = _compute_var_offset(op, model)
                for k in range(op.size):
                    coeff[offset + k] += scale
                return
            visit(op, scale)
            return

        if isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term, scale)
            return

        raise ValueError(f"Unsupported affine argument node {type(e).__name__}: {e}")

    visit(expr, 1.0)
    return coeff, const_acc[0]


def _univariate_arg(expr: Expression) -> tuple[str, Expression] | None:
    """Return (operator_name, argument) for supported univariate nodes."""
    if isinstance(expr, FunctionCall) and len(expr.args) == 1:
        name = expr.func_name
        if name in {"sqrt", "log", "log2", "log10", "exp", "abs"}:
            return name, expr.args[0]
    if isinstance(expr, UnaryOp) and expr.op == "abs":
        return "abs", expr.operand
    return None


def _univariate_value(func_name: str, x: float) -> float:
    """Evaluate a supported scalar univariate function."""
    if func_name == "sqrt":
        return float(np.sqrt(x))
    if func_name == "log":
        return float(np.log(x))
    if func_name == "log2":
        return float(np.log2(x))
    if func_name == "log10":
        return float(np.log10(x))
    if func_name == "exp":
        return float(np.exp(x))
    if func_name == "abs":
        return float(abs(x))
    raise ValueError(f"Unsupported univariate function: {func_name}")


def _univariate_grad(func_name: str, x: float) -> float:
    """Evaluate the first derivative of a smooth supported univariate function."""
    if func_name == "sqrt":
        return float(0.5 / np.sqrt(x))
    if func_name == "log":
        return float(1.0 / x)
    if func_name == "log2":
        return float(1.0 / (x * np.log(2.0)))
    if func_name == "log10":
        return float(1.0 / (x * np.log(10.0)))
    if func_name == "exp":
        return float(np.exp(x))
    raise ValueError(f"No smooth derivative for univariate function: {func_name}")


def _univariate_domain_ok(func_name: str, arg_lb: float, arg_ub: float) -> bool:
    """Return True when the operator can be relaxed on the interval."""
    if not np.isfinite(arg_lb) or not np.isfinite(arg_ub) or arg_lb > arg_ub:
        return False
    if func_name == "sqrt" and arg_lb < 0.0:
        return False
    if func_name in {"log", "log2", "log10"} and arg_lb <= 0.0:
        return False
    if func_name in {"sqrt", "log", "log2", "log10"}:
        return True
    if func_name == "exp":
        return bool(np.isfinite(np.exp(arg_lb)) and np.isfinite(np.exp(arg_ub)))
    if func_name == "abs":
        return True
    return False


def _univariate_value_bounds(func_name: str, arg_lb: float, arg_ub: float) -> tuple[float, float]:
    """Return finite bounds for f(x) on [arg_lb, arg_ub]."""
    if func_name == "abs":
        if arg_lb <= 0.0 <= arg_ub:
            return 0.0, max(abs(arg_lb), abs(arg_ub))
        values = [abs(arg_lb), abs(arg_ub)]
        return min(values), max(values)
    values = [_univariate_value(func_name, arg_lb), _univariate_value(func_name, arg_ub)]
    return min(values), max(values)


def _tangent_points(func_name: str, lb: float, ub: float) -> list[float]:
    """Choose deterministic valid tangent points for smooth univariate cuts."""
    raw_points = [lb, 0.5 * (lb + ub), ub]
    points: list[float] = []
    for pt in raw_points:
        if func_name == "sqrt" and pt <= 0.0:
            continue
        if func_name in {"log", "log2", "log10"} and pt <= 0.0:
            continue
        if not np.isfinite(pt):
            continue
        if all(abs(pt - seen) > 1e-12 for seen in points):
            points.append(float(pt))
    return points


def _collect_univariate_relaxations(
    model: Model,
    n_orig: int,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    start_col: int,
) -> tuple[list[UnivariateRelaxation], dict[int, int], list[tuple[float, float]]]:
    """Collect supported univariate operator nodes and assign auxiliary columns."""
    relaxations: list[UnivariateRelaxation] = []
    var_map: dict[int, int] = {}
    bounds: list[tuple[float, float]] = []
    seen: set[int] = set()
    col_idx = start_col

    def maybe_add(expr: Expression) -> None:
        nonlocal col_idx
        expr_id = id(expr)
        if expr_id in seen:
            return
        op_info = _univariate_arg(expr)
        if op_info is None:
            return
        func_name, arg = op_info
        try:
            arg_coeff, arg_const = _linearize_affine_expr(arg, model, n_orig)
            arg_lb, arg_ub = _linear_expr_bounds(arg_coeff, arg_const, flat_lb, flat_ub)
        except ValueError:
            return
        if not _univariate_domain_ok(func_name, arg_lb, arg_ub):
            return
        val_lb, val_ub = _univariate_value_bounds(func_name, arg_lb, arg_ub)
        if not np.isfinite(val_lb) or not np.isfinite(val_ub):
            return
        seen.add(expr_id)
        var_map[expr_id] = col_idx
        relaxations.append(
            UnivariateRelaxation(
                expr_id=expr_id,
                func_name=func_name,
                aux_col=col_idx,
                arg_coeff=arg_coeff,
                arg_const=arg_const,
                arg_lb=float(arg_lb),
                arg_ub=float(arg_ub),
            )
        )
        bounds.append((float(val_lb), float(val_ub)))
        col_idx += 1

    def visit(expr: Expression) -> None:
        maybe_add(expr)
        if isinstance(expr, BinaryOp):
            visit(expr.left)
            visit(expr.right)
        elif isinstance(expr, UnaryOp):
            visit(expr.operand)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
        elif isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
        elif isinstance(expr, SumExpression):
            visit(expr.operand)
        elif isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)

    return relaxations, var_map, bounds


# ---------------------------------------------------------------------------
# Helpers: expression linearizer
# ---------------------------------------------------------------------------


def _linearize_expr(
    expr: Expression,
    model: Model,
    bilinear_var_map: dict[tuple[int, int], int],
    trilinear_var_map: dict[tuple[int, int, int], int],
    multilinear_var_map: dict[tuple[int, ...], int],
    monomial_var_map: dict[tuple[int, int], int],
    univariate_var_map: dict[int, int],
    n_total_vars: int,
) -> tuple[np.ndarray, float]:
    """Walk expression tree and return (coeff, constant) for linearized form.

    coeff[j] = coefficient of MILP variable j in the linear approximation.
    constant = scalar constant term.

    Nonlinear terms must have a corresponding auxiliary variable in the maps;
    raises ValueError if an unregistered nonlinear term is encountered.
    """
    coeff = np.zeros(n_total_vars, dtype=np.float64)
    const_acc: list[float] = [0.0]

    def visit(e: Expression, scale: float) -> None:  # noqa: C901
        if isinstance(e, Constant):
            const_acc[0] += scale * float(e.value)

        elif isinstance(e, Variable):
            offset = _compute_var_offset(e, model)
            if e.size == 1:
                coeff[offset] += scale
            else:
                # Multi-element variable (unusual in scalar expression)
                for k in range(e.size):
                    coeff[offset + k] += scale

        elif isinstance(e, IndexExpression):
            flat = _get_flat_index(e, model)
            if flat is not None:
                coeff[flat] += scale
            else:
                raise ValueError(f"Cannot linearize IndexExpression: {e}")

        elif isinstance(e, FunctionCall):
            aux_col = univariate_var_map.get(id(e))
            if aux_col is not None:
                coeff[aux_col] += scale
            else:
                raise ValueError(f"Cannot linearize FunctionCall: {e}")

        elif isinstance(e, BinaryOp):
            if e.op == "+":
                visit(e.left, scale)
                visit(e.right, scale)

            elif e.op == "-":
                visit(e.left, scale)
                visit(e.right, -scale)

            elif e.op == "/":
                if isinstance(e.right, Constant):
                    visit(e.left, scale / float(e.right.value))
                else:
                    raise ValueError(f"Cannot linearize non-constant division: {e}")

            elif e.op == "**":
                flat = _get_flat_index(e.left, model)
                if flat is not None and isinstance(e.right, Constant):
                    n = int(float(e.right.value))
                    if n == 1:
                        coeff[flat] += scale
                        return
                    if n == 0:
                        const_acc[0] += scale
                        return
                    key = (flat, n)
                    if key in monomial_var_map:
                        coeff[monomial_var_map[key]] += scale
                    else:
                        raise ValueError(f"Monomial {key} not in monomial_var_map")
                else:
                    raise ValueError(f"Cannot linearize power expression: {e}")

            elif e.op == "*":
                # Constant scaling?
                if isinstance(e.left, Constant):
                    visit(e.right, scale * float(e.left.value))
                    return
                if isinstance(e.right, Constant):
                    visit(e.left, scale * float(e.right.value))
                    return
                # Full product decomposition
                decomp = _decompose_product(e, model)
                if decomp is None:
                    raise ValueError(f"Cannot decompose product: {e}")
                c, indices = decomp
                unique = list(dict.fromkeys(indices))
                if len(indices) == 0:
                    const_acc[0] += scale * c
                elif len(unique) == 1 and len(indices) == 1:
                    coeff[unique[0]] += scale * c
                elif len(unique) == 1:
                    # x^n monomial
                    n = len(indices)
                    key = (unique[0], n)
                    if key in monomial_var_map:
                        coeff[monomial_var_map[key]] += scale * c
                    else:
                        raise ValueError(f"Monomial {key} not in map")
                elif len(unique) == 2:
                    if len(unique) != len(indices):
                        raise ValueError("Mixed repeated-factor products are not supported")
                    i_idx, j_idx = unique[0], unique[1]
                    key = (min(i_idx, j_idx), max(i_idx, j_idx))
                    if key in bilinear_var_map:
                        coeff[bilinear_var_map[key]] += scale * c
                    else:
                        raise ValueError(f"Bilinear {key} not in map")
                elif len(unique) == 3:
                    if len(unique) != len(indices):
                        raise ValueError("Mixed repeated-factor products are not supported")
                    ordered = sorted(unique)
                    tri_key = (ordered[0], ordered[1], ordered[2])
                    if tri_key in trilinear_var_map:
                        coeff[trilinear_var_map[tri_key]] += scale * c
                    else:
                        raise ValueError(f"Trilinear {tri_key} not in map")
                else:
                    if len(unique) != len(indices):
                        raise ValueError("Mixed repeated-factor products are not supported")
                    multilinear_key = tuple(sorted(unique))
                    if multilinear_key in multilinear_var_map:
                        coeff[multilinear_var_map[multilinear_key]] += scale * c
                    else:
                        raise ValueError(f"Multilinear {multilinear_key} not in map")

            else:
                raise ValueError(f"Cannot linearize BinaryOp: {e.op}")

        elif isinstance(e, UnaryOp):
            if e.op == "neg":
                visit(e.operand, -scale)
            elif e.op == "abs":
                aux_col = univariate_var_map.get(id(e))
                if aux_col is not None:
                    coeff[aux_col] += scale
                else:
                    raise ValueError(f"Cannot linearize UnaryOp: {e.op}")
            else:
                raise ValueError(f"Cannot linearize UnaryOp: {e.op}")

        elif isinstance(e, SumExpression):
            op = e.operand
            if isinstance(op, Variable):
                offset = _compute_var_offset(op, model)
                for k in range(op.size):
                    coeff[offset + k] += scale
            else:
                visit(op, scale)

        elif isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term, scale)

        else:
            raise ValueError(f"Cannot linearize {type(e).__name__}: {e}")

    visit(expr, 1.0)
    return coeff, const_acc[0]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_milp_relaxation(
    model: Model,
    terms: NonlinearTerms,
    disc_state: DiscretizationState,
    incumbent: Optional[np.ndarray] = None,
    oa_cuts: Optional[list] = None,
    convhull_formulation: str = "disaggregated",
    convhull_ebd: bool = False,
    convhull_ebd_encoding: str = "gray",
    bound_override: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple["MilpRelaxationModel", dict]:
    """Build a MILP relaxation with piecewise McCormick for bilinear/monomial terms.

    For each bilinear term x_i*x_j: adds standard McCormick envelope constraints
    (4 linear inequalities).  These give the convex hull of the bilinear set on the
    bounding box and are independent of the partition (piecewise refinement via binary
    variables is left for future enhancement).

    For each monomial x_i^n (currently n=2 handled precisely):
    - Piecewise tangent underestimators (one per partition interval midpoint) — gets
      tighter as disc_state gains more intervals.
    - Global secant overestimator — bounds s from above.

    The LP objective and constraints are obtained by substituting auxiliary vars for
    all nonlinear terms.

    Parameters
    ----------
    model : Model
    terms : NonlinearTerms
        Output of classify_nonlinear_terms(model).
    disc_state : DiscretizationState
        Current partition; provides intervals for tangent cut placement.
    incumbent : np.ndarray, optional
        Current best NLP solution (flat).  Used to add OA tangent cuts for
        general nonlinear terms; currently unused (reserved for future use).
    convhull_formulation : str, default "disaggregated"
        Piecewise bilinear formulation. ``"disaggregated"`` keeps the existing
        xbar/wbar construction; ``"sos2"`` and ``"facet"`` use a λ-based
        convex-hull reformulation similar to Alpine.jl.
    convhull_ebd : bool, default False
        Replace SOS2 interval binaries with a logarithmic embedded encoding.
        Only supported with ``convhull_formulation="sos2"`` or ``"lambda"``.
    convhull_ebd_encoding : str, default "gray"
        Embedded encoding scheme. ``"gray"`` is the Alpine-style default and
        the only option that remains SOS2-compatible for arbitrary partition
        counts. ``"binary"`` is only valid for two partitions.

    Returns
    -------
    (MilpRelaxationModel, varmap)
        MilpRelaxationModel has a .solve() method returning MilpRelaxationResult.
        varmap maps auxiliary variable keys to MILP column indices.
    """
    if bound_override is None:
        flat_lb, flat_ub = flat_variable_bounds(model)
    else:
        flat_lb = np.asarray(bound_override[0], dtype=np.float64)
        flat_ub = np.asarray(bound_override[1], dtype=np.float64)
    n_orig = len(flat_lb)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)
    if convhull_ebd and convhull_mode != "sos2":
        raise ValueError(
            "convhull_ebd is only supported with convhull_formulation='sos2' or its 'lambda' alias."
        )

    # ── Assign MILP column indices ──────────────────────────────────────────
    # Original variables keep columns 0..n_orig-1. Additional columns are created
    # for lifted bilinear, trilinear, and monomial terms plus any piecewise binaries.
    bilinear_var_map: dict[tuple[int, int], int] = {}
    trilinear_var_map: dict[tuple[int, int, int], int] = {}
    trilinear_stage_map: dict[tuple[int, int, int], dict[str, object]] = {}
    multilinear_var_map: dict[tuple[int, ...], int] = {}
    multilinear_stage_map: dict[tuple[int, ...], list[dict[str, int]]] = {}
    monomial_var_map: dict[tuple[int, int], int] = {}
    univariate_var_map: dict[int, int] = {}

    col_idx = n_orig
    all_bounds: list[tuple[float, float]] = list(zip(flat_lb.tolist(), flat_ub.tolist()))
    integrality_flags: list[int] = []
    for v in model._variables:
        flag = 1 if v.var_type in (VarType.BINARY, VarType.INTEGER) else 0
        integrality_flags.extend([flag] * v.size)

    bilinear_relation_map: dict[tuple[int, int], int] = {}

    def _ensure_bilinear_aux(lhs_col: int, rhs_col: int) -> int:
        nonlocal col_idx
        key = (min(lhs_col, rhs_col), max(lhs_col, rhs_col))
        if key in bilinear_relation_map:
            return bilinear_relation_map[key]

        lhs_lb, lhs_ub = all_bounds[key[0]]
        rhs_lb, rhs_ub = all_bounds[key[1]]
        corners = [
            lhs_lb * rhs_lb,
            lhs_lb * rhs_ub,
            lhs_ub * rhs_lb,
            lhs_ub * rhs_ub,
        ]
        bilinear_relation_map[key] = col_idx
        all_bounds.append((min(corners), max(corners)))
        integrality_flags.append(0)
        col_idx += 1
        return bilinear_relation_map[key]

    def _ensure_multilinear_aux(term: tuple[int, ...]) -> tuple[int, list[dict[str, int]]]:
        ordered = tuple(sorted(term))
        if len(ordered) < 2:
            raise ValueError("multilinear terms require at least two variables")

        stages: list[dict[str, int]] = []
        current_col = ordered[0]
        for rhs_col in ordered[1:]:
            lhs_col = current_col
            product_col = _ensure_bilinear_aux(lhs_col, rhs_col)
            stages.append(
                {
                    "lhs_col": lhs_col,
                    "rhs_col": rhs_col,
                    "product_col": product_col,
                }
            )
            current_col = product_col
        return current_col, stages

    original_bilinear_keys = sorted({(min(i, j), max(i, j)) for i, j in terms.bilinear})
    for key in original_bilinear_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

    partitioned_vars = set(disc_state.partitions)
    trilinear_terms: list[tuple[int, int, int]] = []
    for term in terms.trilinear:
        ordered = sorted(term)
        canonical = (ordered[0], ordered[1], ordered[2])
        if canonical not in trilinear_terms:
            trilinear_terms.append(canonical)

    for term in sorted(trilinear_terms):
        pair, remaining = _choose_trilinear_pair(term, partitioned_vars)
        pair_col = _ensure_bilinear_aux(*pair)
        final_col = _ensure_bilinear_aux(pair_col, remaining)
        trilinear_var_map[term] = final_col
        trilinear_stage_map[term] = {
            "pair": pair,
            "pair_col": pair_col,
            "remaining_var": remaining,
            "product_col": final_col,
        }

    multilinear_terms = terms.multilinear or _collect_distinct_multilinear_products(model)
    for multi_term in multilinear_terms:
        final_col, stages = _ensure_multilinear_aux(multi_term)
        multilinear_var_map[multi_term] = final_col
        multilinear_stage_map[multi_term] = stages

    for var_idx, n in terms.monomial:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        vals = [lb_i**n, ub_i**n]
        if n % 2 == 0 and lb_i < 0 < ub_i:
            vals.append(0.0)
        monomial_var_map[(var_idx, n)] = col_idx
        all_bounds.append((min(vals), max(vals)))
        integrality_flags.append(0)
        col_idx += 1

    monomial_pw_map: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for var_idx, n in terms.monomial:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if (
            var_idx not in disc_state.partitions
            or not _power_is_convex_on_box(n, lb_i)
            or not np.isfinite(lb_i)
            or not np.isfinite(ub_i)
        ):
            continue

        breakpoints = _monomial_breakpoints(var_idx, lb_i, ub_i, disc_state)
        if len(breakpoints) < 3:
            continue

        monomial_intervals: list[tuple[int, float, float]] = []
        for a_k, b_k in zip(breakpoints[:-1], breakpoints[1:]):
            if b_k <= a_k:
                continue
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            monomial_intervals.append((delta_col, float(a_k), float(b_k)))

        if monomial_intervals:
            monomial_pw_map[(var_idx, n)] = monomial_intervals

    univariate_relaxations, univariate_var_map, univariate_bounds = _collect_univariate_relaxations(
        model,
        n_orig,
        flat_lb,
        flat_ub,
        col_idx,
    )
    for val_bounds in univariate_bounds:
        all_bounds.append(val_bounds)
        integrality_flags.append(0)
        col_idx += 1

    bilinear_pw_map: dict[tuple[int, int], list] = {}
    bilinear_lambda_map: dict[tuple[int, int], dict] = {}

    for (lhs_col, rhs_col), _w_col in bilinear_relation_map.items():
        part_var: Optional[int] = None
        if lhs_col < n_orig and lhs_col in disc_state.partitions:
            part_var = lhs_col
            other_var = rhs_col
        elif rhs_col < n_orig and rhs_col in disc_state.partitions:
            part_var = rhs_col
            other_var = lhs_col
        else:
            continue

        pts = disc_state.partitions[part_var]
        other_lb, other_ub = all_bounds[other_var]

        if convhull_mode == "disaggregated":
            intervals = []
            for k in range(len(pts) - 1):
                a_k = float(pts[k])
                b_k = float(pts[k + 1])
                _, wk_lo, wk_hi = _piecewise_product_bounds(
                    a_k,
                    b_k,
                    float(other_lb),
                    float(other_ub),
                )

                delta_col = col_idx
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(1)
                col_idx += 1

                xbar_col = col_idx
                all_bounds.append((min(a_k, 0.0), max(abs(a_k), abs(b_k))))
                integrality_flags.append(0)
                col_idx += 1

                wbar_col = col_idx
                all_bounds.append((min(wk_lo, 0.0), max(wk_hi, 0.0)))
                integrality_flags.append(0)
                col_idx += 1

                intervals.append((delta_col, xbar_col, wbar_col, a_k, b_k))

            bilinear_pw_map[(lhs_col, rhs_col)] = intervals
        else:
            breakpoints = [float(p) for p in pts]
            lambda_cols: list[int] = []
            alpha_cols: list[int] = []
            theta_cols: list[int] = []
            embedding_cols: list[int] = []
            embedding_info: Optional[EmbeddingMap] = None
            theta_lb = min(0.0, float(other_lb), float(other_ub))
            theta_ub = max(0.0, float(other_lb), float(other_ub))

            for _ in breakpoints:
                lambda_cols.append(col_idx)
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(0)
                col_idx += 1

            if convhull_mode == "sos2" and convhull_ebd and len(breakpoints) > 2:
                embedding_info = build_embedding_map(
                    len(breakpoints),
                    encoding=convhull_ebd_encoding,
                )
                for _ in range(embedding_info["bit_count"]):
                    embedding_cols.append(col_idx)
                    all_bounds.append((0.0, 1.0))
                    integrality_flags.append(1)
                    col_idx += 1
            else:
                for _ in range(len(breakpoints) - 1):
                    alpha_cols.append(col_idx)
                    all_bounds.append((0.0, 1.0))
                    integrality_flags.append(1)
                    col_idx += 1

            for _ in breakpoints:
                theta_cols.append(col_idx)
                all_bounds.append((theta_lb, theta_ub))
                integrality_flags.append(0)
                col_idx += 1

            bilinear_lambda_map[(lhs_col, rhs_col)] = {
                "part_var": part_var,
                "other_var": other_var,
                "breakpoints": breakpoints,
                "lambda_cols": lambda_cols,
                "alpha_cols": alpha_cols,
                "theta_cols": theta_cols,
                "embedding_cols": embedding_cols,
                "embedding_info": embedding_info,
                "mode": convhull_mode,
            }

    n_total = col_idx

    # ── Constraint rows (A_ub @ z ≤ b_ub) ───────────────────────────────────
    A_data: list[float] = []
    A_row_indices: list[int] = []
    A_col_indices: list[int] = []
    b_rows: list[float] = []

    def _add_row(coeff: np.ndarray, rhs: float) -> None:
        coeff_arr = np.asarray(coeff, dtype=np.float64).ravel()
        row_idx = len(b_rows)
        nz = np.flatnonzero(coeff_arr)
        if nz.size:
            A_row_indices.extend([row_idx] * int(nz.size))
            A_col_indices.extend(nz.tolist())
            A_data.extend(coeff_arr[nz].tolist())
        b_rows.append(float(rhs))

    # McCormick constraints for each lifted bilinear relation
    for (i, j), w_col in bilinear_relation_map.items():
        xi_lb_g, xi_ub_g = [float(v) for v in all_bounds[i]]
        xj_lb_g, xj_ub_g = [float(v) for v in all_bounds[j]]

        if (i, j) in bilinear_lambda_map:
            lambda_info = bilinear_lambda_map[(i, j)]
            part_var = int(lambda_info["part_var"])
            other_var = int(lambda_info["other_var"])
            breakpoints = list(lambda_info["breakpoints"])
            lambda_cols = list(lambda_info["lambda_cols"])
            alpha_cols = list(lambda_info["alpha_cols"])
            theta_cols = list(lambda_info["theta_cols"])
            embedding_cols = list(lambda_info.get("embedding_cols", []))
            embedding_info = lambda_info.get("embedding_info")
            mode = str(lambda_info["mode"])
            yj_lb, yj_ub = [float(v) for v in all_bounds[other_var]]

            row_sum_lambda = np.zeros(n_total)
            for lambda_col in lambda_cols:
                row_sum_lambda[lambda_col] = -1.0
            _add_row(row_sum_lambda, -1.0)
            _add_row(-row_sum_lambda, 1.0)

            if alpha_cols:
                row_sum_alpha = np.zeros(n_total)
                for alpha_col in alpha_cols:
                    row_sum_alpha[alpha_col] = -1.0
                _add_row(row_sum_alpha, -1.0)
                _add_row(-row_sum_alpha, 1.0)

            row_x = np.zeros(n_total)
            row_x[part_var] = 1.0
            for p_j, lambda_col in zip(breakpoints, lambda_cols):
                row_x[lambda_col] -= float(p_j)
            _add_row(row_x, 0.0)
            _add_row(-row_x, 0.0)

            row_y = np.zeros(n_total)
            row_y[other_var] = 1.0
            for theta_col in theta_cols:
                row_y[theta_col] -= 1.0
            _add_row(row_y, 0.0)
            _add_row(-row_y, 0.0)

            row_w = np.zeros(n_total)
            row_w[w_col] = 1.0
            for p_j, theta_col in zip(breakpoints, theta_cols):
                row_w[theta_col] -= float(p_j)
            _add_row(row_w, 0.0)
            _add_row(-row_w, 0.0)

            if mode == "sos2":
                assert alpha_cols or embedding_cols, (
                    "Expected either alpha or embedding columns for SOS2 linking"
                )

            if mode == "sos2" and embedding_info is not None:
                for bit_col, positive_set, negative_set in zip(
                    embedding_cols,
                    embedding_info["positive_sets"],
                    embedding_info["negative_sets"],
                ):
                    row = np.zeros(n_total)
                    for lambda_idx in positive_set:
                        row[lambda_cols[lambda_idx]] = 1.0
                    row[bit_col] = -1.0
                    _add_row(row, 0.0)

                    row = np.zeros(n_total)
                    for lambda_idx in negative_set:
                        row[lambda_cols[lambda_idx]] = 1.0
                    row[bit_col] = 1.0
                    _add_row(row, 1.0)
            elif mode == "sos2":
                for idx, lambda_col in enumerate(lambda_cols):
                    row = np.zeros(n_total)
                    row[lambda_col] = 1.0
                    if idx == 0:
                        row[alpha_cols[0]] = -1.0
                    elif idx == len(lambda_cols) - 1:
                        row[alpha_cols[-1]] = -1.0
                    else:
                        row[alpha_cols[idx - 1]] = -1.0
                        row[alpha_cols[idx]] = -1.0
                    _add_row(row, 0.0)
            else:
                for idx in range(len(alpha_cols) - 1):
                    row = np.zeros(n_total)
                    for alpha_col in alpha_cols[: idx + 1]:
                        row[alpha_col] -= 1.0
                    for lambda_col in lambda_cols[: idx + 1]:
                        row[lambda_col] += 1.0
                    _add_row(row, 0.0)

                    row = np.zeros(n_total)
                    for alpha_col in alpha_cols[: idx + 1]:
                        row[alpha_col] += 1.0
                    for lambda_col in lambda_cols[: idx + 2]:
                        row[lambda_col] -= 1.0
                    _add_row(row, 0.0)

            for lambda_col, theta_col in zip(lambda_cols, theta_cols):
                row = np.zeros(n_total)
                row[theta_col] = -1.0
                row[lambda_col] = yj_lb
                _add_row(row, 0.0)

                row = np.zeros(n_total)
                row[theta_col] = -1.0
                row[other_var] = 1.0
                row[lambda_col] = yj_ub
                _add_row(row, yj_ub)

                row = np.zeros(n_total)
                row[theta_col] = 1.0
                row[other_var] = -1.0
                row[lambda_col] = -yj_lb
                _add_row(row, -yj_lb)

                row = np.zeros(n_total)
                row[theta_col] = 1.0
                row[lambda_col] = -yj_ub
                _add_row(row, 0.0)

        elif (i, j) in bilinear_pw_map and bilinear_pw_map[(i, j)]:
            # ── Piecewise McCormick with binary partition selection ──────────
            intervals = bilinear_pw_map[(i, j)]
            if i < n_orig and i in disc_state.partitions:
                part_var, other_var = i, j
            else:
                part_var, other_var = j, i

            yj_lb, yj_ub = [float(v) for v in all_bounds[other_var]]

            # Constraint: Σ δ_k = 1 (select exactly one partition)
            row_sum = np.zeros(n_total)
            for delta_col, _, _, _, _ in intervals:
                row_sum[delta_col] = -1.0
            _add_row(row_sum, -1.0)  # -Σδ_k ≤ -1
            _add_row(-row_sum, 1.0)  # Σδ_k ≤ 1

            # Constraint: x_part = Σ x̄_k (reconstruct partition variable)
            row_recon = np.zeros(n_total)
            row_recon[part_var] = 1.0
            for _, xbar_col, _, _, _ in intervals:
                row_recon[xbar_col] = -1.0
            _add_row(row_recon, 0.0)  # x_part - Σ x̄_k ≤ 0
            _add_row(-row_recon, 0.0)  # -(x_part - Σ x̄_k) ≤ 0

            # Constraint: w = Σ w̄_k
            row_wsum = np.zeros(n_total)
            row_wsum[w_col] = 1.0
            for _, _, wbar_col, _, _ in intervals:
                row_wsum[wbar_col] = -1.0
            _add_row(row_wsum, 0.0)
            _add_row(-row_wsum, 0.0)

            for delta_col, xbar_col, wbar_col, a_k, b_k in intervals:
                corners, wk_lo, wk_hi = _piecewise_product_bounds(
                    a_k,
                    b_k,
                    yj_lb,
                    yj_ub,
                )
                M_k = _compute_piecewise_big_m(corners)

                # x̄_k ≥ a_k * δ_k  (x̄_k is in [a_k, b_k] when δ_k=1)
                row = np.zeros(n_total)
                row[xbar_col] = -1.0
                row[delta_col] = a_k
                _add_row(row, 0.0)  # -x̄_k + a_k*δ_k ≤ 0  → x̄_k ≥ a_k*δ_k

                # x̄_k ≤ b_k * δ_k
                row = np.zeros(n_total)
                row[xbar_col] = 1.0
                row[delta_col] = -b_k
                _add_row(row, 0.0)

                # w̄_k ≤ wk_hi * δ_k  → w̄_k=0 when δ_k=0
                # This forces the bilinear product to 0 when interval k is inactive.
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[delta_col] = -wk_hi
                _add_row(row, 0.0)

                # w̄_k ≥ wk_lo * δ_k  → w̄_k=0 when δ_k=0 (for wk_lo ≥ 0 case)
                if wk_lo > 0:
                    row = np.zeros(n_total)
                    row[wbar_col] = -1.0
                    row[delta_col] = wk_lo
                    _add_row(row, 0.0)

                # Per-interval McCormick with big-M relaxation.
                # The big-M term LOOSENS the constraint when δ_k=0 (interval inactive).
                #
                # cv1: w̄_k ≥ a_k*y + x̄_k*y_lb - a_k*y_lb - M*(1-δ_k)
                #   → -w̄_k + a_k*y + x̄_k*y_lb + M*δ_k ≤ a_k*y_lb + M
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[other_var] += a_k
                row[xbar_col] += yj_lb
                row[delta_col] = M_k  # +M_k so constraint loosens when δ_k=0
                _add_row(row, a_k * yj_lb + M_k)

                # cv2: w̄_k ≥ b_k*y + x̄_k*y_ub - b_k*y_ub - M*(1-δ_k)
                #   → -w̄_k + b_k*y + x̄_k*y_ub + M*δ_k ≤ b_k*y_ub + M
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[other_var] += b_k
                row[xbar_col] += yj_ub
                row[delta_col] = M_k
                _add_row(row, b_k * yj_ub + M_k)

                # cc1: w̄_k ≤ b_k*y + x̄_k*y_lb - b_k*y_lb + M*(1-δ_k)
                #   → w̄_k - b_k*y - x̄_k*y_lb + M*δ_k ≤ M - b_k*y_lb
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[other_var] -= b_k
                row[xbar_col] -= yj_lb
                row[delta_col] = M_k  # +M_k so constraint loosens when δ_k=0
                _add_row(row, M_k - b_k * yj_lb)

                # cc2: w̄_k ≤ a_k*y + x̄_k*y_ub - a_k*y_ub + M*(1-δ_k)
                #   → w̄_k - a_k*y - x̄_k*y_ub + M*δ_k ≤ M - a_k*y_ub
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[other_var] -= a_k
                row[xbar_col] -= yj_ub
                row[delta_col] = M_k
                _add_row(row, M_k - a_k * yj_ub)

        else:
            # ── Standard (global) McCormick ──────────────────────────────────
            # cv1: w ≥ xi_lb*xj + xi*xj_lb - xi_lb*xj_lb
            #   →  -w + xj_lb*xi + xi_lb*xj ≤ xi_lb*xj_lb
            row = np.zeros(n_total)
            row[w_col] = -1.0
            row[i] += xj_lb_g
            row[j] += xi_lb_g
            _add_row(row, xi_lb_g * xj_lb_g)

            # cv2: w ≥ xi_ub*xj + xi*xj_ub - xi_ub*xj_ub
            row = np.zeros(n_total)
            row[w_col] = -1.0
            row[i] += xj_ub_g
            row[j] += xi_ub_g
            _add_row(row, xi_ub_g * xj_ub_g)

            # cc1: w ≤ xi_ub*xj + xi*xj_lb - xi_ub*xj_lb
            row = np.zeros(n_total)
            row[w_col] = 1.0
            row[i] -= xj_lb_g
            row[j] -= xi_ub_g
            _add_row(row, -xi_ub_g * xj_lb_g)

            # cc2: w ≤ xi_lb*xj + xi*xj_ub - xi_lb*xj_ub
            row = np.zeros(n_total)
            row[w_col] = 1.0
            row[i] -= xj_ub_g
            row[j] -= xi_lb_g
            _add_row(row, -xi_lb_g * xj_ub_g)

    # Binary interval selectors for partitioned convex monomial overestimators.
    # A local secant is valid only on its own interval, so the selector links the
    # original variable to one active interval before applying that secant.
    for (var_idx, n), monomial_intervals in monomial_pw_map.items():
        if not monomial_intervals:
            continue
        s_col = monomial_var_map[(var_idx, n)]
        x_lb, x_ub = [float(v) for v in all_bounds[var_idx]]
        _s_lb, s_ub = [float(v) for v in all_bounds[s_col]]

        row_sum = np.zeros(n_total)
        for delta_col, _, _ in monomial_intervals:
            row_sum[delta_col] = -1.0
        _add_row(row_sum, -1.0)
        _add_row(-row_sum, 1.0)

        for delta_col, a_k, b_k in monomial_intervals:
            lower_m = max(0.0, a_k - x_lb)
            row = np.zeros(n_total)
            row[var_idx] = -1.0
            row[delta_col] = lower_m
            _add_row(row, lower_m - a_k)

            upper_m = max(0.0, x_ub - b_k)
            row = np.zeros(n_total)
            row[var_idx] = 1.0
            row[delta_col] = upper_m
            _add_row(row, b_k + upper_m)

            slope, intercept = _power_secant_line(a_k, b_k, n)
            line_at_lb = slope * x_lb + intercept
            line_at_ub = slope * x_ub + intercept
            line_min = min(line_at_lb, line_at_ub)
            secant_m = max(0.0, s_ub - line_min)

            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -slope
            row[delta_col] = secant_m
            _add_row(row, intercept + secant_m)

    # Monomial constraints
    for var_idx, n in terms.monomial:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        s_col = monomial_var_map[(var_idx, n)]
        breakpoints = _monomial_breakpoints(var_idx, lb_i, ub_i, disc_state)

        def _add_under_tangent(t: float) -> None:
            slope, intercept = _power_tangent_line(t, n)
            row = np.zeros(n_total)
            row[s_col] = -1.0
            row[var_idx] = slope
            _add_row(row, -intercept)

        def _add_over_tangent(t: float) -> None:
            slope, intercept = _power_tangent_line(t, n)
            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -slope
            _add_row(row, intercept)

        def _add_under_secant(a: float, b: float) -> None:
            slope, intercept = _power_secant_line(a, b, n)
            row = np.zeros(n_total)
            row[s_col] = -1.0
            row[var_idx] = slope
            _add_row(row, -intercept)

        def _add_over_secant(a: float, b: float) -> None:
            slope, intercept = _power_secant_line(a, b, n)
            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -slope
            _add_row(row, intercept)

        if _power_is_convex_on_box(n, lb_i):
            # Convex on the full domain: tangents underestimate and the secant
            # overestimates. Using all breakpoints makes the relaxation tighten
            # monotonically as the partition is refined.
            for t in breakpoints:
                _add_under_tangent(t)
            _add_over_secant(lb_i, ub_i)
        elif ub_i <= 0.0:
            # Concave on the full domain: the secant underestimates and tangents
            # overestimate.
            _add_under_secant(lb_i, ub_i)
            for t in breakpoints:
                _add_over_tangent(t)
        else:
            # Mixed-sign odd powers change curvature at zero. Keep only tangents that
            # are globally valid on the current box so the relaxation remains sound.
            for t in breakpoints:
                if _odd_mixed_tangent_is_valid(t, lb_i, ub_i, n, "under"):
                    _add_under_tangent(t)
                if _odd_mixed_tangent_is_valid(t, lb_i, ub_i, n, "over"):
                    _add_over_tangent(t)

    # Supported univariate operator graph relaxations.
    def _add_lower_line(relax: UnivariateRelaxation, slope: float, intercept: float) -> None:
        """Add t >= slope * arg + intercept."""
        row = np.zeros(n_total)
        row[:n_orig] = slope * relax.arg_coeff
        row[relax.aux_col] = -1.0
        _add_row(row, -intercept - slope * relax.arg_const)

    def _add_upper_line(relax: UnivariateRelaxation, slope: float, intercept: float) -> None:
        """Add t <= slope * arg + intercept."""
        row = np.zeros(n_total)
        row[:n_orig] = -slope * relax.arg_coeff
        row[relax.aux_col] = 1.0
        _add_row(row, intercept + slope * relax.arg_const)

    def _add_aux_equality(relax: UnivariateRelaxation, coeff: np.ndarray, rhs: float) -> None:
        """Add equality t + coeff @ x = rhs as two inequality rows."""
        row = np.zeros(n_total)
        row[:n_orig] = coeff
        row[relax.aux_col] = 1.0
        _add_row(row, rhs)
        _add_row(-row, -rhs)

    for relax in univariate_relaxations:
        lb_u = relax.arg_lb
        ub_u = relax.arg_ub
        if abs(ub_u - lb_u) <= 1e-12:
            val = _univariate_value(relax.func_name, lb_u)
            row = np.zeros(n_total)
            row[relax.aux_col] = 1.0
            _add_row(row, val)
            _add_row(-row, -val)
            continue

        if relax.func_name == "abs":
            if lb_u >= 0.0:
                # t = arg
                _add_aux_equality(relax, -relax.arg_coeff, relax.arg_const)
            elif ub_u <= 0.0:
                # t = -arg
                _add_aux_equality(relax, relax.arg_coeff, -relax.arg_const)
            else:
                # t >= arg, t >= -arg, and t below the endpoint secant.
                _add_lower_line(relax, 1.0, 0.0)
                _add_lower_line(relax, -1.0, 0.0)
                f_lb = abs(lb_u)
                f_ub = abs(ub_u)
                slope = (f_ub - f_lb) / (ub_u - lb_u)
                intercept = f_lb - slope * lb_u
                _add_upper_line(relax, slope, intercept)
            continue

        f_lb = _univariate_value(relax.func_name, lb_u)
        f_ub = _univariate_value(relax.func_name, ub_u)
        secant_slope = (f_ub - f_lb) / (ub_u - lb_u)
        secant_intercept = f_lb - secant_slope * lb_u

        if relax.func_name == "exp":
            # Convex: tangents are lower bounds; secant is an upper bound.
            for pt in _tangent_points(relax.func_name, lb_u, ub_u):
                slope = _univariate_grad(relax.func_name, pt)
                intercept = _univariate_value(relax.func_name, pt) - slope * pt
                _add_lower_line(relax, slope, intercept)
            _add_upper_line(relax, secant_slope, secant_intercept)
        else:
            # log/log2/log10/sqrt are concave on their supported domains:
            # secant is a lower bound; tangents are upper bounds.
            _add_lower_line(relax, secant_slope, secant_intercept)
            for pt in _tangent_points(relax.func_name, lb_u, ub_u):
                slope = _univariate_grad(relax.func_name, pt)
                intercept = _univariate_value(relax.func_name, pt) - slope * pt
                _add_upper_line(relax, slope, intercept)

    # Model constraints
    for constraint in model._constraints:
        body = constraint.body  # normalized: body <= 0  (sense is always "<=")
        sense = constraint.sense
        try:
            c, const = _linearize_expr(
                body,
                model,
                bilinear_var_map,
                trilinear_var_map,
                multilinear_var_map,
                monomial_var_map,
                univariate_var_map,
                n_total,
            )
            # body ≤ 0  →  c @ z + const ≤ 0  →  c @ z ≤ -const
            if sense == "<=":
                _add_row(c, -const)
            elif sense == "==":
                _add_row(c, -const)
                _add_row(-c, const)
            # (">=" is normalized to "<=" by the Expression operators)
        except ValueError as err:
            # Constraint contains terms we can't linearize (e.g. general nonlinear).
            # Omitting it makes the LP feasible region larger → still a valid lower bound.
            logger.debug("AMP: omitting constraint (cannot linearize): %s", err)

    # ── OA tangent cuts from NLP incumbent ──────────────────────────────────
    # These are outer-approximation linearizations of the original nonlinear
    # constraints at the incumbent point.  They are in terms of ORIGINAL
    # variables (columns 0..n_orig-1) and tighten the LP relaxation.
    if oa_cuts:
        for coeff, rhs in oa_cuts:
            row = np.zeros(n_total)
            row[: len(coeff)] = coeff[: n_total if len(coeff) > n_total else len(coeff)]
            _add_row(row, rhs)

    # ── Objective ────────────────────────────────────────────────────────────
    assert model._objective is not None
    obj_expr = model._objective.expression
    try:
        c_obj, const_obj = _linearize_expr(
            obj_expr,
            model,
            bilinear_var_map,
            trilinear_var_map,
            multilinear_var_map,
            monomial_var_map,
            univariate_var_map,
            n_total,
        )
        objective_bound_valid = True
    except ValueError:
        # Keep a feasibility objective so the relaxation can still produce a point,
        # but do not treat the resulting LP value as a sound global bound.
        c_obj = np.zeros(n_total)
        const_obj = 0.0
        objective_bound_valid = False
        logger.debug("AMP: objective is not linearizable; MILP relaxation bound is unavailable")

    # Negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_obj = -c_obj
        const_obj = -const_obj

    # ── Assemble and return ──────────────────────────────────────────────────
    if b_rows:
        A_ub_arr = sp.csr_matrix(
            (A_data, (A_row_indices, A_col_indices)),
            shape=(len(b_rows), n_total),
            dtype=np.float64,
        )
        b_ub_arr = np.array(b_rows, dtype=np.float64)
    else:
        A_ub_arr = None
        b_ub_arr = None

    # Build integrality array (1 = integer, 0 = continuous)
    integrality_arr = np.array(integrality_flags, dtype=np.int32)
    has_integers = bool(np.any(integrality_arr > 0))

    milp_model = MilpRelaxationModel(
        c=c_obj,
        A_ub=A_ub_arr,
        b_ub=b_ub_arr,
        bounds=all_bounds,
        obj_offset=const_obj,
        integrality=integrality_arr if has_integers else None,
        objective_bound_valid=objective_bound_valid,
    )

    varmap: dict = {
        "original": {k: k for k in range(n_orig)},
        "bilinear": bilinear_var_map,
        "trilinear": trilinear_var_map,
        "trilinear_stages": trilinear_stage_map,
        "multilinear": multilinear_var_map,
        "multilinear_stages": multilinear_stage_map,
        "monomial": monomial_var_map,
        "monomial_pw": monomial_pw_map,
        "univariate": univariate_var_map,
        "univariate_relaxations": univariate_relaxations,
        "bilinear_pw": bilinear_pw_map,
        "bilinear_lambda": bilinear_lambda_map,
        "convhull_formulation": convhull_mode,
        "convhull_ebd": convhull_ebd,
        "convhull_ebd_encoding": convhull_ebd_encoding,
    }

    return milp_model, varmap
