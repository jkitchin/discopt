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

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt._jax._numeric import EFFECTIVE_INF as _EFFECTIVE_INF
from discopt._jax._numeric import is_effectively_finite as _is_effectively_finite
from discopt._jax.discretization import DiscretizationState
from discopt._jax.embedding import EmbeddingMap, build_embedding_map
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.operator_relaxations import (
    critical_points_in_interval as _critical_points_in_interval,
)
from discopt._jax.operator_relaxations import tan_range as _tan_range
from discopt._jax.operator_relaxations import trig_range as _trig_range
from discopt._jax.operator_relaxations import trig_square_curvature as _trig_square_curvature
from discopt._jax.operator_relaxations import (
    trig_square_grad as _trig_square_grad,
)
from discopt._jax.operator_relaxations import trig_square_range as _trig_square_range
from discopt._jax.operator_relaxations import trig_square_value as _trig_square_value
from discopt._jax.term_classifier import (
    NonlinearTerms,
    _compute_var_offset,
    _get_flat_index,
    distribute_products,
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

# Dedupe identical warnings emitted across repeated relaxation builds (AMP iterates).
_warned_messages: set[str] = set()
_MAX_INTEGER_COS_ENUM = 10000
_MAX_FINITE_EXP_ARG = float(np.log(np.finfo(np.float64).max))
_MAX_TRIG_PIECEWISE_SPAN = 2.0 * math.pi
_MAX_TRIG_PIECEWISE_INTERVALS = 32
_MAX_TRIG_IMPORTED_BREAKPOINTS = _MAX_TRIG_PIECEWISE_INTERVALS + 1
_MAX_TRIG_PIECEWISE_WIDTH = math.pi / 6.0
_MAX_RELAXATION_PARTITION_INTERVALS = 128
_MAX_OBJECTIVE_LIFT_POWER = 6
_MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES = 256


def _warn_once(msg: str, *args) -> None:
    formatted = msg % args if args else msg
    if formatted in _warned_messages:
        return
    _warned_messages.add(formatted)
    logger.warning("%s", formatted)


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


@dataclass
class PiecewiseUnivariateInterval:
    """Binary-selected interval for a mixed-curvature univariate relaxation."""

    delta_col: int
    lb: float
    ub: float
    curvature: Optional[str]


@dataclass
class PiecewiseUnivariateRelaxation:
    """Partition-aware relaxation for a lifted univariate operator."""

    relax: UnivariateRelaxation
    intervals: list[PiecewiseUnivariateInterval]


@dataclass
class UnivariateSquareRelaxation:
    """Lifted square of a supported univariate auxiliary."""

    base_col: int
    aux_col: int
    base_lb: float
    base_ub: float


@dataclass
class PiecewiseTrigSquareInterval:
    """Binary-selected interval for a direct trig-square relaxation."""

    delta_col: int
    lb: float
    ub: float
    curvature: Optional[str]


@dataclass
class PiecewiseTrigSquareRelaxation:
    """Partition-aware direct relaxation for sin(arg)^2 or cos(arg)^2."""

    square: UnivariateSquareRelaxation
    func_name: str
    arg_coeff: np.ndarray
    arg_const: float
    arg_lb: float
    arg_ub: float
    intervals: list[PiecewiseTrigSquareInterval]


@dataclass
class FiniteDomainTrigSquareTable:
    """Exact selector table for sin(integer_affine)^2 or cos(integer_affine)^2."""

    square: UnivariateSquareRelaxation
    func_name: str
    var_idx: int
    arg_coeff: float
    arg_const: float
    domain_values: list[int]
    trig_values: list[float]
    square_values: list[float]
    selector_cols: list[int]


@dataclass
class MinMaxObjectiveLift:
    """Epigraph/hypograph lift for a supported objective-level min/max call."""

    func_name: str
    aux_col: int
    branch_exprs: tuple[Expression, ...]
    branch_bounds: tuple[tuple[Optional[float], Optional[float]], ...]
    aux_bounds: tuple[float, float]


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


def _constant_value(expr: Expression) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    values = np.asarray(expr.value, dtype=np.float64).ravel()
    if values.size != 1:
        return None
    return float(values[0])


def _finite_bound_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not _is_effectively_finite(value):
        return None
    return value


def _expand_integer_powers_for_relaxation(expr: Expression, model: Model) -> Expression:
    """Expand small integer powers of affine expressions for existing monomial lifts."""

    def visit(node: Expression) -> Expression:
        if isinstance(node, BinaryOp):
            left = visit(node.left)
            right = visit(node.right)
            if node.op == "**":
                exp = _constant_value(right)
                if exp is not None:
                    n = int(exp)
                    if exp == n and 2 <= n <= _MAX_OBJECTIVE_LIFT_POWER:
                        base = left
                        product = base
                        for _ in range(n - 1):
                            product = BinaryOp("*", product, base)
                        return distribute_products(product)
            return distribute_products(BinaryOp(node.op, left, right))
        if isinstance(node, UnaryOp):
            return UnaryOp(node.op, visit(node.operand))
        if isinstance(node, SumExpression):
            return SumExpression(visit(node.operand), axis=node.axis)
        if isinstance(node, SumOverExpression):
            return SumOverExpression([visit(term) for term in node.terms])
        # Preserve FunctionCall object identity so existing univariate lift maps
        # keyed by id(expr) remain usable during branch linearization.
        return node

    return distribute_products(visit(expr))


def _expression_lower_bound_for_lift(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    expanded = _expand_integer_powers_for_relaxation(expr, model)
    lower = _separable_objective_lower_bound(expanded, model, flat_lb, flat_ub)
    return _finite_bound_or_none(lower)


def _expression_upper_bound_for_lift(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    lower_of_negated = _expression_lower_bound_for_lift(
        UnaryOp("neg", expr),
        model,
        flat_lb,
        flat_ub,
    )
    if lower_of_negated is None:
        return None
    return -lower_of_negated


def _collect_monomial_terms_for_lift(expr: Expression, model: Model) -> set[tuple[int, int]]:
    terms: set[tuple[int, int]] = set()

    def visit(node: Expression) -> None:
        if isinstance(node, BinaryOp):
            if node.op == "*":
                decomp = _decompose_product(node, model)
                if decomp is not None:
                    _scalar, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if len(unique) == 1 and len(indices) >= 2:
                        terms.add((unique[0], len(indices)))
            elif node.op == "**":
                flat = _get_flat_index(node.left, model)
                exp = _constant_value(node.right)
                if flat is not None and exp is not None:
                    n = int(exp)
                    if exp == n and n >= 2:
                        terms.add((flat, n))
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, UnaryOp):
            visit(node.operand)
            return
        if isinstance(node, FunctionCall):
            for arg in node.args:
                visit(arg)
            return
        if isinstance(node, IndexExpression) and not isinstance(node.base, Variable):
            visit(node.base)
            return
        if isinstance(node, SumExpression):
            visit(node.operand)
            return
        if isinstance(node, SumOverExpression):
            for term in node.terms:
                visit(term)

    visit(expr)
    return terms


def _build_minmax_objective_lift(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[MinMaxObjectiveLift]:
    if model._objective is None:
        return None
    expr = model._objective.expression
    if not isinstance(expr, FunctionCall) or len(expr.args) < 2:
        return None
    if model._objective.sense == ObjectiveSense.MINIMIZE and expr.func_name != "max":
        return None
    if model._objective.sense == ObjectiveSense.MAXIMIZE and expr.func_name != "min":
        return None
    if expr.func_name not in {"min", "max"}:
        return None

    branch_exprs = tuple(_expand_integer_powers_for_relaxation(arg, model) for arg in expr.args)
    branch_bounds = tuple(
        (
            _expression_lower_bound_for_lift(branch, model, flat_lb, flat_ub),
            _expression_upper_bound_for_lift(branch, model, flat_lb, flat_ub),
        )
        for branch in branch_exprs
    )

    lower_bounds = [lb for lb, _ub in branch_bounds if lb is not None]
    upper_bounds = [ub for _lb, ub in branch_bounds if ub is not None]
    aux_lb: Optional[float]
    aux_ub: Optional[float]
    if expr.func_name == "max":
        # max(f_i) is at least any available lower bound on a branch.
        aux_lb = max(lower_bounds) if lower_bounds else None
        # max(f_i) is at most max(ub_i) only when every branch has an upper bound.
        aux_ub = max(upper_bounds) if len(upper_bounds) == len(branch_bounds) else None
        directional_bound = aux_lb
    else:
        # min(f_i) is at least min(lb_i) only when every branch has a lower bound.
        aux_lb = min(lower_bounds) if len(lower_bounds) == len(branch_bounds) else None
        # min(f_i) is at most any available upper bound on a branch.
        aux_ub = min(upper_bounds) if upper_bounds else None
        directional_bound = aux_ub

    directional_bound = _finite_bound_or_none(directional_bound)
    if directional_bound is None:
        return None

    lb = _finite_bound_or_none(aux_lb)
    ub = _finite_bound_or_none(aux_ub)
    aux_bounds = (
        lb if lb is not None else -_EFFECTIVE_INF,
        ub if ub is not None else _EFFECTIVE_INF,
    )
    if aux_bounds[0] > aux_bounds[1] + 1e-9:
        return None
    if aux_bounds[0] > aux_bounds[1]:
        mid = 0.5 * (aux_bounds[0] + aux_bounds[1])
        aux_bounds = (mid, mid)

    return MinMaxObjectiveLift(
        func_name=expr.func_name,
        aux_col=-1,
        branch_exprs=branch_exprs,
        branch_bounds=branch_bounds,
        aux_bounds=aux_bounds,
    )


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


def _decompose_product(
    expr: Expression,
    model: Model,
    fractional_power_var_map: Optional[dict[tuple[int, float], int]] = None,
    univariate_var_map: Optional[dict[object, int]] = None,
) -> tuple[float, list[int]] | None:
    """Decompose a product expression into (scalar, [flat_or_aux_idx, ...]).

    Returns None if expr contains non-constant, non-variable leaves.
    Constants are accumulated into the scalar; variable references and
    registered lifted sub-expressions are appended to the index list (using
    their MILP column indices).
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
        if univariate_var_map:
            aux_col = univariate_var_map.get(id(e))
            if aux_col is not None:
                var_indices.append(aux_col)
                return True
        # Recognize var^p (fractional p) when an aux column was allocated.
        if (
            fractional_power_var_map
            and isinstance(e, BinaryOp)
            and e.op == "**"
            and isinstance(e.right, Constant)
        ):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                key = (base_flat, float(e.right.value))
                if key in fractional_power_var_map:
                    var_indices.append(fractional_power_var_map[key])
                    return True
        return False

    if visit(expr):
        return scalar[0], var_indices
    return None


def _collect_lifted_bilinear_products(
    model: Model,
    fractional_power_var_map: dict[tuple[int, float], int],
    univariate_var_map: dict[object, int],
    n_orig: int,
) -> list[tuple[int, int]]:
    """Return products between original variables and lifted auxiliary columns."""
    keys: set[tuple[int, int]] = set()

    def visit(expr: Expression) -> None:
        if isinstance(expr, BinaryOp):
            if expr.op == "*":
                decomp = _decompose_product(
                    expr,
                    model,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_var_map=univariate_var_map,
                )
                if decomp is not None:
                    _scalar, indices = decomp
                    unique = list(dict.fromkeys(indices))
                    if (
                        len(unique) == 2
                        and len(indices) == 2
                        and any(idx >= n_orig for idx in unique)
                    ):
                        i, j = sorted(unique)
                        keys.add((i, j))
            visit(expr.left)
            visit(expr.right)
            return

        if isinstance(expr, UnaryOp):
            visit(expr.operand)
            return

        if isinstance(expr, FunctionCall):
            for arg in expr.args:
                visit(arg)
            return

        if isinstance(expr, IndexExpression):
            if not isinstance(expr.base, Variable):
                visit(expr.base)
            return

        if isinstance(expr, SumExpression):
            visit(expr.operand)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                visit(term)

    if model._objective is not None:
        visit(distribute_products(model._objective.expression))
    for constraint in model._constraints:
        visit(distribute_products(constraint.body))

    return sorted(keys)


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
        if name in {"sqrt", "log", "log2", "log10", "exp", "abs", "sin", "cos", "tan"}:
            return name, expr.args[0]
    if isinstance(expr, UnaryOp) and expr.op == "abs":
        return "abs", expr.operand
    if isinstance(expr, BinaryOp) and expr.op == "/" and _constant_value(expr.left) is not None:
        return "reciprocal", expr.right
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
    if func_name == "reciprocal":
        return float(1.0 / x)
    if func_name == "sin":
        return float(np.sin(x))
    if func_name == "cos":
        return float(np.cos(x))
    if func_name == "tan":
        return float(np.tan(x))
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
    if func_name == "reciprocal":
        return float(-1.0 / (x * x))
    if func_name == "sin":
        return float(np.cos(x))
    if func_name == "cos":
        return float(-np.sin(x))
    if func_name == "tan":
        c = float(np.cos(x))
        return float(1.0 / (c * c))
    raise ValueError(f"No smooth derivative for univariate function: {func_name}")


def _tan_domain_ok(arg_lb: float, arg_ub: float) -> bool:
    """Return True when ``tan`` is finite and continuous on the interval."""
    if not np.isfinite(arg_lb) or not np.isfinite(arg_ub) or arg_lb > arg_ub:
        return False
    half_pi = 0.5 * np.pi
    k = np.ceil((arg_lb - half_pi) / np.pi)
    asymptote = half_pi + k * np.pi
    if arg_lb <= asymptote <= arg_ub:
        return False
    return all(_is_effectively_finite(np.tan(x)) for x in (arg_lb, arg_ub))


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
        return bool(arg_ub <= _MAX_FINITE_EXP_ARG)
    if func_name == "abs":
        return True
    if func_name == "reciprocal":
        return bool(arg_lb > 0.0)
    if func_name in {"sin", "cos"}:
        return True
    if func_name == "tan":
        return _tan_range(arg_lb, arg_ub) is not None
    return False


def _univariate_value_bounds(func_name: str, arg_lb: float, arg_ub: float) -> tuple[float, float]:
    """Return finite bounds for f(x) on [arg_lb, arg_ub]."""
    if func_name == "abs":
        if arg_lb <= 0.0 <= arg_ub:
            return 0.0, max(abs(arg_lb), abs(arg_ub))
        values = [abs(arg_lb), abs(arg_ub)]
        return min(values), max(values)
    if func_name in {"sin", "cos", "tan"}:
        bounds = _trig_range(func_name, arg_lb, arg_ub)
        if bounds is None:
            return np.nan, np.nan
        return bounds
    values = [_univariate_value(func_name, arg_lb), _univariate_value(func_name, arg_ub)]
    return min(values), max(values)


def _tangent_points(func_name: str, lb: float, ub: float) -> list[float]:
    """Choose deterministic valid tangent points for smooth univariate cuts."""
    raw_points = [lb, 0.5 * (lb + ub), ub]
    points: list[float] = []
    for pt in raw_points:
        if func_name == "sqrt" and pt <= 0.0:
            continue
        if func_name in {"log", "log2", "log10", "reciprocal"} and pt <= 0.0:
            continue
        if func_name == "tan" and not _is_effectively_finite(np.tan(pt)):
            continue
        if not np.isfinite(pt):
            continue
        if all(abs(pt - seen) > 1e-12 for seen in points):
            points.append(float(pt))
    return points


def _univariate_curvature(func_name: str, val_lb: float, val_ub: float) -> Optional[str]:
    """Return certified curvature on the interval, or None for mixed curvature."""
    tol = 1e-12
    if func_name in {"exp", "reciprocal"}:
        return "convex"
    if func_name in {"sqrt", "log", "log2", "log10"}:
        return "concave"
    if func_name in {"sin", "cos"}:
        if val_lb >= -tol:
            return "concave"
        if val_ub <= tol:
            return "convex"
        return None
    if func_name == "tan":
        if val_lb >= -tol:
            return "convex"
        if val_ub <= tol:
            return "concave"
    return None


def _trig_partition_breakpoints(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[float]:
    """Return safe breakpoints for a mixed-curvature trig argument interval."""
    lb = float(relax.arg_lb)
    ub = float(relax.arg_ub)
    if relax.func_name not in {"sin", "cos", "tan"} or not (np.isfinite(lb) and np.isfinite(ub)):
        return [lb, ub]

    points = [lb, ub]
    if relax.func_name == "sin":
        curvature_start, critical_start = 0.0, math.pi / 2.0
        points.extend(_critical_points_in_interval(critical_start, math.pi, lb, ub))
    elif relax.func_name == "cos":
        curvature_start, critical_start = math.pi / 2.0, 0.0
        points.extend(_critical_points_in_interval(critical_start, math.pi, lb, ub))
    else:
        curvature_start = 0.0

    points.extend(_critical_points_in_interval(curvature_start, math.pi, lb, ub))

    nz = np.flatnonzero(np.abs(relax.arg_coeff) > 1e-12)
    if nz.size == 1:
        var_idx = int(nz[0])
        if var_idx < n_orig and var_idx in disc_state.partitions:
            coeff = float(relax.arg_coeff[var_idx])
            partition = np.asarray(disc_state.partitions[var_idx], dtype=np.float64)
            if partition.size <= _MAX_TRIG_IMPORTED_BREAKPOINTS:
                transformed = coeff * partition + relax.arg_const
                points.extend(float(p) for p in transformed)

    # A modest fixed split keeps the dedicated trig relaxation useful even when
    # no AMP variable partition exists for the affine argument.
    base = _sorted_unique_points([p for p in points if lb - 1e-12 <= p <= ub + 1e-12])
    refined: list[float] = []
    for a, b in zip(base[:-1], base[1:]):
        if not refined:
            refined.append(float(a))
        width = float(b - a)
        if width > _MAX_TRIG_PIECEWISE_WIDTH:
            n_chunks = int(math.ceil(width / _MAX_TRIG_PIECEWISE_WIDTH))
            for k in range(1, n_chunks):
                refined.append(float(a + width * k / n_chunks))
        refined.append(float(b))
    return _sorted_unique_points(refined or base)


def _trig_piecewise_interval_specs(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[tuple[float, float, Optional[str]]]:
    """Build certified curvature subintervals for mixed-curvature trig functions."""
    if relax.func_name not in {"sin", "cos", "tan"}:
        return []
    if not (np.isfinite(relax.arg_lb) and np.isfinite(relax.arg_ub)):
        return []
    if relax.arg_ub - relax.arg_lb >= _MAX_TRIG_PIECEWISE_SPAN:
        return []
    bounds = _trig_range(relax.func_name, relax.arg_lb, relax.arg_ub)
    if bounds is None:
        return []
    if _univariate_curvature(relax.func_name, bounds[0], bounds[1]) is not None:
        return []

    points = _trig_partition_breakpoints(relax, disc_state, n_orig)
    if len(points) - 1 > _MAX_TRIG_PIECEWISE_INTERVALS:
        return []
    intervals: list[tuple[float, float, Optional[str]]] = []
    for a, b in zip(points[:-1], points[1:]):
        if b <= a + 1e-12:
            continue
        local_bounds = _trig_range(relax.func_name, a, b)
        curvature = None
        if local_bounds is not None:
            curvature = _univariate_curvature(relax.func_name, local_bounds[0], local_bounds[1])
        intervals.append((float(a), float(b), curvature))
    if sum(1 for _a, _b, curvature in intervals if curvature is not None) < 2:
        return []
    return intervals


def _affine_argument_has_continuous_var(arg_coeff: np.ndarray, model: Model) -> bool:
    """Return true if an affine argument depends on at least one continuous variable."""
    offset = 0
    for var in model._variables:
        is_continuous = var.var_type not in (VarType.BINARY, VarType.INTEGER)
        for k in range(var.size):
            if abs(float(arg_coeff[offset + k])) > 1e-12 and is_continuous:
                return True
        offset += var.size
    return False


def _trig_square_partition_breakpoints(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[float]:
    """Return safe breakpoints for a mixed-curvature trig-square argument interval."""
    lb = float(relax.arg_lb)
    ub = float(relax.arg_ub)
    points = [lb, ub]
    points.extend(_critical_points_in_interval(0.0, math.pi / 2.0, lb, ub))
    points.extend(_critical_points_in_interval(math.pi / 4.0, math.pi / 2.0, lb, ub))

    nz = np.flatnonzero(np.abs(relax.arg_coeff) > 1e-12)
    if nz.size == 1:
        var_idx = int(nz[0])
        if var_idx < n_orig and var_idx in disc_state.partitions:
            coeff = float(relax.arg_coeff[var_idx])
            partition = np.asarray(disc_state.partitions[var_idx], dtype=np.float64)
            if partition.size <= _MAX_TRIG_IMPORTED_BREAKPOINTS:
                transformed = coeff * partition + relax.arg_const
                points.extend(float(p) for p in transformed)

    base = _sorted_unique_points([p for p in points if lb - 1e-12 <= p <= ub + 1e-12])
    refined: list[float] = []
    for a, b in zip(base[:-1], base[1:]):
        if not refined:
            refined.append(float(a))
        width = float(b - a)
        if width > _MAX_TRIG_PIECEWISE_WIDTH:
            n_chunks = int(math.ceil(width / _MAX_TRIG_PIECEWISE_WIDTH))
            for k in range(1, n_chunks):
                refined.append(float(a + width * k / n_chunks))
        refined.append(float(b))
    return _sorted_unique_points(refined or base)


def _trig_square_piecewise_interval_specs(
    relax: UnivariateRelaxation,
    disc_state: DiscretizationState,
    n_orig: int,
) -> list[tuple[float, float, Optional[str]]]:
    """Build certified curvature subintervals for mixed-curvature trig-square terms."""
    if relax.func_name not in {"sin", "cos"}:
        return []
    if not (np.isfinite(relax.arg_lb) and np.isfinite(relax.arg_ub)):
        return []
    if relax.arg_ub - relax.arg_lb >= _MAX_TRIG_PIECEWISE_SPAN:
        return []
    if _trig_square_range(relax.func_name, relax.arg_lb, relax.arg_ub) is None:
        return []
    if _trig_square_curvature(relax.func_name, relax.arg_lb, relax.arg_ub) is not None:
        return []

    points = _trig_square_partition_breakpoints(relax, disc_state, n_orig)
    if len(points) - 1 > _MAX_TRIG_PIECEWISE_INTERVALS:
        return []

    intervals: list[tuple[float, float, Optional[str]]] = []
    for a, b in zip(points[:-1], points[1:]):
        if b <= a + 1e-12:
            continue
        curvature = _trig_square_curvature(relax.func_name, a, b)
        intervals.append((float(a), float(b), curvature))
    if sum(1 for _a, _b, curvature in intervals if curvature is not None) < 2:
        return []
    return intervals


def _univariate_signature(
    func_name: str,
    arg_coeff: np.ndarray,
    arg_const: float,
) -> tuple[str, tuple[float, ...], float]:
    return func_name, tuple(float(c) for c in arg_coeff.tolist()), float(arg_const)


def _flatten_additive_terms(
    expr: Expression, scale: float, out: list[tuple[float, Expression]]
) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_additive_terms(expr.left, scale, out)
        _flatten_additive_terms(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_additive_terms(expr.left, scale, out)
        _flatten_additive_terms(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _flatten_additive_terms(expr.operand, -scale, out)
        return
    if isinstance(expr, SumOverExpression):
        for term in expr.terms:
            _flatten_additive_terms(term, scale, out)
        return
    out.append((scale, expr))


def _match_scaled_constant_division(
    expr: Expression,
    scale: float,
) -> Optional[tuple[float, Expression]]:
    """Return (scaled numerator, denominator) for scale * (c / denominator)."""
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_constant_division(expr.operand, -scale)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_constant_division(expr.right, scale * left_const)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_constant_division(expr.left, scale * right_const)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "/":
        return None
    numerator = _constant_value(expr.left)
    if numerator is None or abs(numerator) <= 1e-12:
        return None
    return scale * numerator, expr.right


def _exact_positive_reciprocal_row(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[tuple[np.ndarray, float]]:
    """Match ``constant - numerator / positive_affine <= 0`` as an exact affine row."""
    terms: list[tuple[float, Expression]] = []
    _flatten_additive_terms(expr, 1.0, terms)

    constant_term = 0.0
    reciprocal_match: Optional[tuple[float, Expression]] = None

    for scale, term in terms:
        const_val = _constant_value(term)
        if const_val is not None:
            constant_term += scale * const_val
            continue

        match = _match_scaled_constant_division(term, scale)
        if match is None or reciprocal_match is not None:
            return None
        reciprocal_match = match

    if reciprocal_match is None or constant_term <= 0.0:
        return None

    scaled_numerator, denominator = reciprocal_match
    if scaled_numerator >= 0.0:
        return None

    try:
        denom_coeff, denom_const = _linearize_affine_expr(denominator, model, len(flat_lb))
        denom_lb, _denom_ub = _linear_expr_bounds(denom_coeff, denom_const, flat_lb, flat_ub)
    except ValueError:
        return None

    if denom_lb <= 0.0:
        return None
    rhs = -scaled_numerator / constant_term
    if not np.isfinite(rhs):
        return None
    return denom_coeff, float(rhs - denom_const)


def _flatten_product_factors(expr: Expression, out: list[Expression]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "*":
        _flatten_product_factors(expr.left, out)
        _flatten_product_factors(expr.right, out)
        return
    out.append(expr)


def _monomial_power_term(expr: Expression, model: Model) -> Optional[tuple[int, int]]:
    flat = _get_flat_index(expr, model)
    if flat is not None:
        return flat, 1
    if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
        base = _get_flat_index(expr.left, model)
        if base is None:
            return None
        exp_val = float(expr.right.value)
        n_int = int(exp_val)
        if exp_val == n_int and n_int >= 1:
            return base, n_int
    return None


def _match_scaled_monomial(expr: Expression, model: Model) -> Optional[tuple[float, int, int]]:
    factors: list[Expression] = []
    _flatten_product_factors(expr, factors)
    scalar = 1.0
    var_idx: Optional[int] = None
    power_total = 0
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            scalar *= const
            continue
        power_term = _monomial_power_term(factor, model)
        if power_term is None:
            return None
        factor_var, factor_power = power_term
        if var_idx is None:
            var_idx = factor_var
        elif var_idx != factor_var:
            return None
        power_total += factor_power
    if var_idx is None or power_total < 1:
        return None
    return scalar, var_idx, power_total


def _match_x_exp_product(expr: Expression, model: Model) -> Optional[tuple[float, int]]:
    factors: list[Expression] = []
    _flatten_product_factors(expr, factors)
    scalar = 1.0
    var_idx: Optional[int] = None
    exp_arg_idx: Optional[int] = None
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            scalar *= const
            continue
        flat = _get_flat_index(factor, model)
        if flat is not None:
            if var_idx is not None:
                return None
            var_idx = flat
            continue
        if isinstance(factor, FunctionCall) and factor.func_name == "exp" and len(factor.args) == 1:
            arg_idx = _get_flat_index(factor.args[0], model)
            if arg_idx is None or exp_arg_idx is not None:
                return None
            exp_arg_idx = arg_idx
            continue
        return None
    if var_idx is None or exp_arg_idx is None or var_idx != exp_arg_idx:
        return None
    return scalar, var_idx


def _safe_x_exp_value(x: float) -> Optional[float]:
    if not np.isfinite(x) or x > _MAX_FINITE_EXP_ARG:
        return None
    if x < -745.0:
        return 0.0
    return float(x * np.exp(x))


def _x_exp_upper_bound(var_idx: int, flat_lb: np.ndarray, flat_ub: np.ndarray) -> Optional[float]:
    lb = float(flat_lb[var_idx])
    ub = float(flat_ub[var_idx])
    if not (_is_effectively_finite(lb) and _is_effectively_finite(ub)):
        return None
    values = [_safe_x_exp_value(lb), _safe_x_exp_value(ub)]
    finite_values = [value for value in values if value is not None and np.isfinite(value)]
    if len(finite_values) != len(values):
        return None
    return max(finite_values)


def _is_cos_call(expr: Expression) -> bool:
    return isinstance(expr, FunctionCall) and expr.func_name == "cos" and len(expr.args) == 1


def _flat_variable_types(model: Model) -> list[VarType]:
    types: list[VarType] = []
    for var in model._variables:
        types.extend([var.var_type] * var.size)
    return types


def _integer_domain_values(
    var_idx: int,
    flat_types: list[VarType],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[range]:
    var_type = flat_types[var_idx]
    if var_type not in (VarType.BINARY, VarType.INTEGER):
        return None
    lb_i = float(flat_lb[var_idx])
    ub_i = float(flat_ub[var_idx])
    if not (_is_effectively_finite(lb_i) and _is_effectively_finite(ub_i)):
        return None
    lo = int(np.ceil(lb_i - 1e-9))
    hi = int(np.floor(ub_i + 1e-9))
    if var_type == VarType.BINARY:
        lo = max(lo, 0)
        hi = min(hi, 1)
    if lo > hi:
        return None
    return range(lo, hi + 1)


def _integer_affine_cos_lower_bound(
    expr: Expression,
    scale: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Return exact lower bound for scale*cos(integer-affine expr) on a small box."""
    if not isinstance(expr, FunctionCall) or expr.func_name != "cos" or len(expr.args) != 1:
        return None
    try:
        coeff, const = _linearize_affine_expr(expr.args[0], model, len(flat_lb))
    except ValueError:
        return None

    flat_types = _flat_variable_types(model)
    entries: list[tuple[float, range]] = []
    n_values = 1
    for var_idx, c_i in enumerate(coeff):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue
        values = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
        if values is None:
            return None
        n_values *= len(values)
        if n_values > _MAX_INTEGER_COS_ENUM:
            return None
        entries.append((c, values))

    if not entries:
        value = scale * float(np.cos(const))
        return value if np.isfinite(value) else None

    best = np.inf
    for assignment in itertools.product(*(values for _c, values in entries)):
        arg = float(const)
        for (c, _values), value in zip(entries, assignment):
            arg += c * float(value)
        best = min(best, scale * float(np.cos(arg)))
    return float(best) if np.isfinite(best) else None


def _integer_affine_trig_range(
    func_name: str,
    coeff: np.ndarray,
    const: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Return exact range for trig(affine integer vars) on small finite domains."""
    if func_name not in {"sin", "cos", "tan"}:
        return None

    flat_types = _flat_variable_types(model)
    entries: list[tuple[float, range]] = []
    n_values = 1
    for var_idx, c_i in enumerate(coeff):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue
        values = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
        if values is None:
            return None
        n_values *= len(values)
        if n_values > _MAX_INTEGER_COS_ENUM:
            return None
        entries.append((c, values))

    if not entries:
        value = _univariate_value(func_name, float(const))
        return (value, value) if np.isfinite(value) else None

    values_out: list[float] = []
    for assignment in itertools.product(*(values for _c, values in entries)):
        arg = float(const)
        for (c, _values), value in zip(entries, assignment):
            arg += c * float(value)
        value_out = _univariate_value(func_name, arg)
        if not np.isfinite(value_out):
            return None
        values_out.append(value_out)
    if not values_out:
        return None
    return min(values_out), max(values_out)


def _finite_domain_trig_square_table_values(
    relax: UnivariateRelaxation,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[tuple[int, float, float, list[int], list[float], list[float]]]:
    """Return exact finite-domain values for a single integer trig-square argument."""
    if relax.func_name not in {"sin", "cos"}:
        return None

    nz = np.flatnonzero(np.abs(relax.arg_coeff) > 1e-12)
    if nz.size != 1:
        return None

    var_idx = int(nz[0])
    flat_types = _flat_variable_types(model)
    domain = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
    if domain is None:
        return None

    domain_values = list(domain)
    if not domain_values or len(domain_values) > _MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES:
        return None

    arg_coeff = float(relax.arg_coeff[var_idx])
    arg_const = float(relax.arg_const)
    if not (_is_effectively_finite(arg_coeff) and _is_effectively_finite(arg_const)):
        return None

    trig_values: list[float] = []
    square_values: list[float] = []
    for value in domain_values:
        arg = arg_coeff * float(value) + arg_const
        trig_value = _univariate_value(relax.func_name, arg)
        square_value = trig_value * trig_value
        if not (np.isfinite(trig_value) and np.isfinite(square_value)):
            return None
        trig_values.append(float(trig_value))
        square_values.append(float(square_value))

    return var_idx, arg_coeff, arg_const, domain_values, trig_values, square_values


def _scaled_affine_lower_bound(
    expr: Expression,
    scale: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    try:
        coeff, const = _linearize_affine_expr(expr, model, len(flat_lb))
    except ValueError:
        return None
    want_lower = scale >= 0.0
    bound = float(const)
    for c_i, lb_i, ub_i in zip(coeff, flat_lb, flat_ub):
        c = float(c_i)
        if abs(c) <= 1e-12:
            continue
        chosen = float(lb_i) if (c >= 0.0) == want_lower else float(ub_i)
        if not _is_effectively_finite(chosen):
            return None
        bound += c * chosen
    return scale * bound


def _evaluate_polynomial(coeffs: dict[int, float], x: float) -> Optional[float]:
    max_power = max(coeffs)
    value = 0.0
    for power in range(max_power, -1, -1):
        value = value * x + float(coeffs.get(power, 0.0))
        if not np.isfinite(value):
            return None
    return float(value)


def _polynomial_lower_bound(
    coeffs: dict[int, float],
    lb: float,
    ub: float,
) -> Optional[float]:
    clean = {power: coeff for power, coeff in coeffs.items() if abs(coeff) > 1e-12}
    if not clean:
        return 0.0
    max_power = max(clean)
    if max_power == 0:
        return float(clean[0])

    leading = float(clean[max_power])
    lo_unbounded = not _is_effectively_finite(lb)
    hi_unbounded = not _is_effectively_finite(ub)
    if hi_unbounded and leading < 0.0:
        return None
    if lo_unbounded:
        if max_power % 2 == 0 and leading < 0.0:
            return None
        if max_power % 2 == 1 and leading > 0.0:
            return None

    candidates: list[float] = []
    if not lo_unbounded:
        candidates.append(float(lb))
    if not hi_unbounded:
        candidates.append(float(ub))

    deriv_coeffs = [power * clean.get(power, 0.0) for power in range(max_power, 0, -1)]
    roots = np.roots(deriv_coeffs) if deriv_coeffs else np.array([])
    for root in roots:
        if abs(float(np.imag(root))) > 1e-9:
            continue
        x = float(np.real(root))
        if (lo_unbounded or x >= lb - 1e-9) and (hi_unbounded or x <= ub + 1e-9):
            candidates.append(x)

    values: list[float] = []
    for x in _sorted_unique_points(candidates):
        value = _evaluate_polynomial(clean, x)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return min(values)


def _separable_objective_lower_bound(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Compute a conservative constant lower bound for simple separable objectives."""
    terms: list[tuple[float, Expression]] = []
    _flatten_additive_terms(expr, 1.0, terms)

    total = 0.0
    polynomial_terms: dict[int, dict[int, float]] = {}
    for scale, term in terms:
        if abs(scale) <= 1e-12:
            continue
        const = _constant_value(term)
        if const is not None:
            total += scale * const
            continue

        x_exp = _match_x_exp_product(term, model)
        if x_exp is not None:
            scalar, var_idx = x_exp
            term_scale = scale * scalar
            if abs(term_scale) <= 1e-12:
                continue
            if term_scale > 0.0:
                total += term_scale * (-1.0 / np.e)
                continue
            upper = _x_exp_upper_bound(var_idx, flat_lb, flat_ub)
            if upper is None:
                return None
            total += term_scale * upper
            continue

        if _is_cos_call(term):
            integer_lb = _integer_affine_cos_lower_bound(term, scale, model, flat_lb, flat_ub)
            total += integer_lb if integer_lb is not None else -abs(scale)
            continue

        monomial = _match_scaled_monomial(term, model)
        if monomial is not None:
            scalar, var_idx, power = monomial
            polynomial_terms.setdefault(var_idx, {})
            polynomial_terms[var_idx][power] = (
                polynomial_terms[var_idx].get(power, 0.0) + scale * scalar
            )
            continue

        affine_bound = _scaled_affine_lower_bound(term, scale, model, flat_lb, flat_ub)
        if affine_bound is None:
            return None
        total += affine_bound

    for var_idx, coeffs in polynomial_terms.items():
        lower = _polynomial_lower_bound(coeffs, float(flat_lb[var_idx]), float(flat_ub[var_idx]))
        if lower is None:
            return None
        total += lower

    if not np.isfinite(total):
        return None
    return float(total)


def _collect_univariate_relaxations(
    model: Model,
    n_orig: int,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    start_col: int,
) -> tuple[list[UnivariateRelaxation], dict[object, int], list[tuple[float, float]]]:
    """Collect supported univariate operator nodes and assign auxiliary columns."""
    relaxations: list[UnivariateRelaxation] = []
    var_map: dict[object, int] = {}
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
        exact_integer_range = _integer_affine_trig_range(
            func_name,
            arg_coeff,
            arg_const,
            model,
            flat_lb,
            flat_ub,
        )
        if exact_integer_range is not None:
            val_lb, val_ub = exact_integer_range
        else:
            val_lb, val_ub = _univariate_value_bounds(func_name, arg_lb, arg_ub)
        if not np.isfinite(val_lb) or not np.isfinite(val_ub):
            return
        signature = _univariate_signature(func_name, arg_coeff, arg_const)
        if signature in var_map:
            seen.add(expr_id)
            var_map[expr_id] = var_map[signature]
            return
        seen.add(expr_id)
        var_map[expr_id] = col_idx
        var_map[signature] = col_idx
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


def _collect_univariate_square_relaxations(
    model: Model,
    univariate_var_map: dict[object, int],
    all_bounds: list[tuple[float, float]],
    start_col: int,
) -> tuple[list[UnivariateSquareRelaxation], dict[tuple[int, int], int], list[tuple[float, float]]]:
    """Collect squares of lifted trig calls and assign auxiliary columns."""
    relaxations: list[UnivariateSquareRelaxation] = []
    var_map: dict[tuple[int, int], int] = {}
    bounds: list[tuple[float, float]] = []
    col_idx = start_col

    def maybe_add(expr: Expression) -> None:
        nonlocal col_idx
        if not (
            isinstance(expr, BinaryOp)
            and expr.op == "**"
            and isinstance(expr.left, FunctionCall)
            and expr.left.func_name in {"sin", "cos", "tan"}
            and isinstance(expr.right, Constant)
            and float(expr.right.value) == 2.0
        ):
            return

        base_col = univariate_var_map.get(id(expr.left))
        if base_col is None:
            return
        key = (base_col, 2)
        if key in var_map:
            return

        base_lb, base_ub = [float(v) for v in all_bounds[base_col]]
        vals = [base_lb * base_lb, base_ub * base_ub]
        if base_lb <= 0.0 <= base_ub:
            vals.append(0.0)
        var_map[key] = col_idx
        bounds.append((float(min(vals)), float(max(vals))))
        relaxations.append(
            UnivariateSquareRelaxation(
                base_col=base_col,
                aux_col=col_idx,
                base_lb=base_lb,
                base_ub=base_ub,
            )
        )
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
    univariate_var_map: dict[object, int],
    n_total_vars: int,
    fractional_power_var_map: Optional[dict[tuple[int, float], int]] = None,
    univariate_square_var_map: Optional[dict[tuple[int, int], int]] = None,
) -> tuple[np.ndarray, float]:
    """Walk expression tree and return (coeff, constant) for linearized form.

    coeff[j] = coefficient of MILP variable j in the linear approximation.
    constant = scalar constant term.

    Nonlinear terms must have a corresponding auxiliary variable in the maps;
    raises ValueError if an unregistered nonlinear term is encountered.
    """
    coeff = np.zeros(n_total_vars, dtype=np.float64)
    const_acc: list[float] = [0.0]
    n_orig = sum(var.size for var in model._variables)

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
                elif isinstance(e.left, Constant):
                    aux_col = univariate_var_map.get(id(e))
                    if aux_col is None:
                        try:
                            arg_coeff, arg_const = _linearize_affine_expr(e.right, model, n_orig)
                        except ValueError:
                            aux_col = None
                        else:
                            aux_col = univariate_var_map.get(
                                _univariate_signature("reciprocal", arg_coeff, arg_const)
                            )
                    if aux_col is not None:
                        coeff[aux_col] += scale * float(e.left.value)
                    else:
                        raise ValueError(f"Cannot linearize non-constant division: {e}")
                else:
                    raise ValueError(f"Cannot linearize non-constant division: {e}")

            elif e.op == "**":
                flat = _get_flat_index(e.left, model)
                if flat is not None and isinstance(e.right, Constant):
                    exp_val = float(e.right.value)
                    n_int = int(exp_val)
                    if exp_val == n_int:
                        if n_int == 1:
                            coeff[flat] += scale
                            return
                        if n_int == 0:
                            const_acc[0] += scale
                            return
                        if n_int >= 2:
                            key = (flat, n_int)
                            if key in monomial_var_map:
                                coeff[monomial_var_map[key]] += scale
                                return
                            raise ValueError(f"Monomial {key} not in monomial_var_map")
                    fp_key = (flat, exp_val)
                    if fractional_power_var_map and fp_key in fractional_power_var_map:
                        coeff[fractional_power_var_map[fp_key]] += scale
                        return
                    raise ValueError(f"Fractional power {fp_key} has no aux column")
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
                decomp = _decompose_product(
                    e,
                    model,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_var_map=univariate_var_map,
                )
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
                    elif univariate_square_var_map and key in univariate_square_var_map:
                        coeff[univariate_square_var_map[key]] += scale * c
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
    generation_guardrails: list[str] = []
    generation_guardrail_keys: set[tuple[str, str, int, int]] = set()
    objective_lift = _build_minmax_objective_lift(model, flat_lb, flat_ub)
    objective_lift_monomials: set[tuple[int, int]] = set()
    if objective_lift is not None:
        for branch_expr in objective_lift.branch_exprs:
            objective_lift_monomials.update(_collect_monomial_terms_for_lift(branch_expr, model))
    monomial_terms = sorted(set(terms.monomial) | objective_lift_monomials)

    def _record_generation_guardrail(
        kind: str,
        target: object,
        interval_count: int,
        limit: int,
    ) -> None:
        key = (kind, repr(target), int(interval_count), int(limit))
        if key in generation_guardrail_keys:
            return
        generation_guardrail_keys.add(key)
        note = (
            f"skipped {kind} refinement for {target}: "
            f"{interval_count} intervals exceeds cap {limit}"
        )
        generation_guardrails.append(note)
        logger.debug("AMP: %s", note)

    def _guarded_partition_points(
        kind: str,
        target: object,
        points: list[float] | np.ndarray,
    ) -> Optional[list[float]]:
        finite_points = [float(p) for p in points if np.isfinite(float(p))]
        guarded = _sorted_unique_points(finite_points)
        interval_count = max(0, len(guarded) - 1)
        if interval_count > _MAX_RELAXATION_PARTITION_INTERVALS:
            _record_generation_guardrail(
                kind,
                target,
                interval_count,
                _MAX_RELAXATION_PARTITION_INTERVALS,
            )
            return None
        return guarded

    def _coarse_monomial_breakpoints(lb_i: float, ub_i: float) -> list[float]:
        points = [lb_i, ub_i]
        if lb_i < 0.0 < ub_i:
            points.append(0.0)
        return _sorted_unique_points(points)

    def _monomial_aux_bounds(lb_i: float, ub_i: float, n: int) -> tuple[float, float]:
        """Return safe auxiliary bounds for ``s = x**n``.

        Effectively infinite original bounds cannot support numerically useful
        tangent/secant rows, but the auxiliary still lets constraints and
        objectives reference the lifted monomial instead of dropping the whole
        expression from the MILP relaxation.
        """
        lb_finite = _is_effectively_finite(lb_i)
        ub_finite = _is_effectively_finite(ub_i)
        if lb_finite and ub_finite:
            vals = [lb_i**n, ub_i**n]
            if n % 2 == 0 and lb_i < 0 < ub_i:
                vals.append(0.0)
            return min(vals), max(vals)

        if n % 2 == 0:
            lower = 0.0
            if lb_finite and lb_i > 0.0:
                lower = lb_i**n
            elif ub_finite and ub_i < 0.0:
                lower = ub_i**n
            return float(lower), np.inf

        lower = lb_i**n if lb_finite else -np.inf
        upper = ub_i**n if ub_finite else np.inf
        return float(lower), float(upper)

    # ── Assign MILP column indices ──────────────────────────────────────────
    # Original variables keep columns 0..n_orig-1. Additional columns are created
    # for lifted bilinear, trilinear, and monomial terms plus any piecewise binaries.
    bilinear_var_map: dict[tuple[int, int], int] = {}
    trilinear_var_map: dict[tuple[int, int, int], int] = {}
    trilinear_stage_map: dict[tuple[int, int, int], dict[str, object]] = {}
    multilinear_var_map: dict[tuple[int, ...], int] = {}
    multilinear_stage_map: dict[tuple[int, ...], list[dict[str, int]]] = {}
    monomial_var_map: dict[tuple[int, int], int] = {}
    univariate_var_map: dict[object, int] = {}
    fractional_power_var_map: dict[tuple[int, float], int] = {}

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

    for var_idx, n in monomial_terms:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        monomial_var_map[(var_idx, n)] = col_idx
        all_bounds.append(_monomial_aux_bounds(lb_i, ub_i, n))
        integrality_flags.append(0)
        col_idx += 1

    monomial_pw_map: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for var_idx, n in monomial_terms:
        if (var_idx, n) not in monomial_var_map:
            continue
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if (
            var_idx not in disc_state.partitions
            or not _power_is_convex_on_box(n, lb_i)
            or not _is_effectively_finite(lb_i)
            or not _is_effectively_finite(ub_i)
        ):
            continue

        breakpoints = _guarded_partition_points(
            "monomial piecewise",
            (var_idx, n),
            _monomial_breakpoints(var_idx, lb_i, ub_i, disc_state),
        )
        if breakpoints is None:
            continue
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

    univariate_square_relaxations, univariate_square_var_map, univariate_square_bounds = (
        _collect_univariate_square_relaxations(
            model,
            univariate_var_map,
            all_bounds,
            col_idx,
        )
    )
    for val_bounds in univariate_square_bounds:
        all_bounds.append(val_bounds)
        integrality_flags.append(0)
        col_idx += 1

    univariate_by_aux_col = {relax.aux_col: relax for relax in univariate_relaxations}
    finite_domain_trig_square_tables: list[FiniteDomainTrigSquareTable] = []
    for square_relax in univariate_square_relaxations:
        base_relax = univariate_by_aux_col.get(square_relax.base_col)
        if base_relax is None:
            continue
        table_values = _finite_domain_trig_square_table_values(
            base_relax,
            model,
            flat_lb,
            flat_ub,
        )
        if table_values is None:
            continue
        var_idx, arg_coeff, arg_const, domain_values, trig_values, square_values = table_values

        selector_cols: list[int] = []
        if len(domain_values) > 1:
            for _ in domain_values:
                selector_cols.append(col_idx)
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(1)
                col_idx += 1

        finite_domain_trig_square_tables.append(
            FiniteDomainTrigSquareTable(
                square=square_relax,
                func_name=base_relax.func_name,
                var_idx=var_idx,
                arg_coeff=arg_coeff,
                arg_const=arg_const,
                domain_values=domain_values,
                trig_values=trig_values,
                square_values=square_values,
                selector_cols=selector_cols,
            )
        )

    piecewise_trig_square_relaxations: list[PiecewiseTrigSquareRelaxation] = []
    for square_relax in univariate_square_relaxations:
        base_relax = univariate_by_aux_col.get(square_relax.base_col)
        if base_relax is None or base_relax.func_name not in {"sin", "cos"}:
            continue
        if not _affine_argument_has_continuous_var(base_relax.arg_coeff, model):
            continue
        interval_specs = _trig_square_piecewise_interval_specs(base_relax, disc_state, n_orig)
        if not interval_specs:
            continue

        trig_square_intervals: list[PiecewiseTrigSquareInterval] = []
        for a_k, b_k, curvature in interval_specs:
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            trig_square_intervals.append(
                PiecewiseTrigSquareInterval(
                    delta_col=delta_col,
                    lb=float(a_k),
                    ub=float(b_k),
                    curvature=curvature,
                )
            )

        if trig_square_intervals:
            piecewise_trig_square_relaxations.append(
                PiecewiseTrigSquareRelaxation(
                    square=square_relax,
                    func_name=base_relax.func_name,
                    arg_coeff=base_relax.arg_coeff,
                    arg_const=base_relax.arg_const,
                    arg_lb=base_relax.arg_lb,
                    arg_ub=base_relax.arg_ub,
                    intervals=trig_square_intervals,
                )
            )

    piecewise_univariate_relaxations: list[PiecewiseUnivariateRelaxation] = []
    for relax in univariate_relaxations:
        interval_specs = _trig_piecewise_interval_specs(relax, disc_state, n_orig)
        if not interval_specs:
            continue

        trig_intervals: list[PiecewiseUnivariateInterval] = []
        for a_k, b_k, curvature in interval_specs:
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            trig_intervals.append(
                PiecewiseUnivariateInterval(
                    delta_col=delta_col,
                    lb=float(a_k),
                    ub=float(b_k),
                    curvature=curvature,
                )
            )

        if trig_intervals:
            piecewise_univariate_relaxations.append(
                PiecewiseUnivariateRelaxation(relax=relax, intervals=trig_intervals)
            )

    # ── Fractional-power aux columns: a = x^p with non-integer p ────────────
    # Only handle the cases where the relaxation is well-defined and
    # numerically stable: x ≥ 0 strictly bounded, and either 0 < p < 1
    # (concave) or p > 1 (convex).  Other cases are skipped and remain
    # general_nl, surfacing through the existing warning path.
    fractional_power_specs: list[dict] = []
    for var_idx, p in terms.fractional_power:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if lb_i < 0.0 or ub_i <= lb_i:
            continue
        if p == 0.0 or p == 1.0:
            continue
        # Convexity: p ∈ (0,1) → concave on x ≥ 0; p > 1 or p < 0 → convex on x > 0.
        if 0.0 < p < 1.0:
            convexity = "concave"
        elif p > 1.0 or p < 0.0:
            convexity = "convex"
            if p < 0.0 and lb_i <= 0.0:
                continue
        else:
            continue
        try:
            f_lb = lb_i**p
            f_ub = ub_i**p
        except (ValueError, OverflowError):
            continue
        if not (np.isfinite(f_lb) and np.isfinite(f_ub)):
            continue
        col = col_idx
        fractional_power_var_map[(var_idx, float(p))] = col
        all_bounds.append((min(f_lb, f_ub), max(f_lb, f_ub)))
        integrality_flags.append(0)
        col_idx += 1
        fractional_power_specs.append(
            {
                "var": var_idx,
                "p": float(p),
                "col": col,
                "lb": lb_i,
                "ub": ub_i,
                "f_lb": f_lb,
                "f_ub": f_ub,
                "convexity": convexity,
            }
        )

    # ── Bilinear-with-fractional-power: y * x^p  →  McCormick on (y_col, fp_col)
    bilinear_with_fp_keys: list[tuple[int, int]] = []
    for lin_idx, fp_key in terms.bilinear_with_fp:
        if fp_key not in fractional_power_var_map:
            continue
        fp_col = fractional_power_var_map[fp_key]
        pair = (min(lin_idx, fp_col), max(lin_idx, fp_col))
        bilinear_with_fp_keys.append(pair)

    for key in bilinear_with_fp_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

    lifted_bilinear_keys = _collect_lifted_bilinear_products(
        model,
        fractional_power_var_map,
        univariate_var_map,
        n_orig,
    )
    for key in lifted_bilinear_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

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

        pts = _guarded_partition_points(
            "bilinear piecewise",
            (lhs_col, rhs_col),
            disc_state.partitions[part_var],
        )
        if pts is None:
            continue
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
                for _ in range(embedding_info.bit_count):
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

    # ── Piecewise aux columns for shared disaggregated structure ───────────
    # When a variable has partition breakpoints AND is the base of a concave
    # fractional power ``z = x^p`` with p in (0, 1), replace the global secant
    # with a piecewise version.  Monomial secants use their existing AMP
    # interval-selector formulation below.
    pw_candidate_vars: set[int] = set()
    for spec_pre in terms.fractional_power:
        var_idx, p = spec_pre
        if 0.0 < float(p) < 1.0 and var_idx in disc_state.partitions:
            pw_candidate_vars.add(var_idx)

    piecewise_var_map: dict[int, list[tuple[int, int, float, float]]] = {}
    for var_idx in sorted(pw_candidate_vars):
        pw_pts = _guarded_partition_points(
            "fractional-power piecewise",
            var_idx,
            disc_state.partitions[var_idx],
        )
        if pw_pts is None:
            continue
        if len(pw_pts) < 3:
            # With only 2 breakpoints there's just one interval; the global
            # secant already coincides with the piecewise secant.
            continue
        pw_intervals_list: list[tuple[int, int, float, float]] = []
        for k in range(len(pw_pts) - 1):
            p_lo = float(pw_pts[k])
            p_hi = float(pw_pts[k + 1])
            delta_col = col_idx
            all_bounds.append((0.0, 1.0))
            integrality_flags.append(1)
            col_idx += 1
            xbar_col = col_idx
            xbar_lb = min(p_lo, 0.0)
            xbar_ub = max(p_hi, 0.0)
            all_bounds.append((xbar_lb, xbar_ub))
            integrality_flags.append(0)
            col_idx += 1
            pw_intervals_list.append((delta_col, xbar_col, p_lo, p_hi))
        piecewise_var_map[var_idx] = pw_intervals_list

    if objective_lift is not None:
        objective_lift.aux_col = col_idx
        all_bounds.append(objective_lift.aux_bounds)
        integrality_flags.append(0)
        col_idx += 1

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

    # ── Piecewise structural constraints (once per partitioned variable) ────
    # For each var_idx with a piecewise structure we enforce:
    #   sum δ_k = 1, x = Σ x̄_k, p_k δ_k ≤ x̄_k ≤ p_{k+1} δ_k.
    # Both monomial-secant and concave-fp-secant rows reference these aux
    # columns, so emitting structural rows once avoids duplicate constraints.
    for var_idx, pw_intervals in piecewise_var_map.items():
        # 1) Σ δ_k = 1 (encoded as ≤ 1 and ≥ 1)
        row = np.zeros(n_total)
        for delta_col, _xbar_col, _plo, _phi in pw_intervals:
            row[delta_col] = 1.0
        _add_row(row, 1.0)
        _add_row(-row, -1.0)
        # 2) x = Σ x̄_k  →  x − Σ x̄_k = 0
        row = np.zeros(n_total)
        row[var_idx] = 1.0
        for _delta_col, xbar_col, _plo, _phi in pw_intervals:
            row[xbar_col] = -1.0
        _add_row(row, 0.0)
        _add_row(-row, 0.0)
        # 3) Per-interval bounds: p_k δ_k ≤ x̄_k ≤ p_{k+1} δ_k
        for delta_col, xbar_col, p_lo, p_hi in pw_intervals:
            row = np.zeros(n_total)
            row[xbar_col] = 1.0
            row[delta_col] = -p_hi
            _add_row(row, 0.0)
            row = np.zeros(n_total)
            row[xbar_col] = -1.0
            row[delta_col] = p_lo
            _add_row(row, 0.0)

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
                    embedding_info.positive_sets,
                    embedding_info.negative_sets,
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

                def _inactive_big_m(other_coeff: float, rhs: float) -> float:
                    violations = [
                        float(other_coeff) * yj_lb - float(rhs),
                        float(other_coeff) * yj_ub - float(rhs),
                    ]
                    max_violation = max(0.0, *violations)
                    if max_violation <= 0.0:
                        return 0.0
                    return max_violation * (1.0 + 1e-4) + max(
                        1e-9,
                        1e-9 * max_violation,
                    )

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

                # w̄_k ≥ wk_lo * δ_k. Together with the upper row this forces
                # w̄_k=0 when the interval is inactive, even for negative products.
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[delta_col] = wk_lo
                _add_row(row, 0.0)

                # Per-interval McCormick with big-M relaxation.
                # The big-M term LOOSENS the constraint when δ_k=0 (interval inactive).
                #
                # cv1: w̄_k ≥ a_k*y + x̄_k*y_lb - a_k*y_lb - M*(1-δ_k)
                #   → -w̄_k + a_k*y + x̄_k*y_lb + M*δ_k ≤ a_k*y_lb + M
                rhs = a_k * yj_lb
                big_m = _inactive_big_m(a_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[other_var] += a_k
                row[xbar_col] += yj_lb
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

                # cv2: w̄_k ≥ b_k*y + x̄_k*y_ub - b_k*y_ub - M*(1-δ_k)
                #   → -w̄_k + b_k*y + x̄_k*y_ub + M*δ_k ≤ b_k*y_ub + M
                rhs = b_k * yj_ub
                big_m = _inactive_big_m(b_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = -1.0
                row[other_var] += b_k
                row[xbar_col] += yj_ub
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

                # cc1: w̄_k ≤ b_k*y + x̄_k*y_lb - b_k*y_lb + M*(1-δ_k)
                #   → w̄_k - b_k*y - x̄_k*y_lb + M*δ_k ≤ M - b_k*y_lb
                rhs = -b_k * yj_lb
                big_m = _inactive_big_m(-b_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[other_var] -= b_k
                row[xbar_col] -= yj_lb
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

                # cc2: w̄_k ≤ a_k*y + x̄_k*y_ub - a_k*y_ub + M*(1-δ_k)
                #   → w̄_k - a_k*y - x̄_k*y_ub + M*δ_k ≤ M - a_k*y_ub
                rhs = -a_k * yj_ub
                big_m = _inactive_big_m(-a_k, rhs)
                row = np.zeros(n_total)
                row[wbar_col] = 1.0
                row[other_var] -= a_k
                row[xbar_col] -= yj_ub
                row[delta_col] = big_m
                _add_row(row, rhs + big_m)

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

    # ── β-driven piecewise McCormick on bilinear-with-fp ────────────────────
    # For pairs y = w * z where z = β^p is a fractional-power aux, the standard
    # bilinear McCormick uses z's GLOBAL bounds, so it stays loose even after w
    # is heavily partitioned.  When β has a piecewise structure we can derive
    # per-β-interval tight bounds on z (z ∈ [p_k^p, p_{k+1}^p] when β ∈
    # [p_k, p_{k+1}]) and add per-interval big-M McCormick on top of the
    # existing standard or w-piecewise relaxation.  Their intersection is at
    # least as tight, and is dramatically tighter inside each β cell.
    for lin_idx, fp_key in terms.bilinear_with_fp:
        if fp_key not in fractional_power_var_map:
            continue
        fp_col = fractional_power_var_map[fp_key]
        beta_var, p_exp = fp_key
        beta_var = int(beta_var)
        p_exp = float(p_exp)
        pw_intervals = piecewise_var_map.get(beta_var, [])
        if not pw_intervals:
            continue
        pair_key = (min(lin_idx, fp_col), max(lin_idx, fp_col))
        if pair_key not in bilinear_var_map:
            continue
        y_col = bilinear_var_map[pair_key]
        w_lb, w_ub = [float(v) for v in all_bounds[lin_idx]]
        z_lb_global, z_ub_global = [float(v) for v in all_bounds[fp_col]]
        for delta_col, _xbar_col, p_lo, p_hi in pw_intervals:
            try:
                z_at_lo = p_lo**p_exp
                z_at_hi = p_hi**p_exp
            except (ValueError, OverflowError):
                continue
            if not (np.isfinite(z_at_lo) and np.isfinite(z_at_hi)):
                continue
            z_lb_k = min(z_at_lo, z_at_hi)
            z_ub_k = max(z_at_lo, z_at_hi)
            # Skip degenerate intervals.
            if z_ub_k - z_lb_k < 1e-12:
                continue
            corners = [w_lb * z_lb_k, w_lb * z_ub_k, w_ub * z_lb_k, w_ub * z_ub_k]
            # Big-M sized to dominate the global y range when δ_k = 0; use the
            # max global corner so the relaxation is automatically slack on
            # inactive intervals.
            global_corners = [
                w_lb * z_lb_global,
                w_lb * z_ub_global,
                w_ub * z_lb_global,
                w_ub * z_ub_global,
            ]
            M_k = _compute_piecewise_big_m(global_corners + corners)
            # cv1: y ≥ z_lb_k*w + w_lb*z - z_lb_k*w_lb  (relaxed by M when δ=0)
            #   →  -y + z_lb_k*w + w_lb*z + M*δ_k ≤ z_lb_k*w_lb + M
            row = np.zeros(n_total)
            row[y_col] = -1.0
            row[lin_idx] += z_lb_k
            row[fp_col] += w_lb
            row[delta_col] = M_k
            _add_row(row, z_lb_k * w_lb + M_k)
            # cv2: y ≥ z_ub_k*w + w_ub*z - z_ub_k*w_ub
            row = np.zeros(n_total)
            row[y_col] = -1.0
            row[lin_idx] += z_ub_k
            row[fp_col] += w_ub
            row[delta_col] = M_k
            _add_row(row, z_ub_k * w_ub + M_k)
            # cc1: y ≤ z_ub_k*w + w_lb*z - z_ub_k*w_lb
            #   →  y - z_ub_k*w - w_lb*z + M*δ_k ≤ M - z_ub_k*w_lb
            row = np.zeros(n_total)
            row[y_col] = 1.0
            row[lin_idx] -= z_ub_k
            row[fp_col] -= w_lb
            row[delta_col] = M_k
            _add_row(row, M_k - z_ub_k * w_lb)
            # cc2: y ≤ z_lb_k*w + w_ub*z - z_lb_k*w_ub
            row = np.zeros(n_total)
            row[y_col] = 1.0
            row[lin_idx] -= z_lb_k
            row[fp_col] -= w_ub
            row[delta_col] = M_k
            _add_row(row, M_k - z_lb_k * w_ub)

    # Monomial constraints
    for var_idx, n in monomial_terms:
        if (var_idx, n) not in monomial_var_map:
            continue
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        if not (_is_effectively_finite(lb_i) and _is_effectively_finite(ub_i)):
            continue
        s_col = monomial_var_map[(var_idx, n)]
        breakpoints = _guarded_partition_points(
            "monomial tangent",
            (var_idx, n),
            _monomial_breakpoints(var_idx, lb_i, ub_i, disc_state),
        )
        if breakpoints is None:
            breakpoints = _coarse_monomial_breakpoints(lb_i, ub_i)

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

    def _add_gated_row(row: np.ndarray, rhs: float, delta_col: int, big_m: float) -> None:
        """Add ``row @ z <= rhs`` active when ``delta_col`` is one."""
        gated = row.copy()
        gated[delta_col] += max(0.0, float(big_m))
        _add_row(gated, rhs + max(0.0, float(big_m)))

    def _linear_line_bounds(
        slope: float,
        intercept: float,
        lb: float,
        ub: float,
    ) -> tuple[float, float]:
        values = [slope * lb + intercept, slope * ub + intercept]
        return min(values), max(values)

    def _add_gated_lower_line(
        relax: UnivariateRelaxation,
        interval: PiecewiseUnivariateInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add t >= slope * arg + intercept on one active interval."""
        t_lb, _t_ub = [float(v) for v in all_bounds[relax.aux_col]]
        _line_lb, line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, line_ub - t_lb)

        row = np.zeros(n_total)
        row[:n_orig] = slope * relax.arg_coeff
        row[relax.aux_col] = -1.0
        rhs = -intercept - slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    def _add_gated_upper_line(
        relax: UnivariateRelaxation,
        interval: PiecewiseUnivariateInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add t <= slope * arg + intercept on one active interval."""
        _t_lb, t_ub = [float(v) for v in all_bounds[relax.aux_col]]
        line_lb, _line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, t_ub - line_lb)

        row = np.zeros(n_total)
        row[:n_orig] = -slope * relax.arg_coeff
        row[relax.aux_col] = 1.0
        rhs = intercept + slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    def _add_gated_trig_square_lower_line(
        relax: PiecewiseTrigSquareRelaxation,
        interval: PiecewiseTrigSquareInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add q >= slope * arg + intercept on one active interval."""
        q_lb, _q_ub = [float(v) for v in all_bounds[relax.square.aux_col]]
        _line_lb, line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, line_ub - q_lb)

        row = np.zeros(n_total)
        row[:n_orig] = slope * relax.arg_coeff
        row[relax.square.aux_col] = -1.0
        rhs = -intercept - slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

    def _add_gated_trig_square_upper_line(
        relax: PiecewiseTrigSquareRelaxation,
        interval: PiecewiseTrigSquareInterval,
        slope: float,
        intercept: float,
    ) -> None:
        """Add q <= slope * arg + intercept on one active interval."""
        _q_lb, q_ub = [float(v) for v in all_bounds[relax.square.aux_col]]
        line_lb, _line_ub = _linear_line_bounds(slope, intercept, relax.arg_lb, relax.arg_ub)
        big_m = max(0.0, q_ub - line_lb)

        row = np.zeros(n_total)
        row[:n_orig] = -slope * relax.arg_coeff
        row[relax.square.aux_col] = 1.0
        rhs = intercept + slope * relax.arg_const
        _add_gated_row(row, rhs, interval.delta_col, big_m)

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
        if relax.func_name in {"sin", "cos", "tan"}:
            continuous_bounds = _trig_range(relax.func_name, lb_u, ub_u)
            if continuous_bounds is None:
                continue
            val_lb, val_ub = continuous_bounds
        else:
            val_lb, val_ub = [float(v) for v in all_bounds[relax.aux_col]]
        curvature = _univariate_curvature(relax.func_name, val_lb, val_ub)

        if curvature == "convex":
            # Convex: tangents are lower bounds; secant is an upper bound.
            for pt in _tangent_points(relax.func_name, lb_u, ub_u):
                slope = _univariate_grad(relax.func_name, pt)
                intercept = _univariate_value(relax.func_name, pt) - slope * pt
                _add_lower_line(relax, slope, intercept)
            _add_upper_line(relax, secant_slope, secant_intercept)
        elif curvature == "concave":
            # Concave: secant is a lower bound; tangents are upper bounds.
            _add_lower_line(relax, secant_slope, secant_intercept)
            for pt in _tangent_points(relax.func_name, lb_u, ub_u):
                slope = _univariate_grad(relax.func_name, pt)
                intercept = _univariate_value(relax.func_name, pt) - slope * pt
                _add_upper_line(relax, slope, intercept)

    for table in finite_domain_trig_square_tables:
        base_col = table.square.base_col
        square_col = table.square.aux_col

        if not table.selector_cols:
            trig_value = table.trig_values[0]
            square_value = table.square_values[0]

            row = np.zeros(n_total)
            row[base_col] = 1.0
            _add_row(row, trig_value)
            _add_row(-row, -trig_value)

            row = np.zeros(n_total)
            row[square_col] = 1.0
            _add_row(row, square_value)
            _add_row(-row, -square_value)
            continue

        row_sum = np.zeros(n_total)
        for selector_col in table.selector_cols:
            row_sum[selector_col] = 1.0
        _add_row(row_sum, 1.0)
        _add_row(-row_sum, -1.0)

        row = np.zeros(n_total)
        row[table.var_idx] = 1.0
        for domain_value, selector_col in zip(table.domain_values, table.selector_cols):
            row[selector_col] -= float(domain_value)
        _add_row(row, 0.0)
        _add_row(-row, 0.0)

        row = np.zeros(n_total)
        row[base_col] = 1.0
        for trig_value, selector_col in zip(table.trig_values, table.selector_cols):
            row[selector_col] -= trig_value
        _add_row(row, 0.0)
        _add_row(-row, 0.0)

        row = np.zeros(n_total)
        row[square_col] = 1.0
        for square_value, selector_col in zip(table.square_values, table.selector_cols):
            row[selector_col] -= square_value
        _add_row(row, 0.0)
        _add_row(-row, 0.0)

    for pw_relax in piecewise_univariate_relaxations:
        relax = pw_relax.relax

        row_sum = np.zeros(n_total)
        for interval in pw_relax.intervals:
            row_sum[interval.delta_col] = -1.0
        _add_row(row_sum, -1.0)
        _add_row(-row_sum, 1.0)

        for interval in pw_relax.intervals:
            arg_lb = float(relax.arg_lb)
            arg_ub = float(relax.arg_ub)

            # arg >= interval.lb when selected.
            lower_m = max(0.0, interval.lb - arg_lb)
            row = np.zeros(n_total)
            row[:n_orig] = -relax.arg_coeff
            rhs = relax.arg_const - interval.lb
            _add_gated_row(row, rhs, interval.delta_col, lower_m)

            # arg <= interval.ub when selected.
            upper_m = max(0.0, arg_ub - interval.ub)
            row = np.zeros(n_total)
            row[:n_orig] = relax.arg_coeff
            rhs = interval.ub - relax.arg_const
            _add_gated_row(row, rhs, interval.delta_col, upper_m)

            if interval.curvature not in {"convex", "concave"}:
                continue

            f_lb = _univariate_value(relax.func_name, interval.lb)
            f_ub = _univariate_value(relax.func_name, interval.ub)
            secant_slope = (f_ub - f_lb) / (interval.ub - interval.lb)
            secant_intercept = f_lb - secant_slope * interval.lb

            if interval.curvature == "convex":
                for pt in _tangent_points(relax.func_name, interval.lb, interval.ub):
                    slope = _univariate_grad(relax.func_name, pt)
                    intercept = _univariate_value(relax.func_name, pt) - slope * pt
                    _add_gated_lower_line(relax, interval, slope, intercept)
                _add_gated_upper_line(relax, interval, secant_slope, secant_intercept)
            else:
                _add_gated_lower_line(relax, interval, secant_slope, secant_intercept)
                for pt in _tangent_points(relax.func_name, interval.lb, interval.ub):
                    slope = _univariate_grad(relax.func_name, pt)
                    intercept = _univariate_value(relax.func_name, pt) - slope * pt
                    _add_gated_upper_line(relax, interval, slope, intercept)

    for trig_square_relax in piecewise_trig_square_relaxations:
        row_sum = np.zeros(n_total)
        for trig_square_interval in trig_square_relax.intervals:
            row_sum[trig_square_interval.delta_col] = -1.0
        _add_row(row_sum, -1.0)
        _add_row(-row_sum, 1.0)

        for trig_square_interval in trig_square_relax.intervals:
            arg_lb = float(trig_square_relax.arg_lb)
            arg_ub = float(trig_square_relax.arg_ub)

            # arg >= interval.lb when selected.
            lower_m = max(0.0, trig_square_interval.lb - arg_lb)
            row = np.zeros(n_total)
            row[:n_orig] = -trig_square_relax.arg_coeff
            rhs = trig_square_relax.arg_const - trig_square_interval.lb
            _add_gated_row(row, rhs, trig_square_interval.delta_col, lower_m)

            # arg <= interval.ub when selected.
            upper_m = max(0.0, arg_ub - trig_square_interval.ub)
            row = np.zeros(n_total)
            row[:n_orig] = trig_square_relax.arg_coeff
            rhs = trig_square_interval.ub - trig_square_relax.arg_const
            _add_gated_row(row, rhs, trig_square_interval.delta_col, upper_m)

            if trig_square_interval.curvature not in {"convex", "concave"}:
                continue

            f_lb = _trig_square_value(trig_square_relax.func_name, trig_square_interval.lb)
            f_ub = _trig_square_value(trig_square_relax.func_name, trig_square_interval.ub)
            secant_slope = (f_ub - f_lb) / (trig_square_interval.ub - trig_square_interval.lb)
            secant_intercept = f_lb - secant_slope * trig_square_interval.lb

            tangent_points = [
                trig_square_interval.lb,
                0.5 * (trig_square_interval.lb + trig_square_interval.ub),
                trig_square_interval.ub,
            ]
            if trig_square_interval.curvature == "convex":
                for pt in _sorted_unique_points(tangent_points):
                    slope = _trig_square_grad(trig_square_relax.func_name, pt)
                    intercept = _trig_square_value(trig_square_relax.func_name, pt) - slope * pt
                    _add_gated_trig_square_lower_line(
                        trig_square_relax,
                        trig_square_interval,
                        slope,
                        intercept,
                    )
                _add_gated_trig_square_upper_line(
                    trig_square_relax,
                    trig_square_interval,
                    secant_slope,
                    secant_intercept,
                )
            else:
                _add_gated_trig_square_lower_line(
                    trig_square_relax,
                    trig_square_interval,
                    secant_slope,
                    secant_intercept,
                )
                for pt in _sorted_unique_points(tangent_points):
                    slope = _trig_square_grad(trig_square_relax.func_name, pt)
                    intercept = _trig_square_value(trig_square_relax.func_name, pt) - slope * pt
                    _add_gated_trig_square_upper_line(
                        trig_square_relax,
                        trig_square_interval,
                        slope,
                        intercept,
                    )

    for square_relax in univariate_square_relaxations:
        lb_i = square_relax.base_lb
        ub_i = square_relax.base_ub
        tangent_pts = [lb_i, ub_i]
        if lb_i <= 0.0 <= ub_i:
            tangent_pts.append(0.0)
        for t in _sorted_unique_points(tangent_pts):
            row = np.zeros(n_total)
            row[square_relax.aux_col] = -1.0
            row[square_relax.base_col] = 2.0 * t
            _add_row(row, t * t)
        if abs(ub_i - lb_i) > 1e-12:
            row = np.zeros(n_total)
            row[square_relax.aux_col] = 1.0
            row[square_relax.base_col] = -(lb_i + ub_i)
            _add_row(row, -lb_i * ub_i)

    # ── Fractional-power envelope constraints ──────────────────────────────
    # For a = x^p with x in [lb, ub], lb ≥ 0:
    #   - 0 < p < 1 (concave on x ≥ 0):
    #         secant under-estimator, tangent over-estimators (refined by partition).
    #   - p > 1 or p < 0 with lb > 0 (convex):
    #         tangent under-estimators, secant over-estimator.
    for spec in fractional_power_specs:
        var_idx = spec["var"]
        p = spec["p"]
        a_col = spec["col"]
        lb_i = spec["lb"]
        ub_i = spec["ub"]
        f_lb = spec["f_lb"]
        f_ub = spec["f_ub"]
        convexity = spec["convexity"]

        # Tangent points: include partition breakpoints when available so the
        # relaxation tightens monotonically as AMP refines.
        if var_idx in disc_state.partitions and len(disc_state.partitions[var_idx]) >= 2:
            guarded_tangent_pts = _guarded_partition_points(
                "fractional-power tangent",
                (var_idx, p),
                disc_state.partitions[var_idx],
            )
            if guarded_tangent_pts is None:
                tangent_pts = [lb_i, ub_i]
            else:
                tangent_pts = [float(t) for t in guarded_tangent_pts]
        else:
            tangent_pts = [lb_i, ub_i]
        # Avoid degenerate tangents at zero when the slope or value is undefined.
        tangent_pts = [t for t in tangent_pts if t > 0.0 or (t == 0.0 and p > 1.0)]
        if not tangent_pts:
            tangent_pts = [max(lb_i, 1e-12), ub_i]

        # Secant slope over [lb, ub].
        if abs(ub_i - lb_i) > 1e-12:
            secant_slope = (f_ub - f_lb) / (ub_i - lb_i)
            secant_intercept = f_lb - secant_slope * lb_i
        else:
            secant_slope = 0.0
            secant_intercept = f_lb

        if convexity == "concave":
            pw_intervals = piecewise_var_map.get(var_idx, [])
            if pw_intervals:
                # Piecewise secant under-estimator: per interval k = [p_k, p_{k+1}],
                # a ≥ p_k^p + slope_k (x − p_k), where slope_k = (p_{k+1}^p − p_k^p) /
                # (p_{k+1} − p_k).  Disaggregated form using shared (δ_k, x̄_k):
                #   −a + Σ_k (slope_k x̄_k + (p_k^p − slope_k p_k) δ_k) ≤ 0.
                row = np.zeros(n_total)
                row[a_col] = -1.0
                for delta_col, xbar_col, p_lo, p_hi in pw_intervals:
                    if abs(p_hi - p_lo) > 1e-12:
                        try:
                            f_plo = p_lo**p
                            f_phi = p_hi**p
                        except (ValueError, OverflowError):
                            continue
                        if not (np.isfinite(f_plo) and np.isfinite(f_phi)):
                            continue
                        slope_k = (f_phi - f_plo) / (p_hi - p_lo)
                        intercept_k = f_plo - slope_k * p_lo
                        row[xbar_col] += slope_k
                        row[delta_col] += intercept_k
                _add_row(row, 0.0)
            else:
                # Global secant under-estimator: a ≥ secant_slope*x + secant_intercept
                #   →  -a + secant_slope*x ≤ -secant_intercept
                row = np.zeros(n_total)
                row[a_col] = -1.0
                row[var_idx] = secant_slope
                _add_row(row, -secant_intercept)
            # Tangent over-estimators: a ≤ p*t^(p-1)*(x-t) + t^p
            #   →  a - p*t^(p-1)*x ≤ -((p-1)*t^p) ... derivation below.
            #   t_slope = p*t^(p-1);  t_const = t^p - t_slope*t = (1-p)*t^p
            for t in tangent_pts:
                t_slope = p * (t ** (p - 1.0))
                t_const = (1.0 - p) * (t**p)
                row = np.zeros(n_total)
                row[a_col] = 1.0
                row[var_idx] = -t_slope
                _add_row(row, t_const)
        else:  # convex
            # Tangent under-estimators: a ≥ p*t^(p-1)*(x-t) + t^p
            for t in tangent_pts:
                t_slope = p * (t ** (p - 1.0))
                t_const = (1.0 - p) * (t**p)
                row = np.zeros(n_total)
                row[a_col] = -1.0
                row[var_idx] = t_slope
                _add_row(row, -t_const)
            # Secant over-estimator: a ≤ secant_slope*x + secant_intercept
            row = np.zeros(n_total)
            row[a_col] = 1.0
            row[var_idx] = -secant_slope
            _add_row(row, secant_intercept)

    if objective_lift is not None:
        for branch_expr in objective_lift.branch_exprs:
            try:
                c_branch, const_branch = _linearize_expr(
                    branch_expr,
                    model,
                    bilinear_var_map,
                    trilinear_var_map,
                    multilinear_var_map,
                    monomial_var_map,
                    univariate_var_map,
                    n_total,
                    fractional_power_var_map=fractional_power_var_map,
                    univariate_square_var_map=univariate_square_var_map,
                )
            except ValueError as err:
                logger.debug(
                    "AMP: min/max objective lift uses auxiliary bounds because a branch "
                    "could not be linearized: %s",
                    err,
                )
                continue

            if objective_lift.func_name == "max":
                # minimize max(f_i): f_i <= t  ->  f_i - t <= 0
                row = c_branch.copy()
                row[objective_lift.aux_col] -= 1.0
                _add_row(row, -const_branch)
            else:
                # maximize min(f_i): t <= f_i  ->  t - f_i <= 0
                row = -c_branch
                row[objective_lift.aux_col] += 1.0
                _add_row(row, const_branch)

    # Model constraints
    for constraint in model._constraints:
        body = distribute_products(constraint.body)
        sense = constraint.sense
        if sense == "<=":
            exact_row = _exact_positive_reciprocal_row(body, model, flat_lb, flat_ub)
            if exact_row is not None:
                c_exact, rhs_exact = exact_row
                row_exact = np.zeros(n_total)
                row_exact[:n_orig] = c_exact
                _add_row(row_exact, rhs_exact)
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
                fractional_power_var_map=fractional_power_var_map,
                univariate_square_var_map=univariate_square_var_map,
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
            _warn_once(
                "AMP: omitting constraint %s from the MILP relaxation because it cannot "
                "be linearized safely: %s",
                constraint.name or "<unnamed>",
                err,
            )

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
    obj_expr = distribute_products(model._objective.expression)
    try:
        if objective_lift is not None:
            c_obj = np.zeros(n_total)
            c_obj[objective_lift.aux_col] = 1.0
            const_obj = 0.0
        else:
            c_obj, const_obj = _linearize_expr(
                obj_expr,
                model,
                bilinear_var_map,
                trilinear_var_map,
                multilinear_var_map,
                monomial_var_map,
                univariate_var_map,
                n_total,
                fractional_power_var_map=fractional_power_var_map,
                univariate_square_var_map=univariate_square_var_map,
            )
        objective_bound_valid = True
    except ValueError as err:
        fallback_lb = None
        if model._objective.sense == ObjectiveSense.MINIMIZE:
            fallback_lb = _separable_objective_lower_bound(obj_expr, model, flat_lb, flat_ub)
        if fallback_lb is not None:
            c_obj = np.zeros(n_total)
            const_obj = float(fallback_lb)
            objective_bound_valid = True
            logger.debug(
                "AMP: using separable objective lower bound after linearization failed: %s",
                err,
            )
        else:
            # Keep a feasibility objective so the relaxation can still produce a point,
            # but do not treat the LP value as a sound global bound. Warn loudly:
            # without an objective, AMP's lower-bound machinery is disabled and the
            # solver can only ever return "feasible", never "optimal".
            _warn_once(
                "MILP relaxation could not linearize the objective (%s); falling back to "
                "a feasibility objective. AMP will not be able to produce a lower bound "
                "or certify optimality on this problem.",
                err,
            )
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
        "univariate": {k: v for k, v in univariate_var_map.items() if isinstance(k, int)},
        "univariate_signatures": {
            k: v for k, v in univariate_var_map.items() if not isinstance(k, int)
        },
        "univariate_relaxations": univariate_relaxations,
        "univariate_piecewise_relaxations": piecewise_univariate_relaxations,
        "univariate_square": univariate_square_var_map,
        "univariate_square_relaxations": univariate_square_relaxations,
        "univariate_square_piecewise_relaxations": piecewise_trig_square_relaxations,
        "finite_domain_trig_square_tables": finite_domain_trig_square_tables,
        "fractional_power": fractional_power_var_map,
        "minmax_objective_lift": (
            {
                "func_name": objective_lift.func_name,
                "aux_col": objective_lift.aux_col,
                "branch_bounds": objective_lift.branch_bounds,
                "aux_bounds": objective_lift.aux_bounds,
            }
            if objective_lift is not None
            else None
        ),
        "bilinear_pw": bilinear_pw_map,
        "bilinear_lambda": bilinear_lambda_map,
        "convhull_formulation": convhull_mode,
        "convhull_ebd": convhull_ebd,
        "convhull_ebd_encoding": convhull_ebd_encoding,
        "generation_guardrails": generation_guardrails,
    }

    return milp_model, varmap
