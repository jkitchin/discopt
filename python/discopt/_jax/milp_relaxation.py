"""
MILP Relaxation Builder for AMP (Adaptive Multivariate Partitioning).

Builds a linear programming relaxation of the original MINLP by:
  1. Replacing bilinear terms x_i*x_j with auxiliary variables w_ij and
     adding standard McCormick envelope constraints.
  2. Replacing monomial terms x_i^n with auxiliary variables s_i and adding
     piecewise tangent-cut underestimators (using disc_state partition intervals)
     and a global secant overestimator.
  3. Linearizing the original objective and constraints.

The LP relaxation gives a valid lower bound:
  LP_opt ≤ global NLP_opt

As the partition becomes finer (more intervals in disc_state), more tangent cuts
are added for monomials in the objective, tightening the lower bound.

Theory: Nagarajan et al., JOGO 2018, Section 4 (piecewise McCormick relaxation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt._jax.discretization import DiscretizationState
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


def _choose_trilinear_pair(
    term: tuple[int, int, int],
    partitioned_vars: set[int],
) -> tuple[tuple[int, int], int]:
    """Choose a deterministic trilinear decomposition pair.

    Prefer a pair that already uses partitioned original variables so the
    existing piecewise bilinear machinery can tighten one stage of the lifted
    relaxation.
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


# ---------------------------------------------------------------------------
# Helpers: expression linearizer
# ---------------------------------------------------------------------------


def _linearize_expr(
    expr: Expression,
    model: Model,
    bilinear_var_map: dict[tuple[int, int], int],
    trilinear_var_map: dict[tuple[int, int, int], int],
    monomial_var_map: dict[tuple[int, int], int],
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
                    raise ValueError(f"Higher-order product ({len(unique)} vars) not supported")

            else:
                raise ValueError(f"Cannot linearize BinaryOp: {e.op}")

        elif isinstance(e, UnaryOp):
            if e.op == "neg":
                visit(e.operand, -scale)
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

    Returns
    -------
    (MilpRelaxationModel, varmap)
        MilpRelaxationModel has a .solve() method returning MilpRelaxationResult.
        varmap maps auxiliary variable keys to MILP column indices.
    """
    flat_lb, flat_ub = flat_variable_bounds(model)
    n_orig = len(flat_lb)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)

    # ── Assign MILP column indices ──────────────────────────────────────────
    bilinear_var_map: dict[tuple[int, int], int] = {}
    trilinear_var_map: dict[tuple[int, int, int], int] = {}
    trilinear_stage_map: dict[tuple[int, int, int], dict[str, object]] = {}
    monomial_var_map: dict[tuple[int, int], int] = {}

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

    original_bilinear_keys = sorted({(min(i, j), max(i, j)) for i, j in terms.bilinear})
    for key in original_bilinear_keys:
        bilinear_var_map[key] = _ensure_bilinear_aux(*key)

    partitioned_vars = set(disc_state.partitions)
    trilinear_terms: list[tuple[int, int, int]] = []
    for term in terms.trilinear:
        i0, i1, i2 = sorted(term)
        ordered: tuple[int, int, int] = (i0, i1, i2)
        if ordered not in trilinear_terms:
            trilinear_terms.append(ordered)

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
            theta_lb = min(0.0, float(other_lb), float(other_ub))
            theta_ub = max(0.0, float(other_lb), float(other_ub))

            for _ in breakpoints:
                lambda_cols.append(col_idx)
                all_bounds.append((0.0, 1.0))
                integrality_flags.append(0)
                col_idx += 1

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
            mode = str(lambda_info["mode"])
            yj_lb, yj_ub = [float(v) for v in all_bounds[other_var]]

            row_sum_lambda = np.zeros(n_total)
            for lambda_col in lambda_cols:
                row_sum_lambda[lambda_col] = -1.0
            _add_row(row_sum_lambda, -1.0)
            _add_row(-row_sum_lambda, 1.0)

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
            # Determine partition var vs other var
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

                # w̄_k ≥ wk_lo * δ_k  → w̄_k=0 when δ_k=0
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

    # Monomial constraints
    for var_idx, n in terms.monomial:
        lb_i = float(flat_lb[var_idx])
        ub_i = float(flat_ub[var_idx])
        s_col = monomial_var_map[(var_idx, n)]

        if n == 2:
            # Piecewise tangent underestimators: s ≥ 2*t*x - t^2  for tangent point t.
            # → -s + 2*t*x ≤ t^2
            #
            # KEY monotonicity requirement: as partitions get finer, the set of tangent
            # points must grow (never shrink).  Using ALL BREAKPOINTS achieves this:
            # linspace(lb,ub,k+1) breakpoints are a superset of linspace(lb,ub,k) for
            # the test sequence n_init=1,2,4,8 (each is a refinement of the previous).
            # Using midpoints would fail because n_init=1's midpoint (2.5) gives a
            # tighter cut than n_init=2's midpoints (1.75, 3.25) at the LP optimum.
            if var_idx in disc_state.partitions and len(disc_state.partitions[var_idx]) >= 2:
                pts = disc_state.partitions[var_idx]
                tangent_pts = [float(p) for p in pts]  # breakpoints as tangent points
            else:
                tangent_pts = [lb_i, ub_i]

            for t in tangent_pts:
                row = np.zeros(n_total)
                row[s_col] = -1.0
                row[var_idx] = 2.0 * t
                _add_row(row, t * t)

            # Global secant overestimator: s ≤ (lb+ub)*x - lb*ub
            # → s - (lb+ub)*x ≤ -lb*ub
            row = np.zeros(n_total)
            row[s_col] = 1.0
            row[var_idx] = -(lb_i + ub_i)
            _add_row(row, -lb_i * ub_i)

        else:
            # General n: secant overestimator
            if abs(ub_i - lb_i) > 1e-12:
                slope = (ub_i**n - lb_i**n) / (ub_i - lb_i)
                intercept = lb_i**n - slope * lb_i
                row = np.zeros(n_total)
                row[s_col] = 1.0
                row[var_idx] = -slope
                _add_row(row, intercept)

            # Tangent at midpoint
            mid = 0.5 * (lb_i + ub_i)
            t_slope = n * (mid ** (n - 1))
            t_intercept = -(n - 1) * (mid**n)
            row = np.zeros(n_total)
            row[s_col] = -1.0
            row[var_idx] = t_slope
            _add_row(row, -t_intercept)

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
                monomial_var_map,
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
            logger.warning(
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
    obj_expr = model._objective.expression
    try:
        c_obj, const_obj = _linearize_expr(
            obj_expr,
            model,
            bilinear_var_map,
            trilinear_var_map,
            monomial_var_map,
            n_total,
        )
        objective_bound_valid = True
    except ValueError:
        # Keep a feasibility objective so the relaxation can still produce a point,
        # but do not treat the LP value as a sound global bound.
        c_obj = np.zeros(n_total)
        const_obj = 0.0
        objective_bound_valid = False

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
        "monomial": monomial_var_map,
        "bilinear_pw": bilinear_pw_map,
        "bilinear_lambda": bilinear_lambda_map,
        "convhull_formulation": convhull_mode,
    }

    return milp_model, varmap
