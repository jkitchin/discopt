"""
Optimality-Based Bound Tightening (OBBT).

For each variable x_i, solves two LPs to find the tightest possible bounds:
  min x_i  subject to linear constraints  -> tightened lower bound
  max x_i  subject to linear constraints  -> tightened upper bound

Uses the HiGHS LP solver with warm-starting for efficiency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import Model
from discopt.solvers import SolveStatus
from discopt.solvers.lp_highs import solve_lp


@dataclass
class ObbtResult:
    """Result of running OBBT on a model."""

    tightened_lb: np.ndarray
    tightened_ub: np.ndarray
    n_lp_solves: int
    n_tightened: int
    total_lp_time: float


def _extract_linear_constraints(
    model: Model,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
]:
    """Extract linear inequality and equality constraints from a model.

    Walks each constraint, extracts coefficients for linear constraints.
    Non-linear constraints are skipped (OBBT uses only the LP relaxation).

    Returns:
        (A_ub, b_ub, A_eq, b_eq, n_vars) or (None, None, None, None, n_vars)
        if no linear constraints found.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        Constraint,
        Expression,
        IndexExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    n_vars = sum(v.size for v in model._variables)

    def _compute_var_offset(var: Variable) -> int:
        offset = 0
        for v in model._variables[: var._index]:
            offset += v.size
        return offset

    def _extract_coeffs(
        expr: Expression,
    ) -> Optional[tuple[dict[int, float], float]]:
        """Extract linear coefficients. Returns (var_idx -> coeff, offset)."""
        if isinstance(expr, Constant):
            return {}, float(np.sum(expr.value))

        if isinstance(expr, Variable):
            offset = _compute_var_offset(expr)
            if expr.size == 1:
                return {offset: 1.0}, 0.0
            return None

        if isinstance(expr, Parameter):
            return {}, float(np.sum(expr.value))

        if isinstance(expr, BinaryOp):
            if expr.op == "+":
                left = _extract_coeffs(expr.left)
                right = _extract_coeffs(expr.right)
                if left is None or right is None:
                    return None
                lc, lo = left
                rc, ro = right
                merged = dict(lc)
                for k, v in rc.items():
                    merged[k] = merged.get(k, 0.0) + v
                return merged, lo + ro

            if expr.op == "-":
                left = _extract_coeffs(expr.left)
                right = _extract_coeffs(expr.right)
                if left is None or right is None:
                    return None
                lc, lo = left
                rc, ro = right
                merged = dict(lc)
                for k, v in rc.items():
                    merged[k] = merged.get(k, 0.0) - v
                return merged, lo - ro

            if expr.op == "*":
                left = _extract_coeffs(expr.left)
                right = _extract_coeffs(expr.right)
                if left is None or right is None:
                    return None
                lc, lo = left
                rc, ro = right
                if not lc:
                    # Left is constant
                    return {k: v * lo for k, v in rc.items()}, lo * ro
                if not rc:
                    # Right is constant
                    return {k: v * ro for k, v in lc.items()}, lo * ro
                return None

            if expr.op == "/":
                left = _extract_coeffs(expr.left)
                right = _extract_coeffs(expr.right)
                if left is None or right is None:
                    return None
                lc, lo = left
                rc, ro = right
                if not rc and abs(ro) > 1e-30:
                    inv = 1.0 / ro
                    return {k: v * inv for k, v in lc.items()}, lo * inv
                return None

            return None

        if isinstance(expr, UnaryOp):
            if expr.op == "neg":
                inner = _extract_coeffs(expr.operand)
                if inner is None:
                    return None
                coeffs, offset = inner
                return {k: -v for k, v in coeffs.items()}, -offset
            return None

        if isinstance(expr, IndexExpression):
            if isinstance(expr.base, Variable):
                base_offset = _compute_var_offset(expr.base)
                idx = expr.index
                if isinstance(idx, int):
                    return {base_offset + idx: 1.0}, 0.0
                if isinstance(idx, tuple) and len(idx) == 1:
                    return {base_offset + idx[0]: 1.0}, 0.0
            return None

        if isinstance(expr, SumExpression):
            return _extract_coeffs(expr.operand)

        if isinstance(expr, SumOverExpression):
            merged: dict[int, float] = {}
            total_offset = 0.0
            for t in expr.terms:
                result = _extract_coeffs(t)
                if result is None:
                    return None
                coeffs, offset = result
                total_offset += offset
                for k, v in coeffs.items():
                    merged[k] = merged.get(k, 0.0) + v
            return merged, total_offset

        return None

    ub_rows: list[tuple[dict[int, float], float]] = []
    eq_rows: list[tuple[dict[int, float], float]] = []

    for constr in model._constraints:
        if not isinstance(constr, Constraint):
            continue

        result = _extract_coeffs(constr.body)
        if result is None:
            continue

        coeffs, offset = result
        adjusted_rhs = constr.rhs - offset

        if constr.sense == "<=":
            ub_rows.append((coeffs, adjusted_rhs))
        elif constr.sense == "==":
            eq_rows.append((coeffs, adjusted_rhs))
        elif constr.sense == ">=":
            # a >= b  <=>  -a <= -b
            neg_coeffs = {k: -v for k, v in coeffs.items()}
            ub_rows.append((neg_coeffs, -adjusted_rhs))

    A_ub = None
    b_ub = None
    if ub_rows:
        A_ub = np.zeros((len(ub_rows), n_vars), dtype=np.float64)
        b_ub = np.zeros(len(ub_rows), dtype=np.float64)
        for i, (coeffs, rhs) in enumerate(ub_rows):
            for var_idx, coeff in coeffs.items():
                if var_idx < n_vars:
                    A_ub[i, var_idx] = coeff
            b_ub[i] = rhs

    A_eq = None
    b_eq = None
    if eq_rows:
        A_eq = np.zeros((len(eq_rows), n_vars), dtype=np.float64)
        b_eq = np.zeros(len(eq_rows), dtype=np.float64)
        for i, (coeffs, rhs) in enumerate(eq_rows):
            for var_idx, coeff in coeffs.items():
                if var_idx < n_vars:
                    A_eq[i, var_idx] = coeff
            b_eq[i] = rhs

    return A_ub, b_ub, A_eq, b_eq, n_vars


def _extract_linear_objective(
    model: Model,
    n_vars: int,
) -> Optional[np.ndarray]:
    """Extract linear objective coefficients, or None if nonlinear."""
    if model._objective is None:
        return None
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        Expression,
        IndexExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    def _compute_var_offset(var: Variable) -> int:
        offset = 0
        for v in model._variables[: var._index]:
            offset += v.size
        return offset

    def _extract(expr: Expression) -> Optional[tuple[dict[int, float], float]]:
        if isinstance(expr, Constant):
            return {}, float(np.sum(expr.value))
        if isinstance(expr, Variable):
            offset = _compute_var_offset(expr)
            if expr.size == 1:
                return {offset: 1.0}, 0.0
            return None
        if isinstance(expr, Parameter):
            return {}, float(np.sum(expr.value))
        if isinstance(expr, BinaryOp):
            if expr.op == "+":
                le = _extract(expr.left)
                ri = _extract(expr.right)
                if le is None or ri is None:
                    return None
                merged = dict(le[0])
                for k, v in ri[0].items():
                    merged[k] = merged.get(k, 0.0) + v
                return merged, le[1] + ri[1]
            if expr.op == "-":
                le = _extract(expr.left)
                ri = _extract(expr.right)
                if le is None or ri is None:
                    return None
                merged = dict(le[0])
                for k, v in ri[0].items():
                    merged[k] = merged.get(k, 0.0) - v
                return merged, le[1] - ri[1]
            if expr.op == "*":
                le = _extract(expr.left)
                ri = _extract(expr.right)
                if le is None or ri is None:
                    return None
                if not le[0]:
                    return {k: v * le[1] for k, v in ri[0].items()}, le[1] * ri[1]
                if not ri[0]:
                    return {k: v * ri[1] for k, v in le[0].items()}, le[1] * ri[1]
                return None
            return None
        if isinstance(expr, UnaryOp) and expr.op == "neg":
            inner = _extract(expr.operand)
            if inner is None:
                return None
            return {k: -v for k, v in inner[0].items()}, -inner[1]
        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            base_offset = _compute_var_offset(expr.base)
            idx = expr.index
            if isinstance(idx, int):
                return {base_offset + idx: 1.0}, 0.0
            if isinstance(idx, tuple) and len(idx) == 1:
                return {base_offset + idx[0]: 1.0}, 0.0
            return None
        if isinstance(expr, SumExpression):
            return _extract(expr.operand)
        if isinstance(expr, SumOverExpression):
            merged: dict[int, float] = {}
            total = 0.0
            for t in expr.terms:
                r = _extract(t)
                if r is None:
                    return None
                total += r[1]
                for k, v in r[0].items():
                    merged[k] = merged.get(k, 0.0) + v
            return merged, total
        return None

    result = _extract(model._objective.expression)
    if result is None:
        return None
    coeffs, _offset = result
    c = np.zeros(n_vars, dtype=np.float64)
    for idx, coeff in coeffs.items():
        if idx < n_vars:
            c[idx] = coeff
    return c


def _get_var_bounds(
    model: Model,
) -> tuple[np.ndarray, np.ndarray]:
    """Get flat variable bounds from the model."""
    lbs = []
    ubs = []
    for v in model._variables:
        lbs.append(v.lb.flatten())
        ubs.append(v.ub.flatten())
    lb = np.concatenate(lbs).astype(np.float64)
    ub = np.concatenate(ubs).astype(np.float64)
    return lb, ub


def run_obbt(
    model: Model,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    min_width: float = 1e-6,
    time_limit_per_lp: Optional[float] = None,
    total_time_limit: Optional[float] = None,
    incumbent_cutoff: Optional[float] = None,
) -> ObbtResult:
    """Run OBBT to tighten variable bounds.

    For each variable with sufficiently wide bounds, solves two LPs
    (min and max) subject to the model's linear constraints to find
    the tightest possible bounds.

    When ``incumbent_cutoff`` is provided, the constraint ``f(x) <= z*``
    is added (using the objective's linear coefficients), which can
    dramatically tighten bounds by excluding regions that cannot improve
    on the incumbent.

    Args:
        model: The optimization model.
        lb: Initial lower bounds. If None, uses model variable bounds.
        ub: Initial upper bounds. If None, uses model variable bounds.
        min_width: Skip variables whose bound width is below this threshold.
        time_limit_per_lp: Time limit per LP solve in seconds.
        total_time_limit: Wall-clock limit for the whole OBBT pass in seconds.
        incumbent_cutoff: If provided, adds ``c'x <= incumbent_cutoff``
            as an additional inequality constraint (using linear objective
            coefficients). Only effective when the objective is linear.

    Returns:
        ObbtResult with tightened bounds and statistics.
    """
    if total_time_limit is not None and total_time_limit < 0.0:
        raise ValueError("total_time_limit must be non-negative")

    model_lb, model_ub = _get_var_bounds(model)
    if lb is None:
        lb = model_lb.copy()
    else:
        lb = np.maximum(lb, model_lb)
    if ub is None:
        ub = model_ub.copy()
    else:
        ub = np.minimum(ub, model_ub)

    n_vars = len(lb)
    A_ub, b_ub, A_eq, b_eq, _ = _extract_linear_constraints(model)

    # --- Add incumbent cutoff constraint: c'x <= z* ---
    if incumbent_cutoff is not None and model._objective is not None:
        from discopt.modeling.core import ObjectiveSense

        obj_coeffs = _extract_linear_objective(model, n_vars)
        if obj_coeffs is not None:
            # For maximization, negate: max c'x equiv min -c'x,
            # so c'x >= z* becomes -c'x <= -z*
            if model._objective.sense == ObjectiveSense.MAXIMIZE:
                cutoff_row = -obj_coeffs.reshape(1, -1)
                cutoff_rhs = np.array([-incumbent_cutoff])
            else:
                cutoff_row = obj_coeffs.reshape(1, -1)
                cutoff_rhs = np.array([incumbent_cutoff])
            if A_ub is not None and b_ub is not None:
                A_ub = np.vstack([A_ub, cutoff_row])
                b_ub = np.concatenate([b_ub, cutoff_rhs])
            else:
                A_ub = cutoff_row
                b_ub = cutoff_rhs

    if A_ub is None and A_eq is None:
        return ObbtResult(
            tightened_lb=lb,
            tightened_ub=ub,
            n_lp_solves=0,
            n_tightened=0,
            total_lp_time=0.0,
        )

    # Identify candidate variables
    candidates = []
    for i in range(n_vars):
        width = ub[i] - lb[i]
        if width > min_width and np.isfinite(lb[i]) and np.isfinite(ub[i]):
            candidates.append(i)

    n_lp_solves = 0
    n_tightened = 0
    total_lp_time = 0.0
    warm_basis = None
    deadline = time.perf_counter() + total_time_limit if total_time_limit is not None else None

    bounds_list = [(float(lb[i]), float(ub[i])) for i in range(n_vars)]

    def _remaining_time() -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.perf_counter())

    def _lp_time_limit() -> Optional[float]:
        remaining = _remaining_time()
        if remaining is not None and remaining <= 0.0:
            return None
        if remaining is None:
            return time_limit_per_lp
        if time_limit_per_lp is None:
            return remaining
        return min(time_limit_per_lp, remaining)

    for var_idx in candidates:
        lp_time_limit = _lp_time_limit()
        if lp_time_limit is None and deadline is not None:
            break

        # Minimize x_i
        c = np.zeros(n_vars, dtype=np.float64)
        c[var_idx] = 1.0

        result = solve_lp(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_list,
            warm_basis=warm_basis,
            time_limit=lp_time_limit,
        )
        n_lp_solves += 1
        total_lp_time += result.wall_time

        if result.status == SolveStatus.OPTIMAL:
            warm_basis = result.basis
            new_lb = result.objective
            if new_lb is not None and new_lb > lb[var_idx] + 1e-8:
                lb[var_idx] = new_lb
                bounds_list[var_idx] = (float(lb[var_idx]), float(ub[var_idx]))
                n_tightened += 1

        # Maximize x_i (minimize -x_i)
        lp_time_limit = _lp_time_limit()
        if lp_time_limit is None and deadline is not None:
            break

        c[var_idx] = -1.0

        result = solve_lp(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_list,
            warm_basis=warm_basis,
            time_limit=lp_time_limit,
        )
        n_lp_solves += 1
        total_lp_time += result.wall_time

        if result.status == SolveStatus.OPTIMAL:
            warm_basis = result.basis
            new_ub = -result.objective if result.objective is not None else None
            if new_ub is not None and new_ub < ub[var_idx] - 1e-8:
                ub[var_idx] = new_ub
                bounds_list[var_idx] = (float(lb[var_idx]), float(ub[var_idx]))
                n_tightened += 1

    return ObbtResult(
        tightened_lb=lb,
        tightened_ub=ub,
        n_lp_solves=n_lp_solves,
        n_tightened=n_tightened,
        total_lp_time=total_lp_time,
    )


def run_obbt_on_relaxation(
    relaxation,
    n_orig: int,
    candidate_idxs: Optional[list[int]] = None,
    time_limit_per_lp: Optional[float] = 1.0,
    incumbent_cutoff: Optional[float] = None,
    min_width: float = 1e-6,
    eps: float = 1e-7,
    deadline: Optional[float] = None,
) -> ObbtResult:
    """OBBT over the LP relaxation of a MILP relaxation envelope.

    For each candidate variable solves min/max x_i subject to the MILP
    relaxation's linear constraints (with integrality dropped), and tightens
    the variable's bound when the LP optimum is strictly inside the prior box.
    The LP feasible region contains the MILP feasible region, which contains
    the original MINLP feasible region, so any tightened bound is valid.

    When ``incumbent_cutoff`` is provided, the constraint
    ``c_obj^T x <= incumbent_cutoff - obj_offset`` is added (using the
    relaxation's objective row), excluding regions that cannot beat the best
    known feasible objective and often delivering a bigger tightening than
    the structural envelopes alone.

    Parameters
    ----------
    relaxation : MilpRelaxationModel
        Built by ``build_milp_relaxation``.
    n_orig : int
        Number of original (non-aux) columns; only these are returned.
    candidate_idxs : list[int], optional
        Subset of column indices to tighten.  Defaults to all original columns.
    time_limit_per_lp : float, optional
        Per-LP time limit in seconds.  ``None`` for no limit.
    incumbent_cutoff : float, optional
        Cutoff on ``c_obj^T x + obj_offset`` (i.e. the relaxation objective in
        the same scale as the user's objective).
    min_width : float
        Skip variables whose box width is below this.
    deadline : float, optional
        Absolute ``time.perf_counter()`` deadline; pass to abort early.

    Returns
    -------
    ObbtResult
        ``tightened_lb`` and ``tightened_ub`` are length-``n_orig`` arrays.
    """
    import time as _time

    import scipy.sparse as sp

    A_ub = relaxation._A_ub
    b_ub = relaxation._b_ub
    bounds = list(relaxation._bounds)
    n_total = len(bounds)
    c_obj = relaxation._c
    obj_offset = float(relaxation._obj_offset)

    if incumbent_cutoff is not None and c_obj is not None:
        cutoff_row = np.asarray(c_obj, dtype=np.float64).reshape(1, -1)
        cutoff_rhs = np.array([float(incumbent_cutoff) - obj_offset], dtype=np.float64)
        if A_ub is not None and b_ub is not None:
            if sp.issparse(A_ub):
                A_ub = sp.vstack([A_ub, sp.csr_matrix(cutoff_row)], format="csr")
            else:
                A_ub = np.vstack([np.asarray(A_ub), cutoff_row])
            b_ub = np.concatenate([np.asarray(b_ub, dtype=np.float64), cutoff_rhs])
        else:
            A_ub = cutoff_row
            b_ub = cutoff_rhs

    lb_arr = np.array([b[0] for b in bounds], dtype=np.float64)
    ub_arr = np.array([b[1] for b in bounds], dtype=np.float64)

    if candidate_idxs is None:
        candidate_idxs = list(range(min(n_orig, n_total)))
    else:
        candidate_idxs = [i for i in candidate_idxs if 0 <= i < n_total]

    n_lp_solves = 0
    n_tightened = 0
    total_lp_time = 0.0
    warm_basis = None

    def _bounds_list() -> list[tuple[float, float]]:
        return [(float(lb_arr[i]), float(ub_arr[i])) for i in range(n_total)]

    for var_idx in candidate_idxs:
        if deadline is not None and _time.perf_counter() >= deadline:
            break
        width = ub_arr[var_idx] - lb_arr[var_idx]
        if width <= min_width:
            continue
        if not (np.isfinite(lb_arr[var_idx]) and np.isfinite(ub_arr[var_idx])):
            continue

        # min x_i
        c = np.zeros(n_total, dtype=np.float64)
        c[var_idx] = 1.0
        result = solve_lp(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=_bounds_list(),
            warm_basis=warm_basis,
            time_limit=time_limit_per_lp,
        )
        n_lp_solves += 1
        total_lp_time += result.wall_time
        if result.status == SolveStatus.OPTIMAL and result.objective is not None:
            warm_basis = result.basis
            new_lb = float(result.objective)
            if new_lb > lb_arr[var_idx] + eps:
                # Don't cross the upper bound.
                lb_arr[var_idx] = min(new_lb, ub_arr[var_idx])
                n_tightened += 1

        if deadline is not None and _time.perf_counter() >= deadline:
            break

        # max x_i (minimize -x_i)
        c[var_idx] = -1.0
        result = solve_lp(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=_bounds_list(),
            warm_basis=warm_basis,
            time_limit=time_limit_per_lp,
        )
        n_lp_solves += 1
        total_lp_time += result.wall_time
        if result.status == SolveStatus.OPTIMAL and result.objective is not None:
            warm_basis = result.basis
            new_ub = -float(result.objective)
            if new_ub < ub_arr[var_idx] - eps:
                ub_arr[var_idx] = max(new_ub, lb_arr[var_idx])
                n_tightened += 1

    return ObbtResult(
        tightened_lb=lb_arr[:n_orig].copy(),
        tightened_ub=ub_arr[:n_orig].copy(),
        n_lp_solves=n_lp_solves,
        n_tightened=n_tightened,
        total_lp_time=total_lp_time,
    )
