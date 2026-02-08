"""
Optimality-Based Bound Tightening (OBBT).

For each variable x_i, solves two LPs to find the tightest possible bounds:
  min x_i  subject to linear constraints  -> tightened lower bound
  max x_i  subject to linear constraints  -> tightened upper bound

Uses the HiGHS LP solver with warm-starting for efficiency.
"""

from __future__ import annotations

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
) -> ObbtResult:
    """Run OBBT to tighten variable bounds.

    For each variable with sufficiently wide bounds, solves two LPs
    (min and max) subject to the model's linear constraints to find
    the tightest possible bounds.

    Args:
        model: The optimization model.
        lb: Initial lower bounds. If None, uses model variable bounds.
        ub: Initial upper bounds. If None, uses model variable bounds.
        min_width: Skip variables whose bound width is below this threshold.
        time_limit_per_lp: Time limit per LP solve in seconds.

    Returns:
        ObbtResult with tightened bounds and statistics.
    """
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

    bounds_list = [(float(lb[i]), float(ub[i])) for i in range(n_vars)]

    for var_idx in candidates:
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
            time_limit=time_limit_per_lp,
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
        c[var_idx] = -1.0

        result = solve_lp(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_list,
            warm_basis=warm_basis,
            time_limit=time_limit_per_lp,
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
