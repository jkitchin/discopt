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
from discopt.solvers.lp_backend import (
    get_exact_dual_lp_solver,
    get_exact_lp_solver,
    get_lp_solver,
)

# Beyond this magnitude the McCormick relaxation's LP is too ill-conditioned for
# an OBBT tightening to be rigorous. OBBT shrinks a variable's domain to an LP
# optimum, and a feasibility tolerance ``ftol`` against a constraint coefficient
# of size ``M`` lets the LP vertex sit ~``M*ftol`` past the true feasible point
# in that variable. Once that absolute slack approaches the integrality /
# feasibility tolerances the tightening can cut off a feasible (even optimal)
# point — e.g. the fractional power ``x**-3.5`` near 0 builds envelope slopes
# ~1e14 and OBBT then prunes the true optimum (the nvs08/fp soundness model).
# When the relaxation's largest magnitude exceeds this limit OBBT returns the
# box untightened: skipping is always sound, only weaker. Well-conditioned
# stiff models (e.g. the 1e6-coefficient #145 ratio) stay well under it and are
# still tightened by the exact (Rust simplex / HiGHS) oracle.
_OBBT_COND_LIMIT = 1e10

# OBBT tightens to the LP vertex objective, which is accurate on a well-conditioned
# basis but can drift a few ``ulp * cond`` *above* the true optimum when the
# McCormick coefficient spread makes the basis ill-conditioned (the nvs22
# false-certificate: a 1.000996 vertex for a true min of 1.0, rounded inward to
# prune the optimum). The Neumaier-Shcherbina safe dual bound is a rigorous lower
# bound on that LP min; when the vertex sits this far *above* it (relative), the
# vertex is untrustworthy and OBBT falls back to the safe bound instead of
# over-tightening. The threshold is far above honest vertex noise (~1e-9 on
# well-conditioned LPs) so trustworthy tightenings are preserved bit-for-bit —
# avoiding spurious box perturbations on knife-edge instances (st_e11) — while the
# gross ill-conditioned drift that prunes a feasible optimum is caught.
_OBBT_NS_GUARD = 1e-6


def _ns_safe_lp_lower_bound(c, dual_values, A_ub, b_ub, lo, hi):
    """Neumaier-Shcherbina rigorous lower bound on ``min c^T x`` over the box LP.

    The LP is ``min c^T x  s.t.  A_ub x <= b_ub,  lo <= x <= hi``. For *any*
    ``y >= 0`` (multipliers on the ``<=`` rows), weak duality gives

        g(y) = -b^T y + sum_j  min_{x_j in [lo_j, hi_j]} (c + A^T y)_j x_j

    and ``g(y) <= true min c^T x``. We take ``y = max(-row_dual, 0)`` from the
    LP's reported duals: HiGHS returns ``row_dual <= 0`` on an active ``<=`` row,
    so ``-row_dual`` is the non-negative multiplier and ``g(y)`` reproduces the
    LP optimum *exactly* on a well-conditioned solve. The crucial property is
    that ``g(y) <= true min`` holds for the clamped ``y`` regardless of how the
    duals were obtained, so when the simplex basis is ill-conditioned (a wide
    McCormick coefficient spread) and the reported vertex objective drifts a few
    ``ulp * cond`` *above* the true minimum, ``g(y)`` stays a sound under-estimate
    rather than an over-tightening that could prune the global optimum (the
    nvs22 false-certificate). Tightening OBBT bounds to ``g(y)`` instead of the
    raw vertex objective is therefore sound at any conditioning, and loses no
    tightness where the LP is well-behaved.

    A magnitude-scaled margin is subtracted to dominate the floating-point error
    of the dual evaluation itself (done in plain float64, not directed-rounding
    interval arithmetic). Returns ``None`` when no usable lower bound exists.
    """
    import scipy.sparse as sp

    if dual_values is None:
        return None
    y = np.array(dual_values, dtype=np.float64)  # copy: must not mutate caller's duals
    if y.size == 0:
        return None
    np.clip(-y, 0.0, None, out=y)  # y := max(-row_dual, 0) >= 0
    c = np.asarray(c, dtype=np.float64)
    if A_ub is None or b_ub is None:
        rc = c.copy()
        const = 0.0
    else:
        b_arr = np.asarray(b_ub, dtype=np.float64)
        if y.shape[0] != b_arr.shape[0]:
            return None
        At_y = (A_ub.T @ y) if sp.issparse(A_ub) else (np.asarray(A_ub).T @ y)
        rc = c + np.asarray(At_y).ravel()
        const = -float(b_arr @ y)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    # min_{x_j in [lo_j,hi_j]} rc_j x_j  = lo_j if rc_j>0, hi_j if rc_j<0, else 0
    # (the rc_j==0 case contributes 0 even when that bound is infinite).
    contrib = np.zeros_like(rc)
    pos = rc > 0.0
    neg = rc < 0.0
    contrib[pos] = rc[pos] * lo[pos]
    contrib[neg] = rc[neg] * hi[neg]
    g = const + float(contrib.sum())
    if not np.isfinite(g):
        return None
    # Margin >> the float64 rounding error of the two dot products above
    # (~n * eps * magnitude); keeps g a rigorous under-estimate.
    margin = 1e-10 * (1.0 + abs(const) + float(np.abs(contrib).sum()))
    return g - margin


def _equilibrate_rows(A_ub, b_ub):
    """Row-equilibrate an inequality system ``A x <= b`` for an OBBT projection LP.

    Returns ``(A_s, b_s)`` where each row ``i`` is scaled by a positive factor
    ``d_i = 1 / max(||A[i,:]||_inf, |b_i|, 1)``. Multiplying a ``<=`` row by a
    positive constant is an *exact equivalence* — it leaves the feasible set, and
    therefore the optimum of every ``min/max x_j`` projection, unchanged in exact
    arithmetic. What it changes is the *floating-point* conditioning of the
    simplex basis: an OBBT relaxation can mix coefficients spanning many orders of
    magnitude (cleared-denominator / lifted-monomial rows reach ~1e9), and solving
    an ill-conditioned basis ``B y = b`` amplifies the unit roundoff by ``cond(B)``,
    so the reported vertex — and hence ``min/max x_j`` — can sit ~``cond(B)*u`` past
    the true projection. On an *integer* variable that sub-unit slack rounds inward
    to a full unit and can prune the global optimum, producing a certified-WRONG
    answer (nvs22). Equilibrating the rows to unit infinity-norm removes that
    spurious spread before the exact solve, so the vertex the simplex returns is
    accurate and the tightening is rigorous. The objective ``c`` (a unit ``±e_j``)
    and the variable bounds are untouched, so the returned objective is directly
    the bound; a warm basis stays valid because row scaling does not change which
    constraints are active.
    """
    import scipy.sparse as sp

    if A_ub is None or b_ub is None:
        return A_ub, b_ub
    b_arr = np.asarray(b_ub, dtype=np.float64)
    if sp.issparse(A_ub):
        A_csr = A_ub.tocsr()
        # Per-row infinity norm: abs() makes all stored entries non-negative, so
        # the row max (which also accounts for implicit zeros) is ||A[i,:]||_inf.
        row_inf = np.abs(A_csr).max(axis=1).toarray().ravel()
        scale = np.maximum.reduce([row_inf, np.abs(b_arr), np.ones_like(row_inf)])
        d = 1.0 / scale
        return (sp.diags(d) @ A_csr).tocsr(), b_arr * d
    A_arr = np.asarray(A_ub, dtype=np.float64)
    row_inf = np.abs(A_arr).max(axis=1) if A_arr.size else np.zeros(A_arr.shape[0])
    scale = np.maximum.reduce([row_inf, np.abs(b_arr), np.ones_like(row_inf)])
    d = 1.0 / scale
    return A_arr * d[:, None], b_arr * d


def solve_lp(**kwargs):
    """Module-level default LP solve (HiGHS-first, POUNCE fallback).

    Indirected through the backend selector so discopt runs POUNCE-only
    and so tests can monkeypatch this symbol. Resolved lazily — importing
    this module never requires a solver to be installed.
    """
    return get_lp_solver(prefer_pounce=False)(**kwargs)


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
    prefer_pounce: bool = False,
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

    # OBBT requires an EXACT LP oracle for sound tightening — see
    # ``run_obbt_on_relaxation`` / ``get_exact_lp_solver`` (issue #145). The IPM
    # is never used here; if no exact (Rust simplex / HiGHS) oracle is available
    # the pass is a sound no-op. ``prefer_pounce`` is kept for compatibility.
    _lp = get_exact_lp_solver()
    if _lp is None:
        eff_lb = lb if lb is not None else _get_var_bounds(model)[0].copy()
        eff_ub = ub if ub is not None else _get_var_bounds(model)[1].copy()
        return ObbtResult(
            tightened_lb=np.asarray(eff_lb, dtype=np.float64),
            tightened_ub=np.asarray(eff_ub, dtype=np.float64),
            n_lp_solves=0,
            n_tightened=0,
            total_lp_time=0.0,
        )
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

        result = _lp(
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

        result = _lp(
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
    prefer_pounce: bool = False,
    full_result: bool = False,
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
    full_result : bool, optional
        When ``True`` the returned ``tightened_lb`` / ``tightened_ub`` span all
        ``n_total`` columns (original + auxiliary), not just the first ``n_orig``.
        Used by the OBBT-on-aux diagnostic (#208) to *capture* the tightening of
        the lifted (product/ratio) aux columns that the default path discards.
        Does not change the default (``False``) behavior for solve-path callers.
    """
    import time as _time

    import scipy.sparse as sp

    # OBBT must solve its min/max subproblems with an EXACT LP oracle: the
    # tightened bound is taken from the LP optimum, so an inexact optimum (the
    # POUNCE IPM's analytic-center objective, which can be grossly wrong on
    # ill-conditioned LPs while still reporting OPTIMAL) yields an unsound
    # tightening that cuts off feasible points (issue #145). ``prefer_pounce``
    # is accepted for signature compatibility but never honoured here — when no
    # exact (Rust simplex / HiGHS) oracle is available we return the box
    # untightened rather than risk an unsound shrink.
    # Prefer the dual-exposing (HiGHS) oracle: OBBT tightens to a Neumaier-
    # Shcherbina *safe* dual bound (``_ns_safe_lp_lower_bound``) rather than the
    # raw vertex objective, which is sound at any conditioning and cannot prune a
    # feasible optimum (the nvs22 false-certificate). Without a dual oracle we
    # fall back to the exact vertex value (the conditioning guard below keeps that
    # path from over-tightening on the worst LPs).
    _dual_lp = get_exact_dual_lp_solver()
    _lp = _dual_lp or get_exact_lp_solver()
    _use_ns = _dual_lp is not None
    if _lp is None:
        return ObbtResult(
            tightened_lb=np.array([b[0] for b in relaxation._bounds[:n_orig]], dtype=np.float64),
            tightened_ub=np.array([b[1] for b in relaxation._bounds[:n_orig]], dtype=np.float64),
            n_lp_solves=0,
            n_tightened=0,
            total_lp_time=0.0,
        )
    A_ub = relaxation._A_ub
    b_ub = relaxation._b_ub
    bounds = list(relaxation._bounds)
    n_total = len(bounds)
    c_obj = relaxation._c
    obj_offset = float(relaxation._obj_offset)

    # Refuse to tighten over a numerically ill-conditioned relaxation: an OBBT
    # bound from such an LP is not rigorous and can prune feasible points (see
    # ``_OBBT_COND_LIMIT``). Returning the box untightened is sound.
    def _max_abs(x) -> float:
        if x is None:
            return 0.0
        if sp.issparse(x):
            return float(np.abs(x.data).max()) if x.nnz else 0.0
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        finite = arr[np.isfinite(arr)]
        return float(np.abs(finite).max()) if finite.size else 0.0

    _bound_vals = np.array([v for pair in bounds for v in pair], dtype=np.float64)
    _cond = max(_max_abs(A_ub), _max_abs(b_ub), _max_abs(_bound_vals))
    # Conditioning guard. The raw vertex objective from an ill-conditioned LP can
    # drift *above* the true minimum and over-tighten (the nvs22 false-certificate),
    # so without a rigorous safe-bound oracle we abstain entirely. WITH the NS-safe
    # dual path (``_use_ns``) every tightening below is clamped to the rigorous
    # Neumaier-Shcherbina bound ``g`` -- valid for any dual ``y >= 0`` by weak
    # duality, hence sound at ANY conditioning. So rather than abstain we only
    # *require* that rigorous bound (``require_ns``): a variable whose safe bound is
    # unavailable is left untightened instead of being trusted to its vertex. This
    # lets OBBT engage on ill-conditioned factorable relaxations the bare guard
    # disabled -- e.g. nvs05, whose 1e15-coefficient term drives ``_cond ~ 2e13``
    # yet whose boxes (x5,x6 spanning [~0, 1.36e4]) only shrink once OBBT runs --
    # while never tightening from a non-rigorous value.
    require_ns = _cond > _OBBT_COND_LIMIT
    if require_ns and not _use_ns:
        return ObbtResult(
            tightened_lb=np.array([b[0] for b in bounds[:n_orig]], dtype=np.float64),
            tightened_ub=np.array([b[1] for b in bounds[:n_orig]], dtype=np.float64),
            n_lp_solves=0,
            n_tightened=0,
            total_lp_time=0.0,
        )

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

    # Row-equilibrate the projection LP so an ill-conditioned (wide-coefficient)
    # relaxation cannot return an inaccurate vertex that over-tightens — and, on
    # an integer variable, rounds inward to prune the global optimum (the nvs22
    # false-certificate). Exact equivalence on the feasible set; see
    # ``_equilibrate_rows``. Done once here (after the cutoff row is appended) and
    # reused for every min/max solve in this box.
    A_ub, b_ub = _equilibrate_rows(A_ub, b_ub)

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
        result = _lp(
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
            if _use_ns:
                # Distrust the vertex only when the rigorous safe bound sits well
                # below it (ill-conditioned basis); otherwise keep it bit-for-bit.
                g = _ns_safe_lp_lower_bound(
                    c, getattr(result, "dual_values", None), A_ub, b_ub, lb_arr, ub_arr
                )
                if g is None:
                    if require_ns:
                        # Ill-conditioned LP and no rigorous safe bound available:
                        # the raw vertex is untrustworthy here, so do not tighten.
                        new_lb = None
                elif new_lb > g + _OBBT_NS_GUARD * (1.0 + abs(new_lb)):
                    new_lb = g
            if new_lb is not None and new_lb > lb_arr[var_idx] + eps:
                # Don't cross the upper bound.
                lb_arr[var_idx] = min(new_lb, ub_arr[var_idx])
                n_tightened += 1

        if deadline is not None and _time.perf_counter() >= deadline:
            break

        # max x_i (minimize -x_i)
        c[var_idx] = -1.0
        result = _lp(
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
            lp_min = float(result.objective)  # = min(-x_i) = -max(x_i)
            if _use_ns:
                # Same guard: ``g`` bounds ``min(-x_i)`` from below, so when the
                # vertex sits well above it the basis is ill-conditioned and the
                # safe bound ``-g`` is the sound (looser) upper bound on ``x_i``.
                g = _ns_safe_lp_lower_bound(
                    c, getattr(result, "dual_values", None), A_ub, b_ub, lb_arr, ub_arr
                )
                if g is None:
                    if require_ns:
                        lp_min = None
                elif lp_min > g + _OBBT_NS_GUARD * (1.0 + abs(lp_min)):
                    lp_min = g
            if lp_min is not None:
                new_ub = -lp_min
                if new_ub < ub_arr[var_idx] - eps:
                    ub_arr[var_idx] = max(new_ub, lb_arr[var_idx])
                    n_tightened += 1

    keep = n_total if full_result else n_orig
    return ObbtResult(
        tightened_lb=lb_arr[:keep].copy(),
        tightened_ub=ub_arr[:keep].copy(),
        n_lp_solves=n_lp_solves,
        n_tightened=n_tightened,
        total_lp_time=total_lp_time,
    )


def dbbt_on_relaxation(
    relaxation,
    n_orig: int,
    incumbent_cutoff: float,
    *,
    rc_tol: float = 1e-7,
    eps: float = 1e-7,
    time_limit_per_lp: Optional[float] = 1.0,
) -> ObbtResult:
    """Duality-based bound tightening (DBBT) from the relaxation LP reduced costs.

    Solves the McCormick LP relaxation once, minimizing the objective, to obtain
    a valid dual bound ``z_lp`` and exact reduced costs ``d``. With a valid
    incumbent ``z_inc = incumbent_cutoff`` (an upper bound on the optimum), LP
    duality gives, for every feasible point with objective ``<= z_inc``,

        d_j * (x_j - bound_j) <= z_inc - z_lp =: gap   (>= 0),

    where complementary slackness presses a variable with ``d_j > 0`` to its
    lower bound and one with ``d_j < 0`` to its upper bound. Hence

        d_j >  rc_tol:  x_j <= lb_j + gap / d_j
        d_j < -rc_tol:  x_j >= ub_j - gap / |d_j|.

    This is the continuous analogue of integer reduced-cost fixing. The
    relaxation is an OUTER approximation and ``z_lp`` / ``z_inc`` are valid
    bounds, so the true optimum satisfies these — DBBT never removes it. A single
    LP solve tightens every variable at once, far cheaper than OBBT's ``2n``
    solves. Tightening only: any failure (no exact oracle, ill-conditioned LP,
    non-optimal solve, missing reduced costs, or no finite cutoff) returns the
    box unchanged, so it can never make the solve unsound.
    """
    import scipy.sparse as sp

    bounds = list(relaxation._bounds)
    base_lb = np.array([b[0] for b in bounds[:n_orig]], dtype=np.float64)
    base_ub = np.array([b[1] for b in bounds[:n_orig]], dtype=np.float64)

    def _noop() -> ObbtResult:
        return ObbtResult(base_lb.copy(), base_ub.copy(), 0, 0, 0.0)

    if incumbent_cutoff is None or not np.isfinite(incumbent_cutoff):
        return _noop()
    # DBBT needs exact *reduced costs*; only a dual-exposing exact oracle (HiGHS)
    # qualifies. With none available it no-ops soundly (OBBT still runs).
    _lp = get_exact_dual_lp_solver()
    if _lp is None:
        return _noop()
    c_obj = relaxation._c
    if c_obj is None:
        return _noop()
    A_ub = relaxation._A_ub
    b_ub = relaxation._b_ub
    obj_offset = float(relaxation._obj_offset)

    # Same conditioning guard as OBBT: reduced costs from an ill-conditioned LP
    # are not rigorous, so abstain (sound) rather than risk an unsound tightening.
    def _max_abs(x) -> float:
        if x is None:
            return 0.0
        if sp.issparse(x):
            return float(np.abs(x.data).max()) if x.nnz else 0.0
        arr = np.asarray(x, dtype=np.float64)
        finite = arr[np.isfinite(arr)] if arr.size else arr
        return float(np.abs(finite).max()) if finite.size else 0.0

    _bound_vals = np.array([v for pair in bounds for v in pair], dtype=np.float64)
    if max(_max_abs(A_ub), _max_abs(b_ub), _max_abs(_bound_vals)) > _OBBT_COND_LIMIT:
        return _noop()

    # One LP: minimize the relaxation objective over the full polytope. We do NOT
    # append the cutoff row here — we want the relaxation's own reduced costs and
    # use the cutoff only to size the gap.
    result = _lp(
        c=np.asarray(c_obj, dtype=np.float64),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=list(bounds),
        time_limit=time_limit_per_lp,
    )
    if (
        result.status != SolveStatus.OPTIMAL
        or result.objective is None
        or getattr(result, "reduced_costs", None) is None
    ):
        return _noop()

    z_lp = float(result.objective) + obj_offset
    gap = float(incumbent_cutoff) - z_lp
    if not np.isfinite(gap) or gap < 0.0:
        # gap < 0 means the relaxation already excludes the incumbent value; the
        # caller's OBBT-with-cutoff pass detects emptiness. Stay tightening-only.
        return _noop()
    # Safety margin so any residual dual tolerance cannot over-tighten.
    gap += 1e-6 * (1.0 + abs(float(incumbent_cutoff)))

    d = np.asarray(result.reduced_costs, dtype=np.float64)
    lb = base_lb.copy()
    ub = base_ub.copy()
    n_tightened = 0
    for j in range(min(n_orig, d.shape[0])):
        dj = float(d[j])
        if not np.isfinite(dj):
            continue
        if dj > rc_tol:
            new_ub = lb[j] + gap / dj
            if new_ub < ub[j] - eps:
                ub[j] = max(lb[j], new_ub)
                n_tightened += 1
        elif dj < -rc_tol:
            new_lb = ub[j] - gap / (-dj)
            if new_lb > lb[j] + eps:
                lb[j] = min(ub[j], new_lb)
                n_tightened += 1

    return ObbtResult(
        tightened_lb=lb,
        tightened_ub=ub,
        n_lp_solves=1,
        n_tightened=n_tightened,
        total_lp_time=float(getattr(result, "wall_time", 0.0)),
    )


@dataclass
class RootObbtResult:
    """Result of a structural root OBBT pass over the McCormick relaxation."""

    lb: np.ndarray
    ub: np.ndarray
    n_tightened: int
    n_rounds: int
    total_lp_time: float
    infeasible: bool = False


def _scalar_flat_index(expr, model) -> Optional[int]:
    """Flat column index if *expr* is a single scalar variable reference.

    Returns the flat index (matching :func:`_get_var_bounds` ordering) for a
    scalar ``Variable`` or a scalar ``IndexExpression`` over a variable, else
    ``None``.
    """
    from discopt.modeling.core import IndexExpression, Variable

    def _offset(var) -> int:
        off = 0
        for v in model._variables[: var._index]:
            off += v.size
        return off

    if isinstance(expr, Variable) and expr.size == 1:
        return _offset(expr)
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        idx = expr.index
        if isinstance(idx, int):
            return _offset(expr.base) + idx
        if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
            return _offset(expr.base) + idx[0]
    return None


def _flat_indices(expr, model) -> set:
    """Conservative set of flat variable indices referenced anywhere in *expr*.

    A scalar reference contributes its single index; a whole multi-element
    variable or an unresolved index contributes its entire flat range.  Used to
    verify the isolated variable appears in no other additive term — erring
    toward *over*-inclusion (it may skip a valid isolation, never assert a false
    one).
    """
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    def _offset(var) -> int:
        off = 0
        for v in model._variables[: var._index]:
            off += v.size
        return off

    out: set = set()

    def walk(e) -> None:
        scalar = _scalar_flat_index(e, model)
        if scalar is not None:
            out.add(scalar)
            return
        if isinstance(e, Variable):
            base = _offset(e)
            out.update(range(base, base + e.size))
            return
        if isinstance(e, IndexExpression) and isinstance(e.base, Variable):
            base = _offset(e.base)
            out.update(range(base, base + e.base.size))
            return
        if isinstance(e, BinaryOp):
            walk(e.left)
            walk(e.right)
            return
        if isinstance(e, UnaryOp):
            walk(e.operand)
            return
        if isinstance(e, FunctionCall):
            for a in e.args:
                walk(a)
            return
        if isinstance(e, SumExpression):
            walk(e.operand)
            return
        if isinstance(e, SumOverExpression):
            for t in e.terms:
                walk(t)
            return

    walk(expr)
    return out


def _additive_terms(expr, sign: float = 1.0):
    """Flatten the top-level additive structure into ``(coeff, subexpr)`` pairs.

    Descends through ``+``, ``-`` and unary ``neg`` only; a ``coeff * leaf`` or
    ``leaf * coeff`` product contributes its constant scale.  Anything else is
    returned as a single ``(sign, expr)`` leaf.  Used to spot a constraint of
    the form ``c*v + g(other vars) == rhs`` so ``v`` can be isolated.
    """
    from discopt.modeling.core import BinaryOp, Constant, UnaryOp

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            return _additive_terms(expr.left, sign) + _additive_terms(expr.right, sign)
        if expr.op == "-":
            return _additive_terms(expr.left, sign) + _additive_terms(expr.right, -sign)
        if expr.op == "*":
            if isinstance(expr.left, Constant) and expr.left.value.ndim == 0:
                return _additive_terms(expr.right, sign * float(expr.left.value))
            if isinstance(expr.right, Constant) and expr.right.value.ndim == 0:
                return _additive_terms(expr.left, sign * float(expr.right.value))
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _additive_terms(expr.operand, -sign)
    return [(sign, expr)]


def propagate_equality_defined_bounds(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    max_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Finitize equality-defined variables by forward interval propagation.

    A ``.nl`` model commonly defines an intermediate quantity with an equality
    ``v == g(x)`` (encoded as ``c*v + (-g) == rhs``), where ``g`` is a division,
    log, exp or other expression that interval FBBT in the Rust core does not
    propagate through — so ``v`` keeps its declared ``±inf`` bound even though
    ``g`` has a finite range once the variables it depends on are bounded.  This
    pass closes that gap: for every equality constraint that is *linear in a
    single open variable* ``v`` (``c*v + g == rhs`` with ``v`` appearing nowhere
    else in the row), it bounds ``v == (rhs - g)/c`` by evaluating ``g``'s
    interval over the current box and tightens ``v``.

    This is sound: the equality is exact, so the propagated interval is the
    variable's true reachable range under that constraint — the pass can only
    shrink the box.  It runs to a fixpoint (``max_passes``) because finitizing
    one variable can unlock another (e.g. ``v == g(w)`` once ``w`` is finite).

    Pairs with :func:`bootstrap_finite_bounds`: the bootstrap finitizes the
    variables reachable from *linear* constraints, which then become finite
    inputs to the ``g`` expressions this pass propagates through.

    Returns ``(lb, ub, n_finitized)``.  Any failure leaves the box unchanged.
    """
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    n_vars = len(lb)

    if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
        return lb, ub, 0

    try:
        from discopt._jax.gdp_reformulate import _bound_expression
        from discopt.modeling.core import Constraint, VarType

        # _bound_expression reads bounds off the model's Variable nodes, so drive
        # it with the current box by temporarily writing (lb, ub) onto the model
        # and restoring the originals afterwards.
        saved = [(v.lb, v.ub) for v in model._variables]
        is_int = np.zeros(n_vars, dtype=bool)

        def _apply_box() -> None:
            off = 0
            for v in model._variables:
                sz = v.size
                v.lb = lb[off : off + sz].reshape(v.lb.shape)
                v.ub = ub[off : off + sz].reshape(v.ub.shape)
                off += sz

        off = 0
        for v in model._variables:
            flag = v.var_type in (VarType.BINARY, VarType.INTEGER)
            for _ in range(v.size):
                if off < n_vars:
                    is_int[off] = flag
                off += 1

        eq_constraints = [
            c for c in model._constraints if isinstance(c, Constraint) and c.sense == "=="
        ]

        n_finitized = 0
        try:
            for _ in range(max(1, max_passes)):
                _apply_box()
                changed = False
                for c in eq_constraints:
                    terms = _additive_terms(c.body)
                    rhs = float(c.rhs) if np.ndim(c.rhs) == 0 else None
                    if rhs is None:
                        continue
                    for pos, (coeff, sub) in enumerate(terms):
                        if abs(coeff) < 1e-30:
                            continue
                        vi = _scalar_flat_index(sub, model)
                        if vi is None or vi >= n_vars:
                            continue
                        if np.isfinite(lb[vi]) and np.isfinite(ub[vi]):
                            continue
                        # v must not appear in any other term of the row.
                        others = [t for k, t in enumerate(terms) if k != pos]
                        if any(vi in _flat_indices(t[1], model) for t in others):
                            continue
                        # v == (rhs - sum(others)) / coeff
                        rest_lo, rest_hi = 0.0, 0.0
                        ok = True
                        for ocoeff, oexpr in others:
                            elo, ehi = _bound_expression(oexpr, model)
                            if ocoeff >= 0:
                                rest_lo += ocoeff * elo
                                rest_hi += ocoeff * ehi
                            else:
                                rest_lo += ocoeff * ehi
                                rest_hi += ocoeff * elo
                            if not (np.isfinite(rest_lo) and np.isfinite(rest_hi)):
                                ok = False
                                break
                        if not ok:
                            continue
                        num_lo = rhs - rest_hi
                        num_hi = rhs - rest_lo
                        if coeff > 0:
                            v_lo, v_hi = num_lo / coeff, num_hi / coeff
                        else:
                            v_lo, v_hi = num_hi / coeff, num_lo / coeff
                        if not (np.isfinite(v_lo) and np.isfinite(v_hi)):
                            continue
                        margin = 1e-6 * (1.0 + max(abs(v_lo), abs(v_hi)))
                        v_lo -= margin
                        v_hi += margin
                        if is_int[vi]:
                            v_lo = np.ceil(v_lo - 1e-7)
                            v_hi = np.floor(v_hi + 1e-7)
                        new_lb = max(lb[vi], v_lo) if np.isfinite(lb[vi]) else v_lo
                        new_ub = min(ub[vi], v_hi) if np.isfinite(ub[vi]) else v_hi
                        if (
                            new_lb > lb[vi] + 1e-12
                            or new_ub < ub[vi] - 1e-12
                            or (not np.isfinite(lb[vi]) or not np.isfinite(ub[vi]))
                        ):
                            if (np.isfinite(new_lb) != np.isfinite(lb[vi])) or (
                                np.isfinite(new_ub) != np.isfinite(ub[vi])
                            ):
                                changed = True
                            lb[vi] = new_lb
                            ub[vi] = new_ub
                            n_finitized += 1
                if not changed:
                    break
        finally:
            for v, (olb, oub) in zip(model._variables, saved):
                v.lb = olb
                v.ub = oub

        return lb, ub, n_finitized
    except Exception:
        return lb, ub, 0


def bootstrap_finite_bounds(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    deadline: Optional[float] = None,
    time_limit_per_lp: float = 0.2,
    incumbent_cutoff: Optional[float] = None,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Finitize open variable bounds via LP over the model's linear subsystem.

    Both OBBT and the rounding/diving primal heuristics need a finite box:
    :func:`obbt_tighten_root` bails the moment any bound is ``±inf`` (it cannot
    build McCormick envelopes on an open variable), and a heuristic cannot
    sample an unbounded box.  A model whose only finite information about a
    variable flows through *nonlinear* constraints (which interval FBBT cannot
    propagate tightly) therefore stalls — OBBT could derive a finite bound, but
    it refuses to start because the box is not already finite.

    This bootstrap breaks that chicken-and-egg by deriving finite bounds for
    the open variables directly from the model's *linear* constraints.  The
    linear feasible region is a polyhedral OUTER approximation of the MINLP
    feasible set, so ``max x_i`` / ``min x_i`` over it is a rigorously valid
    bound on the true problem — the pass can only ever shrink the admissible
    region, never cut off a feasible point.

    For each variable with ``ub == +inf`` we maximize ``x_i``; for ``lb ==
    -inf`` we minimize ``x_i``.  A bounded LP optimum (plus a small outward
    safety margin that keeps the bound rigorous under float / LP-conditioning
    error) becomes the new finite bound; an unbounded direction leaves the
    bound open.  Integer bounds are rounded inward.

    Returns ``(lb, ub, n_finitized, total_lp_time)``.  Any internal failure
    returns the input box unchanged — it can never make the solve unsound.
    """
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    n_vars = len(lb)

    open_lb = ~np.isfinite(lb)
    open_ub = ~np.isfinite(ub)
    if not (open_lb.any() or open_ub.any()):
        # Common case: box already finite, nothing to bootstrap.
        return lb, ub, 0, 0.0

    # OBBT-grade soundness requires an EXACT LP oracle: the finitized bound is
    # the LP optimum, so an inexact optimum could cut off a feasible point
    # (issue #145).  Without one this is a sound no-op.
    _lp = get_exact_lp_solver()
    if _lp is None:
        return lb, ub, 0, 0.0

    try:
        from discopt.modeling.core import VarType

        A_ub, b_ub, A_eq, b_eq, _ = _extract_linear_constraints(model)

        # Optimality-based finitization: fold the incumbent cutoff into the
        # polytope so an open variable that can only grow by worsening the
        # objective gets a finite bound from the cutoff alone.
        if incumbent_cutoff is not None and model._objective is not None:
            from discopt.modeling.core import ObjectiveSense

            obj_coeffs = _extract_linear_objective(model, n_vars)
            if obj_coeffs is not None and np.any(obj_coeffs):
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
            # No linear structure to bound against.
            return lb, ub, 0, 0.0

        is_int = np.zeros(n_vars, dtype=bool)
        flat_idx = 0
        for v in model._variables:
            flag = v.var_type in (VarType.BINARY, VarType.INTEGER)
            for _ in range(v.size):
                if flat_idx < n_vars:
                    is_int[flat_idx] = flag
                flat_idx += 1

        bounds_list = [(float(lb[i]), float(ub[i])) for i in range(n_vars)]
        n_finitized = 0
        total_lp_time = 0.0
        warm_basis = None

        # Only variables that are open in at least one direction are targets.
        targets = np.where(open_lb | open_ub)[0]
        for var_idx in targets:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            # Lower bound: minimize x_i (only when currently open below).
            if open_lb[var_idx]:
                c = np.zeros(n_vars, dtype=np.float64)
                c[var_idx] = 1.0
                result = _lp(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds_list,
                    warm_basis=warm_basis,
                    time_limit=time_limit_per_lp,
                )
                total_lp_time += result.wall_time
                if (
                    result.status == SolveStatus.OPTIMAL
                    and result.objective is not None
                    and np.isfinite(result.objective)
                ):
                    warm_basis = result.basis
                    margin = 1e-6 * (1.0 + abs(result.objective))
                    new_lb = result.objective - margin
                    if is_int[var_idx]:
                        new_lb = np.ceil(new_lb - eps)
                    lb[var_idx] = new_lb
                    bounds_list[var_idx] = (float(lb[var_idx]), float(ub[var_idx]))
                    n_finitized += 1

            if deadline is not None and time.perf_counter() >= deadline:
                break

            # Upper bound: maximize x_i (minimize -x_i) when currently open above.
            if open_ub[var_idx]:
                c = np.zeros(n_vars, dtype=np.float64)
                c[var_idx] = -1.0
                result = _lp(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds_list,
                    warm_basis=warm_basis,
                    time_limit=time_limit_per_lp,
                )
                total_lp_time += result.wall_time
                if (
                    result.status == SolveStatus.OPTIMAL
                    and result.objective is not None
                    and np.isfinite(result.objective)
                ):
                    warm_basis = result.basis
                    new_ub = -result.objective
                    margin = 1e-6 * (1.0 + abs(new_ub))
                    new_ub = new_ub + margin
                    if is_int[var_idx]:
                        new_ub = np.floor(new_ub + eps)
                    ub[var_idx] = new_ub
                    bounds_list[var_idx] = (float(lb[var_idx]), float(ub[var_idx]))
                    n_finitized += 1

        return lb, ub, n_finitized, total_lp_time
    except Exception:
        # Bootstrap is best-effort: never let it break the solve.
        return lb, ub, 0, 0.0


def obbt_tighten_root(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    rounds: int = 3,
    deadline: Optional[float] = None,
    time_limit_per_lp: float = 0.2,
    min_width: float = 1e-6,
    eps: float = 1e-7,
    incumbent_cutoff: Optional[float] = None,
    superposition: bool = False,
    prefer_pounce: bool = False,
    cascade_aux: bool = False,
) -> RootObbtResult:
    """Structural root OBBT over the LP-form McCormick relaxation.

    For each original variable, solves ``min x_i`` and ``max x_i`` subject to
    the McCormick LP relaxation of the model at the current bound box, and
    tightens the variable's bound to the LP optimum.  The LP feasible region is
    a polyhedral OUTER approximation of the nonconvex MINLP feasible set, so any
    tightened bound is rigorously valid — OBBT never removes a feasible point.
    Tightening shrinks the box, which strengthens the McCormick envelopes, so
    the pass is iterated up to ``rounds`` times (stopping early at a fixpoint).

    Unlike :func:`run_obbt` (which only sees the model's *linear* constraints
    and is therefore a no-op for models whose only constraints are nonlinear),
    this builds the full relaxation and so tightens bounds from the bilinear /
    trilinear / monomial / fractional-power envelopes themselves.

    The pass is purely a tightening: the returned box is always a subset of the
    input box (intersected, never loosened), integer bounds are rounded inward,
    and any internal failure returns the input box unchanged (``n_tightened=0``)
    rather than raising — it can never make the solve unsound.

    Parameters
    ----------
    model : Model
        The (already reformulated) MINLP model.
    lb, ub : np.ndarray
        Current flat variable bounds (length = number of scalar variables).
    rounds : int
        Maximum OBBT sweeps; each sweep rebuilds the envelope at the tightened
        box.  Stops early when a sweep tightens nothing.
    deadline : float, optional
        Absolute ``time.perf_counter()`` budget; the loop aborts when reached.
    incumbent_cutoff : float, optional
        If a feasible objective is known, add ``obj <= cutoff`` to the polytope
        (optimality-based tightening) — often a much larger reduction.
    cascade_aux : bool, default False
        Capture OBBT's tightening of the lifted *auxiliary* (product/ratio)
        columns and propagate it back onto the original variables through the
        nonlinear term definitions — reverse FBBT (#208). This recovers the
        hyperbolic/root bounds the linear McCormick rows cannot express, and is
        sound by construction (an OBBT aux bound is valid over the relaxation
        polytope, which outer-approximates the MINLP feasible set, and reverse
        FBBT only removes term-infeasible points). **Opt-in (default off):** on
        the vendored corpus it is sound (every optimum preserved) and measurably
        shrinks the root box on several instances (e.g. nvs11/12/13: box
        log-volume −4 nats), but it did **not** yield a net node-count / wall
        reduction and slightly regressed one instance (nvs13 39→53 nodes), so it
        does not meet #208's "the extra OBBT cost must pay for itself" bar yet. It
        is kept available behind this flag for a future targeted-budget A/B.
    """
    from discopt.modeling.core import VarType

    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    n_orig = len(lb)

    # Per-variable integrality, to round integer bounds inward after each sweep.
    is_int = np.zeros(n_orig, dtype=bool)
    flat_idx = 0
    for v in model._variables:
        flag = v.var_type in (VarType.BINARY, VarType.INTEGER)
        for _ in range(v.size):
            if flat_idx < n_orig:
                is_int[flat_idx] = flag
            flat_idx += 1

    total_tight = 0
    total_lp_time = 0.0
    n_rounds = 0

    # Bootstrap: if the box is not fully finite, derive finite bounds for the
    # open variables before the round loop, which otherwise bails immediately on
    # its finite-box guard (it cannot build a McCormick envelope on an open
    # variable).  Two complementary rigorous passes, interleaved to a fixpoint
    # because each unlocks the other:
    #   * ``bootstrap_finite_bounds`` — LP over the linear subsystem, finitizing
    #     variables reachable from linear constraints;
    #   * ``propagate_equality_defined_bounds`` — forward interval propagation
    #     through equality-defined quantities (``v == g(x)``: divisions, logs,
    #     …) whose inputs the bootstrap just made finite.
    # Both only shrink the box, so the result is always a sound subset.
    for _ in range(2):
        if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
            break
        lb, ub, n_boot, boot_time = bootstrap_finite_bounds(
            model,
            lb,
            ub,
            deadline=deadline,
            time_limit_per_lp=time_limit_per_lp,
            incumbent_cutoff=incumbent_cutoff,
            eps=eps,
        )
        total_tight += n_boot
        total_lp_time += boot_time
        lb, ub, n_prop = propagate_equality_defined_bounds(model, lb, ub)
        total_tight += n_prop
        if n_boot == 0 and n_prop == 0:
            break

    try:
        from discopt._jax.mccormick_lp import (
            MccormickLPRelaxer,
            build_milp_relaxation,
        )

        relaxer = MccormickLPRelaxer(model, superposition=superposition)
        if not relaxer.has_relaxable_nonlinearity:
            return RootObbtResult(lb, ub, 0, 0, 0.0)

        # #208 cascade: carry OBBT-tightened auxiliary-column bounds across rounds,
        # keyed by (stable) column index. The aux columns are a fixed function of
        # the model's nonlinear terms, so their indices are identical across
        # rebuilds even as the original box shrinks. Intersecting a previously
        # captured (valid) aux bound into a freshly built relaxation keeps it a
        # valid outer approximation and lets the tighter aux box cascade onto the
        # original variables via the McCormick rows.
        carried_aux: dict[int, list[float]] = {}

        def _apply_carried_aux(milp) -> None:
            if not (cascade_aux and carried_aux):
                return
            n_total = len(milp._bounds)
            for col, (alb, aub) in carried_aux.items():
                if n_orig <= col < n_total:
                    lo, hi = milp._bounds[col]
                    milp._bounds[col] = (max(float(lo), alb), min(float(hi), aub))

        def _capture_aux(milp, res) -> None:
            if not cascade_aux:
                return
            n_total = len(milp._bounds)
            tl, tu = res.tightened_lb, res.tightened_ub
            if len(tl) < n_total:  # not a full_result run; nothing to capture
                return
            for col in range(n_orig, n_total):
                alo, ahi = float(tl[col]), float(tu[col])
                if not (np.isfinite(alo) and np.isfinite(ahi)):
                    continue
                if col in carried_aux:
                    cur = carried_aux[col]
                    carried_aux[col] = [max(cur[0], alo), min(cur[1], ahi)]
                else:
                    carried_aux[col] = [alo, ahi]

        for _ in range(max(1, rounds)):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            # OBBT requires a finite box to build the envelopes; bail (return
            # whatever tightening prior rounds achieved) if any bound is open.
            if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
                break
            try:
                milp, varmap = build_milp_relaxation(
                    relaxer._model,
                    relaxer._terms,
                    relaxer._disc,
                    bound_override=(lb, ub),
                    superposition=relaxer._superposition,
                )
            except Exception:
                break
            _apply_carried_aux(milp)

            # Duality-based bound tightening first: one objective LP yields
            # reduced costs that tighten every variable at once (cheap), before
            # OBBT's 2n min/max solves. Both tighten against the same relaxation,
            # so their results intersect soundly. DBBT only fires with a cutoff.
            if incumbent_cutoff is not None:
                try:
                    dbbt_res = dbbt_on_relaxation(
                        milp,
                        relaxer._n_orig,
                        incumbent_cutoff,
                        eps=eps,
                        time_limit_per_lp=time_limit_per_lp,
                    )
                    total_lp_time += dbbt_res.total_lp_time
                    if dbbt_res.n_tightened > 0:
                        m = min(n_orig, len(dbbt_res.tightened_lb))
                        new_lb = dbbt_res.tightened_lb[:m]
                        new_ub = dbbt_res.tightened_ub[:m]
                        if np.any(is_int[:m]):
                            new_lb = np.where(is_int[:m], np.ceil(new_lb - eps), new_lb)
                            new_ub = np.where(is_int[:m], np.floor(new_ub + eps), new_ub)
                        lb[:m] = np.maximum(lb[:m], new_lb)
                        ub[:m] = np.minimum(ub[:m], new_ub)
                        total_tight += int(dbbt_res.n_tightened)
                        if np.any(lb[:m] > ub[:m] + 1e-9):
                            return RootObbtResult(
                                lb, ub, total_tight, n_rounds, total_lp_time, True
                            )
                        # Rebuild the envelope at the DBBT-tightened box so OBBT
                        # below sees the strengthened relaxation.
                        try:
                            milp, _ = build_milp_relaxation(
                                relaxer._model,
                                relaxer._terms,
                                relaxer._disc,
                                bound_override=(lb, ub),
                                superposition=relaxer._superposition,
                            )
                            _apply_carried_aux(milp)
                        except Exception:
                            break
                except Exception:
                    pass

            # Include the aux columns as OBBT candidates (and request the full
            # column vector) when cascading, so their tightening is captured and
            # carried instead of discarded.
            n_total = len(milp._bounds)
            obbt_candidates = list(range(n_total)) if cascade_aux else None
            res = run_obbt_on_relaxation(
                milp,
                relaxer._n_orig,
                candidate_idxs=obbt_candidates,
                time_limit_per_lp=time_limit_per_lp,
                incumbent_cutoff=incumbent_cutoff,
                min_width=min_width,
                eps=eps,
                deadline=deadline,
                prefer_pounce=prefer_pounce,
                full_result=cascade_aux,
            )
            total_lp_time += res.total_lp_time
            n_rounds += 1
            _capture_aux(milp, res)

            sweep_tight = 0
            for i in range(min(n_orig, len(res.tightened_lb))):
                new_lo = res.tightened_lb[i]
                new_hi = res.tightened_ub[i]
                if is_int[i]:
                    new_lo = np.ceil(new_lo - eps)
                    new_hi = np.floor(new_hi + eps)
                if new_lo > lb[i] + eps:
                    lb[i] = new_lo
                    sweep_tight += 1
                if new_hi < ub[i] - eps:
                    ub[i] = new_hi
                    sweep_tight += 1
                if lb[i] > ub[i] + 1e-9:
                    # The tightened box is empty -> the (sub)problem is infeasible.
                    return RootObbtResult(
                        lb, ub, total_tight + sweep_tight, n_rounds, total_lp_time, True
                    )

            # #208 cascade: propagate the freshly tightened aux-column bounds back
            # through the nonlinear term definitions (reverse FBBT). This is the
            # step that actually shrinks the *original* box — the hyperbolic/root
            # bounds the linear McCormick rows can't express — turning the captured
            # aux tightening into a real reduction instead of a self-implied no-op.
            if cascade_aux and len(res.tightened_lb) >= len(milp._bounds):
                fb = reverse_fbbt_from_aux(
                    lb,
                    ub,
                    np.asarray(res.tightened_lb, dtype=np.float64),
                    np.asarray(res.tightened_ub, dtype=np.float64),
                    varmap,
                    is_int=is_int,
                    eps=eps,
                )
                sweep_tight += fb
                if np.any(lb[:n_orig] > ub[:n_orig] + 1e-9):
                    return RootObbtResult(
                        lb, ub, total_tight + sweep_tight, n_rounds, total_lp_time, True
                    )

            total_tight += sweep_tight
            if sweep_tight == 0:
                break
    except Exception:
        # Never let bound tightening crash or corrupt the solve.
        return RootObbtResult(lb, ub, total_tight, n_rounds, total_lp_time)

    return RootObbtResult(lb, ub, total_tight, n_rounds, total_lp_time)


# ---------------------------------------------------------------------------
# Reverse FBBT from tightened auxiliary bounds (#208 cascade)
# ---------------------------------------------------------------------------


def _interval_mul(al: float, au: float, bl: float, bu: float) -> tuple[float, float]:
    ps = (al * bl, al * bu, au * bl, au * bu)
    return min(ps), max(ps)


def _interval_div(nl: float, nu: float, dl: float, du: float) -> Optional[tuple[float, float]]:
    """``[nl, nu] / [dl, du]`` when ``0`` is not in the denominator interval."""
    if dl <= 0.0 <= du:
        return None  # denominator straddles zero -> unbounded, no tightening
    return _interval_mul(nl, nu, 1.0 / du, 1.0 / dl)


def reverse_fbbt_from_aux(
    lb: np.ndarray,
    ub: np.ndarray,
    aux_lb: np.ndarray,
    aux_ub: np.ndarray,
    varmap: dict,
    *,
    is_int: Optional[np.ndarray] = None,
    eps: float = 1e-7,
) -> int:
    """Propagate tightened aux-column bounds back onto the original variables.

    The McCormick LP relaxation links a lifted aux ``w`` to its arguments only
    through *linear* envelope rows, so a tightened ``w`` box does not, on its own,
    tighten the arguments inside that same LP (the bound is self-implied — this is
    the "frozen envelope blocks the cascade" gap of #208). Propagating it back
    through the *nonlinear definition* of the term, however, yields **hyperbolic /
    root** bounds the linear rows cannot represent:

    * bilinear ``w = a*b`` with ``w in [wl, wu]`` and ``0`` not in ``[bl, bu]``
      gives ``a in [wl, wu] / [bl, bu]`` (interval division), and symmetrically
      for ``b``;
    * monomial ``w = a**p`` with ``w in [wl, wu]`` gives the ``p``-th-root box
      (sign-aware for even ``p``).

    Every such deduction is a sound FBBT step (it only removes ``a`` values that
    cannot satisfy the term equation for any admissible partner), so the box stays
    a valid enclosure. Mutates ``lb`` / ``ub`` in place over the original columns
    and returns the number of bounds tightened.
    """
    n_orig = len(lb)

    def _tighten(col: int, lo: float, hi: float) -> int:
        if col >= n_orig or not (np.isfinite(lo) and np.isfinite(hi)):
            return 0
        if is_int is not None and is_int[col]:
            lo = np.ceil(lo - eps)
            hi = np.floor(hi + eps)
        c = 0
        if lo > lb[col] + eps:
            lb[col] = min(lo, ub[col])
            c += 1
        if hi < ub[col] - eps:
            ub[col] = max(hi, lb[col])
            c += 1
        return c

    n_tight = 0
    for (i, j), cw in varmap.get("bilinear", {}).items():
        if not (0 <= i < n_orig and 0 <= j < n_orig and 0 <= cw < len(aux_lb)):
            continue
        wl, wu = float(aux_lb[cw]), float(aux_ub[cw])
        if not (np.isfinite(wl) and np.isfinite(wu)):
            continue
        # a = w / b
        d = _interval_div(wl, wu, float(lb[j]), float(ub[j]))
        if d is not None:
            n_tight += _tighten(i, d[0], d[1])
        # b = w / a
        d = _interval_div(wl, wu, float(lb[i]), float(ub[i]))
        if d is not None:
            n_tight += _tighten(j, d[0], d[1])

    for (i, p), cw in varmap.get("monomial", {}).items():
        if not (0 <= i < n_orig and 0 <= cw < len(aux_lb)):
            continue
        wl, wu = float(aux_lb[cw]), float(aux_ub[cw])
        if not (np.isfinite(wl) and np.isfinite(wu)) or p < 2:
            continue
        if p % 2 == 0:
            if wu < 0:
                continue  # a**even < 0 infeasible; leave it to the LP/feasibility
            root = wu ** (1.0 / p)
            inner = (max(wl, 0.0)) ** (1.0 / p)  # |a| >= inner
            al, au = float(lb[i]), float(ub[i])
            if al >= 0:
                n_tight += _tighten(i, inner, root)
            elif au <= 0:
                n_tight += _tighten(i, -root, -inner)
            else:
                n_tight += _tighten(i, -root, root)  # straddles 0: only |a| <= root
        else:
            lo = -((-wl) ** (1.0 / p)) if wl < 0 else wl ** (1.0 / p)
            hi = -((-wu) ** (1.0 / p)) if wu < 0 else wu ** (1.0 / p)
            n_tight += _tighten(i, lo, hi)
    return n_tight


# ---------------------------------------------------------------------------
# OBBT-on-auxiliaries diagnostic (#208 decision gate)
# ---------------------------------------------------------------------------


@dataclass
class AuxTighteningReport:
    """How much OBBT *would* tighten the lifted (aux) columns, currently discarded.

    The McCormick envelope rows are baked over the build-time aux bounds and never
    regenerated, so any OBBT tightening of an aux column (e.g. a ratio aux ``r``)
    is thrown away. This report measures that discarded tightening **without
    touching the solve path** — it is a pure diagnostic for the #208 decision gate
    ("is the envelope-rebuild worth building?"). Soundness is not at stake: nothing
    here feeds back into the relaxation.

    Attributes:
        n_aux: Number of auxiliary (lifted) columns in the relaxation.
        n_aux_tightened: Aux columns whose lb or ub strictly tightened.
        mean_rel_reduction: Mean over *finite-width* aux columns of
            ``(old_width - new_width) / old_width`` in ``[0, 1]``.
        max_rel_reduction: Worst-case relative width reduction.
        aux_rel_reductions: Per-aux relative width reduction (finite-width only).
        n_lp_solves: OBBT LP solves spent on the aux columns.
        lp_time: Wall time of those LP solves.
    """

    n_aux: int
    n_aux_tightened: int
    mean_rel_reduction: float
    max_rel_reduction: float
    aux_rel_reductions: list[float]
    n_lp_solves: int
    lp_time: float


def measure_discarded_aux_tightening(
    model: Model,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    time_limit_per_lp: float = 0.2,
    incumbent_cutoff: Optional[float] = None,
    deadline: Optional[float] = None,
    superposition: bool = False,
) -> Optional[AuxTighteningReport]:
    """Measure the OBBT tightening of the lifted aux columns that #208 discards.

    Builds the McCormick relaxation at the box ``[lb, ub]``, runs OBBT over the
    **auxiliary** columns (indices ``n_orig .. n_total``), and reports how much
    their bounds shrink. This is the cheap, no-risk decision-gate step the issue
    prescribes before investing in the envelope-rebuild (part 2): if many aux
    columns shrink a lot, the discarded tightening is worth cascading back.

    Returns ``None`` when the model has no relaxable nonlinearity / no aux columns
    / the box is not finite (nothing to measure), else an
    :class:`AuxTighteningReport`. Never raises into the caller (pure diagnostic).
    """
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        return None
    try:
        from discopt._jax.mccormick_lp import MccormickLPRelaxer, build_milp_relaxation

        relaxer = MccormickLPRelaxer(model, superposition=superposition)
        if not relaxer.has_relaxable_nonlinearity:
            return None
        milp, _ = build_milp_relaxation(
            relaxer._model,
            relaxer._terms,
            relaxer._disc,
            bound_override=(lb, ub),
            superposition=relaxer._superposition,
        )
        n_orig = relaxer._n_orig
        n_total = len(milp._bounds)
        if n_total <= n_orig:
            return None  # no auxiliary columns

        old = np.asarray(milp._bounds, dtype=np.float64)
        old_w = old[:, 1] - old[:, 0]
        aux_idxs = list(range(n_orig, n_total))

        res = run_obbt_on_relaxation(
            milp,
            n_orig,
            candidate_idxs=aux_idxs,
            time_limit_per_lp=time_limit_per_lp,
            incumbent_cutoff=incumbent_cutoff,
            deadline=deadline,
            full_result=True,
        )
        new_lb = np.asarray(res.tightened_lb, dtype=np.float64)
        new_ub = np.asarray(res.tightened_ub, dtype=np.float64)
        new_w = new_ub - new_lb

        rels: list[float] = []
        n_tight = 0
        for j in aux_idxs:
            ow = old_w[j]
            if not np.isfinite(ow) or ow <= 1e-12:
                continue
            r = float((ow - new_w[j]) / ow)
            r = max(0.0, min(1.0, r))  # clamp round-off
            rels.append(r)
            if r > 1e-6:
                n_tight += 1
        return AuxTighteningReport(
            n_aux=len(aux_idxs),
            n_aux_tightened=n_tight,
            mean_rel_reduction=float(np.mean(rels)) if rels else 0.0,
            max_rel_reduction=float(np.max(rels)) if rels else 0.0,
            aux_rel_reductions=rels,
            n_lp_solves=res.n_lp_solves,
            lp_time=res.total_lp_time,
        )
    except Exception:
        return None
