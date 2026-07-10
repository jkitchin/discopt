"""POUNCE QP solver: solve a convex quadratic program through the pure-Rust IPM.

This is the QP backend: discopt's QP path is HiGHS-free (issue #359), so
``solve_qp`` here is the sole matrix-form continuous-QP engine. It follows the
shared QP contract (signature + ``QPResult`` with HiGHS-convention duals).
Differences, all inherent to an interior-point backend:

- **No integrality**: POUNCE has no branch & bound, so ``integrality`` with
  any integer entries raises ``ValueError`` (the model-level seam keeps MIQPs
  on the self-hosted B&B path).
- **Convex Q assumed**: an IPM converges to a KKT point; for indefinite ``Q``
  that is only a local solution. The model-level dispatch already gates this
  path on detected convexity.
- On a degenerate optimal face the primal/dual is the analytic center, not a
  vertex (objective matches); there is no basis.

The objective is ``0.5 * x^T Q x + c^T x`` (the shared QP convention): the
callbacks expose gradient ``Q x + c``, the constant linear Jacobian, and the
constant Hessian ``obj_factor * Q`` (lower triangle).

Status mapping, dual sign reconciliation (Ipopt ``mult_g`` is the negation of
the HiGHS shadow-price convention), and the elastic Phase-1 infeasibility
certificate (roadmap P0.2) are shared with :mod:`discopt.solvers.lp_pounce` —
the Phase-1 probe only involves the linear constraints and the box, so it is
objective-independent and applies to QPs unchanged.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import QPResult, SolveStatus
from discopt.solvers.lp_pounce import (
    _FINITE_BOUND_THRESHOLD,
    _INF,
    _LP_STATUS_MAP,
    POUNCE_AVAILABLE,
    PounceKKTError,
    _build_certificate,
    _interior_start,
    _is_infeasible_violation,
    _phase1_min_violation,
    _stack_constraints,
)


class _QPCallbacks:
    """cyipopt/POUNCE callbacks for ``min 0.5 x^T Q x + c^T x`` + linear rows.

    The Jacobian is constant; the Hessian of the Lagrangian is the constant
    ``obj_factor * Q`` (constraints are linear, so they contribute nothing).
    """

    def __init__(self, Q: np.ndarray, c: np.ndarray, A: np.ndarray) -> None:
        self._Q = Q
        self._c = c
        self._A = A
        self._m, self._n = A.shape
        self._jac_flat = A.ravel().astype(np.float64)
        _rows, _cols = np.meshgrid(np.arange(self._m), np.arange(self._n), indexing="ij")
        self._jac_rows = _rows.ravel()
        self._jac_cols = _cols.ravel()
        self._hess_rows, self._hess_cols = np.tril_indices(self._n)
        self._hess_vals = Q[self._hess_rows, self._hess_cols].astype(np.float64)

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self._Q @ x + self._c @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._Q @ x + self._c, dtype=np.float64)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._A @ x, dtype=np.float64)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._jac_flat

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._jac_rows, self._jac_cols

    def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
        return obj_factor * self._hess_vals

    def hessianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._hess_rows, self._hess_cols


def solve_qp(
    Q: np.ndarray,
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    x0: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
    certificate: bool = False,
) -> QPResult:
    """Solve ``min 0.5 x^T Q x + c^T x`` s.t. linear constraints via POUNCE.

    Follows the shared QP contract for the pure-continuous case; ``bounds``
    default to ``(-inf, +inf)`` per variable.

    Raises:
        ImportError: If POUNCE is not installed.
        ValueError: On inconsistent dimensions, or any integer-marked variable (no MIQP support).
    """
    if not POUNCE_AVAILABLE:
        raise ImportError(
            "pounce is required for this backend. Install it with:\n  pip install pounce-solver"
        )
    if integrality is not None and np.any(np.asarray(integrality) == 1):
        raise ValueError(
            "qp_pounce.solve_qp is a continuous QP solver; integrality is not "
            "supported (MIQPs go through HiGHS or the B&B path)."
        )

    Q_arr = np.asarray(Q, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    if Q_arr.shape != (n, n):
        raise ValueError(f"Q has shape {Q_arr.shape} but c has {n} elements")

    # ---- variable bounds (shared QP contract default: free variables) -------
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")
        lb = np.array([b[0] for b in bounds], dtype=np.float64)
        ub = np.array([b[1] for b in bounds], dtype=np.float64)
    else:
        lb = np.full(n, -_INF, dtype=np.float64)
        ub = np.full(n, _INF, dtype=np.float64)
    lb = np.where(lb <= -_FINITE_BOUND_THRESHOLD, -_INF, lb)
    ub = np.where(ub >= _FINITE_BOUND_THRESHOLD, _INF, ub)

    # ---- stacked linear constraints ------------------------------------------
    A, cl, cu = _stack_constraints(A_ub, b_ub, A_eq, b_eq, n)
    m = A.shape[0]
    n_ineq = A_ub.shape[0] if (A_ub is not None and b_ub is not None) else 0

    if x0 is None:
        x0 = _interior_start(lb, ub)
    x0 = np.asarray(x0, dtype=np.float64).ravel()

    opts: dict[str, Any] = {"print_level": 0}
    if options:
        opts.update(options)
    if time_limit is not None:
        opts.setdefault("max_wall_time", float(time_limit))

    result = _solve_qp_core(Q_arr, c_arr, A, cl, cu, lb, ub, x0, opts)

    # ---- infeasibility certificate (roadmap P0.2; same logic as lp_pounce) ---
    # Constraints are linear, so the elastic Phase-1 LP is an exact Farkas
    # disambiguation regardless of the quadratic objective. UNBOUNDED is included
    # because Ipopt codes 3/4 (too-small direction / diverging iterates) cannot
    # distinguish an unbounded QP from an infeasible one — Phase-1 settles it.
    if m > 0 and result.status in (
        SolveStatus.ITERATION_LIMIT,
        SolveStatus.ERROR,
        SolveStatus.UNBOUNDED,
    ):
        slacks = _phase1_min_violation(A, cl, cu, lb, ub, opts)
        if slacks is not None and _is_infeasible_violation(slacks, cl, cu):
            return QPResult(
                status=SolveStatus.INFEASIBLE,
                iterations=result.iterations,
                wall_time=result.wall_time,
                infeasibility_certificate=_build_certificate(slacks, n_ineq),
            )
    elif certificate and result.status == SolveStatus.INFEASIBLE and m > 0:
        slacks = _phase1_min_violation(A, cl, cu, lb, ub, opts)
        if slacks is not None and _is_infeasible_violation(slacks, cl, cu):
            result.infeasibility_certificate = _build_certificate(slacks, n_ineq)

    return result


def solve_qp_kkt(
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x_l: np.ndarray,
    x_u: np.ndarray,
    options: Optional[dict] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``min 0.5 xᵀQx + cᵀx  s.t.  A x = b,  x_l ≤ x ≤ x_u`` and return the
    interior-point KKT point ``(obj, x, y, z_l, z_u)``.

    Same sign convention as :func:`discopt.solvers.lp_pounce.solve_lp_kkt`:
    stationarity ``Qx + c − Aᵀy − z_l + z_u = 0`` with ``z_l, z_u ≥ 0``, so
    ``y = −mult_g``, ``z_l = mult_x_L``, ``z_u = mult_x_U``. All-equality form
    only (the differentiable QP layer feeds ``A_eq``/``b_eq``).
    """
    if not POUNCE_AVAILABLE:
        raise ImportError(
            "pounce is required for this backend. Install it with:\n  pip install pounce-solver"
        )
    import pounce

    Q_arr = np.asarray(Q, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.size
    A_arr = (
        np.asarray(A, dtype=np.float64).reshape(-1, n)
        if A is not None
        else np.empty((0, n), dtype=np.float64)
    )
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    m = A_arr.shape[0]

    lb = np.asarray(x_l, dtype=np.float64).ravel().copy()
    ub = np.asarray(x_u, dtype=np.float64).ravel().copy()
    lb = np.where(lb <= -_FINITE_BOUND_THRESHOLD, -_INF, lb)
    ub = np.where(ub >= _FINITE_BOUND_THRESHOLD, _INF, ub)

    cl = b_arr.copy()
    cu = b_arr.copy()
    x0 = _interior_start(lb, ub)

    opts: dict[str, Any] = {"print_level": 0}
    if options:
        opts.update(options)

    problem = pounce.Problem(
        n=n, m=m, problem_obj=_QPCallbacks(Q_arr, c_arr, A_arr), lb=lb, ub=ub, cl=cl, cu=cu
    )
    for key, value in opts.items():
        try:
            if isinstance(value, (np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except (TypeError, ValueError, RuntimeError):
            pass

    x, info = problem.solve(x0)
    # The differentiable QP layer linearizes the KKT system here, so a
    # non-converged solve would yield silently wrong gradients. Fail loudly.
    status_code = info.get("status", -100)
    if status_code not in (0, 1):
        raise PounceKKTError(
            f"solve_qp_kkt did not converge (Ipopt status {status_code}); "
            "the KKT point is non-stationary and would give invalid gradients."
        )
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    mult_g = np.asarray(info.get("mult_g", np.zeros(m)), dtype=np.float64).ravel()
    z_l = np.asarray(info.get("mult_x_L", np.zeros(n)), dtype=np.float64).ravel()
    z_u = np.asarray(info.get("mult_x_U", np.zeros(n)), dtype=np.float64).ravel()
    if mult_g.size != m:
        mult_g = np.zeros(m)
    if z_l.size != n:
        z_l = np.zeros(n)
    if z_u.size != n:
        z_u = np.zeros(n)
    y = -mult_g
    obj = float(info.get("obj_val", 0.5 * x_arr @ Q_arr @ x_arr + c_arr @ x_arr))
    return obj, x_arr, y, z_l, z_u


def _solve_qp_core(
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
    opts: dict,
) -> QPResult:
    """Single POUNCE entry point for the QP; mirrors lp_pounce._solve_core."""
    import pounce

    n = len(c)
    m = A.shape[0]
    problem = pounce.Problem(
        n=n, m=m, problem_obj=_QPCallbacks(Q, c, A), lb=lb, ub=ub, cl=cl, cu=cu
    )
    for key, value in opts.items():
        try:
            if isinstance(value, (np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except (TypeError, ValueError, RuntimeError):
            pass

    t0 = time.perf_counter()
    x, info = problem.solve(x0)
    wall_time = time.perf_counter() - t0

    status = _LP_STATUS_MAP.get(info.get("status", -100), SolveStatus.ERROR)
    iters = int(info.get("iter_count", 0))

    if status != SolveStatus.OPTIMAL:
        return QPResult(status=status, iterations=iters, wall_time=wall_time)

    x_arr = np.asarray(x, dtype=np.float64)
    obj = float(info.get("obj_val", 0.5 * x_arr @ Q @ x_arr + c @ x_arr))
    dual = info.get("mult_g", None)
    # Same sign reconciliation as lp_pounce: Ipopt multipliers are the
    # negation of the HiGHS shadow-price convention QPResult documents.
    dual_values = -np.asarray(dual, dtype=np.float64) if dual is not None and len(dual) else None
    mult_l = np.asarray(info.get("mult_x_L", []), dtype=np.float64)
    mult_u = np.asarray(info.get("mult_x_U", []), dtype=np.float64)
    reduced_costs = (mult_l - mult_u) if mult_l.size and mult_u.size else None
    # Final KKT residual, when POUNCE reports one. Lets a POUNCE-first QP default
    # detect an unconverged "optimal" (issue #145) and degrade to HiGHS rather
    # than return a drifted objective. ``final_unscaled_kkt_error`` is preferred
    # (the M1-fixed unscaled residual, pounce#174) when present; ``final_kkt_error``
    # (scaled) is the fallback available in released 0.6.0.
    _kkt = info.get("final_unscaled_kkt_error", info.get("final_kkt_error", None))
    kkt_error = float(_kkt) if _kkt is not None else None

    return QPResult(
        status=status,
        x=x_arr,
        objective=obj,
        dual_values=dual_values,
        reduced_costs=reduced_costs,
        node_count=0,
        iterations=iters,
        wall_time=wall_time,
        kkt_error=kkt_error,
    )
