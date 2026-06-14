"""POUNCE LP solver: solve a linear program through the pure-Rust IPM.

Mirrors :func:`discopt.solvers.lp_highs.solve_lp` in signature and return
type so the two are drop-in interchangeable at call sites (OBBT, McCormick-LP,
OA/GDP masters). POUNCE is an interior-point method, so for a degenerate or
dual-degenerate LP it returns a point on the analytic center of the optimal
face rather than a simplex vertex: the *objective* matches the simplex optimum,
but the primal/dual point may differ. There is no simplex basis, so
``LPResult.basis`` is always ``None`` and warm-starting is not supported (an
IPM does not warm-start from a basis; see the POUNCE-only roadmap, P0.1).

An LP has a zero objective Hessian and a constant Jacobian; the callbacks
below expose exactly that to POUNCE.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import scipy.sparse as sp

from discopt.solvers import InfeasibilityCertificate, LPResult, SolveStatus

try:
    import pounce as _pounce  # noqa: F401

    POUNCE_AVAILABLE = True
except ImportError:
    POUNCE_AVAILABLE = False

# POUNCE/Ipopt treats |bound| beyond this as infinite. discopt's modeling layer
# also emits ~1e20 for unbounded variables, so anything past the "very large"
# threshold is mapped to a single sentinel infinity here.
_INF = 1e20
_FINITE_BOUND_THRESHOLD = 1e15
# Above this total constraint violation, the elastic Phase-1 LP certifies the
# original LP infeasible (roadmap P0.2).
_FEAS_TOL = 1e-6


class PounceKKTError(RuntimeError):
    """A KKT-point solve (``solve_lp_kkt`` / ``solve_qp_kkt``) failed to
    converge, so the returned point is not stationary. Raised instead of
    returning silently-wrong sensitivities to a differentiable layer."""

# Ipopt return codes (POUNCE is shape-compatible). For a *convex* LP, local
# infeasibility is global, so code 2 is a sound INFEASIBLE; diverging iterates
# (4) and a too-small search direction (3) on an LP signal unboundedness.
_LP_STATUS_MAP = {
    0: SolveStatus.OPTIMAL,  # Solve_Succeeded
    1: SolveStatus.OPTIMAL,  # Solved_To_Acceptable_Level
    2: SolveStatus.INFEASIBLE,  # Infeasible_Problem_Detected (global for LP)
    3: SolveStatus.UNBOUNDED,  # Search_Direction_Becomes_Too_Small
    4: SolveStatus.UNBOUNDED,  # Diverging_Iterates
    -1: SolveStatus.ITERATION_LIMIT,  # Maximum_Iterations_Exceeded
    -4: SolveStatus.TIME_LIMIT,  # Maximum_CpuTime_Exceeded
    -5: SolveStatus.TIME_LIMIT,  # Maximum_WallTime_Exceeded
}


class _LPCallbacks:
    """cyipopt/POUNCE callback object for ``min c^T x`` with linear rows.

    ``A`` is the dense stacked constraint matrix (inequalities then
    equalities); its values and structure are constant in ``x``. The Hessian
    of the Lagrangian is identically zero.
    """

    def __init__(self, c: np.ndarray, A: np.ndarray) -> None:
        self._c = c
        self._A = A
        self._m, self._n = A.shape
        # Constant dense Jacobian, flattened row-major to match jacobianstructure.
        self._jac_flat = A.ravel().astype(np.float64)
        _rows, _cols = np.meshgrid(np.arange(self._m), np.arange(self._n), indexing="ij")
        self._jac_rows = _rows.ravel()
        self._jac_cols = _cols.ravel()

    def objective(self, x: np.ndarray) -> float:
        return float(self._c @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._c

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._A @ x, dtype=np.float64)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self._jac_flat

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._jac_rows, self._jac_cols

    def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
        # LP: Hessian of the Lagrangian is zero — no structural entries.
        return np.empty(0, dtype=np.float64)

    def hessianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)


def _to_dense(A: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    if sp.issparse(A):
        return np.asarray(cast(sp.spmatrix, A).todense(), dtype=np.float64)
    return np.asarray(A, dtype=np.float64)


def _stack_constraints(
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack inequality and equality rows into (A, cl, cu).

    Inequalities ``A_ub x <= b_ub`` become rows with ``cl=-inf, cu=b_ub``;
    equalities ``A_eq x == b_eq`` become rows with ``cl=cu=b_eq``.
    """
    parts_A: list[np.ndarray] = []
    parts_cl: list[np.ndarray] = []
    parts_cu: list[np.ndarray] = []

    if A_ub is not None and b_ub is not None:
        b = np.asarray(b_ub, dtype=np.float64).ravel()
        parts_A.append(_to_dense(A_ub))
        parts_cl.append(np.full(len(b), -_INF))
        parts_cu.append(b)
    if A_eq is not None and b_eq is not None:
        b = np.asarray(b_eq, dtype=np.float64).ravel()
        parts_A.append(_to_dense(A_eq))
        parts_cl.append(b.copy())
        parts_cu.append(b.copy())

    if not parts_A:
        return (
            np.empty((0, n), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return np.vstack(parts_A), np.concatenate(parts_cl), np.concatenate(parts_cu)


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    warm_basis: Optional[object] = None,
    time_limit: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
    certificate: bool = False,
) -> LPResult:
    """Solve ``min c^T x`` s.t. linear constraints and bounds via POUNCE.

    Same semantics as :func:`discopt.solvers.lp_highs.solve_lp`. ``bounds``
    default to ``(0, +inf)`` per variable when ``None``.

    ``warm_basis`` is accepted for signature compatibility but ignored: an IPM
    does not warm-start from a simplex basis. ``LPResult.basis`` is ``None``.

    When the result is ``INFEASIBLE``, an
    :class:`~discopt.solvers.InfeasibilityCertificate` is attached if one was
    computed: always for infeasibility found via the Phase-1 disambiguation
    path (free — Phase-1 already ran), and on demand for a directly
    POUNCE-detected infeasibility when ``certificate=True`` (one extra Phase-1
    solve).

    Raises:
        ImportError: If POUNCE is not installed.
        ValueError: If matrix dimensions are inconsistent.
    """
    if not POUNCE_AVAILABLE:
        raise ImportError(
            "pounce is required for this backend. Install it with:\n  pip install pounce-solver"
        )
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)

    # ---- validate dimensions (parity with lp_highs) -------------------------
    if A_ub is not None:
        shape = A_ub.shape if sp.issparse(A_ub) else np.asarray(A_ub).shape
        if len(shape) != 2 or shape[1] != n:
            raise ValueError(f"A_ub has {shape[1]} columns but c has {n} elements")
        if b_ub is None:
            raise ValueError("b_ub is required when A_ub is provided")
        if np.asarray(b_ub).ravel().shape[0] != shape[0]:
            raise ValueError(
                f"A_ub has {shape[0]} rows but b_ub has "
                f"{np.asarray(b_ub).ravel().shape[0]} elements"
            )
    if A_eq is not None:
        shape = A_eq.shape if sp.issparse(A_eq) else np.asarray(A_eq).shape
        if len(shape) != 2 or shape[1] != n:
            raise ValueError(f"A_eq has {shape[1]} columns but c has {n} elements")
        if b_eq is None:
            raise ValueError("b_eq is required when A_eq is provided")
        if np.asarray(b_eq).ravel().shape[0] != shape[0]:
            raise ValueError(
                f"A_eq has {shape[0]} rows but b_eq has "
                f"{np.asarray(b_eq).ravel().shape[0]} elements"
            )
    if bounds is not None and len(bounds) != n:
        raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")

    # ---- variable bounds ----------------------------------------------------
    if bounds is not None:
        lb = np.array([b[0] for b in bounds], dtype=np.float64)
        ub = np.array([b[1] for b in bounds], dtype=np.float64)
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, _INF, dtype=np.float64)
    lb = np.where(lb <= -_FINITE_BOUND_THRESHOLD, -_INF, lb)
    ub = np.where(ub >= _FINITE_BOUND_THRESHOLD, _INF, ub)

    # ---- stacked linear constraints -----------------------------------------
    A, cl, cu = _stack_constraints(A_ub, b_ub, A_eq, b_eq, n)
    m = A.shape[0]
    # Row split for mapping a certificate back to the caller's matrices: the
    # stack is inequality rows then equality rows (see _stack_constraints).
    n_ineq = A_ub.shape[0] if (A_ub is not None and b_ub is not None) else 0

    # ---- starting point: strictly interior where bounds are finite ----------
    if x0 is None:
        x0 = _interior_start(lb, ub)
    x0 = np.asarray(x0, dtype=np.float64).ravel()

    opts: dict[str, Any] = {"print_level": 0}
    if options:
        opts.update(options)
    if time_limit is not None:
        opts.setdefault("max_wall_time", float(time_limit))

    result = _solve_core(c_arr, A, cl, cu, lb, ub, x0, opts)

    # ---- infeasibility certificate (roadmap P0.2) ---------------------------
    # An IPM does not always certify infeasibility: an inconsistent system can
    # exit at the iteration limit, as a generic error, or — because diverging
    # iterates / a too-small search direction (Ipopt codes 4/3) look the same on
    # an infeasible LP as on an unbounded one — as a spurious UNBOUNDED.
    # Disambiguate with an elastic Phase-1 LP that minimizes total constraint
    # violation. For an LP this is exact (by LP duality a positive minimal
    # violation is a Farkas certificate): >0 proves infeasibility; ~0 proves the
    # original was feasible, so the prior status (numerical failure, or a genuine
    # UNBOUNDED once feasibility is established) is reported honestly.
    if m > 0 and result.status in (
        SolveStatus.ITERATION_LIMIT,
        SolveStatus.ERROR,
        SolveStatus.UNBOUNDED,
    ):
        slacks = _phase1_min_violation(A, cl, cu, lb, ub, opts)
        if slacks is not None and float(slacks.sum()) > _FEAS_TOL:
            return LPResult(
                status=SolveStatus.INFEASIBLE,
                iterations=result.iterations,
                wall_time=result.wall_time,
                infeasibility_certificate=_build_certificate(slacks, n_ineq),
            )
    elif certificate and result.status == SolveStatus.INFEASIBLE and m > 0:
        # POUNCE detected infeasibility directly; spend one Phase-1 solve to
        # build the requested witness.
        slacks = _phase1_min_violation(A, cl, cu, lb, ub, opts)
        if slacks is not None and float(slacks.sum()) > _FEAS_TOL:
            result.infeasibility_certificate = _build_certificate(slacks, n_ineq)

    return result


def solve_lp_kkt(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x_l: np.ndarray,
    x_u: np.ndarray,
    options: Optional[dict] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``min cᵀx  s.t.  A x = b,  x_l ≤ x ≤ x_u`` and return the full
    interior-point KKT point ``(obj, x, y, z_l, z_u)``.

    The signs follow the differentiable-layer stationarity convention
    ``c − Aᵀy − z_l + z_u = 0`` with ``z_l, z_u ≥ 0``; from Ipopt's stationarity
    ``∇f + Aᵀ·mult_g − mult_x_L + mult_x_U = 0`` that means ``y = −mult_g``,
    ``z_l = mult_x_L``, ``z_u = mult_x_U``. Because POUNCE is an IPM it returns
    the analytic center of the optimal face (strictly positive complementarity
    slacks), so the KKT sensitivity system used by ``differentiable_lp`` stays
    nonsingular — unlike a degenerate simplex vertex. All-equality form only
    (the differentiable LP layer feeds ``A_eq``/``b_eq``).
    """
    if not POUNCE_AVAILABLE:
        raise ImportError(
            "pounce is required for this backend. Install it with:\n  pip install pounce-solver"
        )
    import pounce

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.size
    A_arr = _to_dense(A).reshape(-1, n) if A is not None else np.empty((0, n), dtype=np.float64)
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
        n=n, m=m, problem_obj=_LPCallbacks(c_arr, A_arr), lb=lb, ub=ub, cl=cl, cu=cu
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
    # The differentiable LP layer linearizes the KKT system at this point, so a
    # non-converged solve (anything but Solve_Succeeded / Solved_To_Acceptable)
    # would yield silently wrong gradients. Fail loudly instead.
    status_code = info.get("status", -100)
    if status_code not in (0, 1):
        raise PounceKKTError(
            f"solve_lp_kkt did not converge (Ipopt status {status_code}); "
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
    obj = float(info.get("obj_val", c_arr @ x_arr))
    return obj, x_arr, y, z_l, z_u


def _build_certificate(slacks: np.ndarray, n_ineq: int) -> InfeasibilityCertificate:
    """Split the Phase-1 per-row slacks into an inequality/equality witness."""
    return InfeasibilityCertificate(
        total_violation=float(slacks.sum()),
        ineq_violations=np.asarray(slacks[:n_ineq], dtype=np.float64),
        eq_violations=np.asarray(slacks[n_ineq:], dtype=np.float64),
    )


def _solve_core(
    c: np.ndarray,
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
    opts: dict,
) -> LPResult:
    """Solve ``min c^T x`` over stacked rows ``cl <= A x <= cu`` and bounds.

    The single POUNCE entry point shared by :func:`solve_lp` and the Phase-1
    feasibility probe. Status is mapped via :data:`_LP_STATUS_MAP`.
    """
    import pounce

    n = len(c)
    m = A.shape[0]
    problem = pounce.Problem(n=n, m=m, problem_obj=_LPCallbacks(c, A), lb=lb, ub=ub, cl=cl, cu=cu)
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
        return LPResult(status=status, iterations=iters, wall_time=wall_time)

    x_arr = np.asarray(x, dtype=np.float64)
    obj = float(info.get("obj_val", c @ x_arr))
    dual = info.get("mult_g", None)
    # Ipopt's multipliers enter the Lagrangian as f + mult_g^T g, so they are
    # the *negation* of the shadow-price convention HiGHS reports (y = dz/db)
    # and that LPResult documents. Negate so both backends agree; the reduced
    # costs (mult_x_L - mult_x_U == c - A^T y) already match. On a
    # dual-degenerate LP the IPM returns an interior point of the dual optimal
    # face rather than a vertex — a valid dual solution, just not simplex's.
    dual_values = -np.asarray(dual, dtype=np.float64) if dual is not None and len(dual) else None
    mult_l = np.asarray(info.get("mult_x_L", []), dtype=np.float64)
    mult_u = np.asarray(info.get("mult_x_U", []), dtype=np.float64)
    reduced_costs = (mult_l - mult_u) if mult_l.size and mult_u.size else None

    return LPResult(
        status=status,
        x=x_arr,
        objective=obj,
        dual_values=dual_values,
        reduced_costs=reduced_costs,
        basis=None,
        iterations=iters,
        wall_time=wall_time,
    )


def _interior_start(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """A starting point in the box interior where bounds are finite."""
    lo = np.where(np.isfinite(lb) & (lb > -_INF), lb, -1.0)
    hi = np.where(np.isfinite(ub) & (ub < _INF), ub, 1.0)
    return np.clip(0.5 * (lo + hi), -1e3, 1e3)


def _phase1_min_violation(
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: dict,
) -> Optional[np.ndarray]:
    """Per-row minimal constraint violation of ``cl <= A x <= cu`` over the box.

    Builds and solves the elastic LP

        min  1^T s
        s.t. A x - s <= cu,   A x + s >= cl,   lb <= x <= ub,   s >= 0

    in the variables ``[x, s]`` (one slack per row). The elastic LP is always
    feasible and bounded below by 0; at the optimum each ``s_i`` is the minimal
    violation row ``i`` must incur. Returns that length-``m`` slack vector
    (its sum is the total minimal violation), or ``None`` if even the
    (well-posed) Phase-1 solve did not reach optimality.
    """
    m, n = A.shape
    eye = np.eye(m, dtype=np.float64)
    # [A | -I] bounded above by cu;  [A | +I] bounded below by cl.
    A2 = np.vstack([np.hstack([A, -eye]), np.hstack([A, eye])])
    cl2 = np.concatenate([np.full(m, -_INF), cl])
    cu2 = np.concatenate([cu, np.full(m, _INF)])
    c2 = np.concatenate([np.zeros(n), np.ones(m)])
    lb2 = np.concatenate([lb, np.zeros(m)])
    ub2 = np.concatenate([ub, np.full(m, _INF)])
    x0 = np.concatenate([_interior_start(lb, ub), np.ones(m)])

    res = _solve_core(c2, A2, cl2, cu2, lb2, ub2, x0, opts)
    if res.status == SolveStatus.OPTIMAL and res.x is not None:
        # The slacks are the trailing m entries; clip tiny negatives from the
        # interior-point tolerance.
        return np.clip(np.asarray(res.x[n:], dtype=np.float64), 0.0, None)
    return None
