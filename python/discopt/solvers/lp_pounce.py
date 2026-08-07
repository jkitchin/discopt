"""POUNCE LP solver: solve a linear program through the pure-Rust IPM.

Mirrors :func:`discopt.solvers.lp_simplex.solve_lp` in signature and return
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

import logging
import time
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import scipy.sparse as sp

from discopt.solvers import (
    InfeasibilityCertificate,
    LPResult,
    SolveStatus,
    pounce_option_defaults,
)

logger = logging.getLogger(__name__)

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
# Tiny floating-point bound inversions (lb just above ub, e.g. ~1e-11 out of
# relaxation/bound-tightening) are snapped to a single fixed value before they
# reach POUNCE. POUNCE's IPM strictly validates bounds and rejects ``lb > ub``
# as Invalid_Problem_Definition; presolve-based solvers (HiGHS) silently snap
# them. Inversions beyond this tolerance are left intact so they surface as
# genuine infeasibility rather than being masked.
_BOUND_SNAP_TOL = 1e-7
# The POUNCE option baseline (constr_viol_tol, bound_relax_factor) lives in
# :func:`discopt.solvers.pounce_option_defaults` — one source of truth shared
# by every entry point that hands options to POUNCE. See the measurements and
# the per-build analysis there (issue #940).


# A recession direction must lower the objective by at least this much, relative
# to the cost vector's magnitude, before ``UNBOUNDED`` is certified. The ray LP's
# directions are normalized to the unit box, so a genuine ray registers as an
# O(|c|) drop while interior-point noise registers near zero. Erring toward NOT
# certifying is the safe direction: the caller then degrades to the exact simplex.
_RAY_COST_TOL = 1e-7


def _certify_unbounded_ray(
    c: np.ndarray,
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: dict,
    Q: Optional[np.ndarray] = None,
) -> bool:
    """Whether a genuine improving recession direction exists — an *earned*
    ``UNBOUNDED``, rather than one inferred from a numerical-failure code.

    POUNCE reaches ``UNBOUNDED`` through :data:`_LP_STATUS_MAP` from Ipopt codes
    3 and 4 (``Search_Direction_Becomes_Too_Small`` / ``Diverging_Iterates``).
    Those codes are *ambiguous*: they report that the interior-point iteration
    stalled or blew up, not that the problem has a ray. POUNCE never claims
    unboundedness — that inference was discopt's, and it is unsound. Measured
    (issue #940): on random LPs of data magnitude ~1e7 over an ordinary
    ``[0, 1e8]`` box, ~4 in 10 instances whose true status is ``optimal`` exit
    this way, and 42 of 90 do so on a box carrying one infinite bound, where the
    objective is ``min c'x`` with ``c >= 0, x >= 0`` and therefore cannot decrease
    along any ray by construction. Those bounds sit far below the
    ``[1e15, 1e20)`` window of the #850 Obs 1 deferral, so
    ``solver._solve_lp_matrix`` certified the verdict: ``model.solve()`` returned
    ``status='unbounded'`` for an LP whose exact optimum is 5.1e7.

    Rather than drop the verdict (which would cost the Benders dual seam its
    feasibility-cut signal, where an unbounded dual is the normal case), settle it
    with the recession cone: a **feasible** LP is unbounded **iff** its recession
    cone contains a direction of strictly negative cost. That cone is described by
    the same data:

        min  c'd   s.t.   (A d)_i <= 0 where cu_i is finite,
                          (A d)_i >= 0 where cl_i is finite,
                          d_j >= 0 where lb_j is finite,
                          d_j <= 0 where ub_j is finite,
                          -1 <= d <= 1                        (normalization)

    which is itself always feasible (``d = 0``) and bounded, so it is a far easier
    solve than the one that just failed. A strictly negative optimum exhibits the
    ray and ``x + t·d`` stays feasible for every ``t >= 0``, certifying
    unboundedness; anything else means the exit was numerical and the caller
    should degrade to the exact simplex.

    On the feasibility hypothesis of that "iff": ``solve_lp``'s elastic Phase-1
    establishes it on the paths where Phase-1 actually runs — ``m > 0`` and the
    Phase-1 solve itself returning slacks. It is *not* established when there are
    no constraint rows at all, or when that solve fails and returns ``None``
    (``qp_pounce.solve_qp`` has the identical structure). That gap costs nothing,
    because this test is one-directional: it can only ever WITHDRAW an ``UNBOUNDED``
    verdict, never add one. An infeasible LP that happens to own a recession ray
    keeps whatever verdict it already had, so an unestablished premise costs a
    certificate, never soundness.

    ``Q`` extends this to a convex QP: along ``d`` the objective is
    ``½t²·d'Qd + t·(c'd + x'Qd)``, so it can only fall without bound when
    ``d'Qd = 0``, which for positive-semidefinite ``Q`` is equivalent to
    ``Qd = 0`` — added here as equality rows.

    Returns ``False`` whenever the ray LP does not itself reach a clean optimum,
    so an undecidable case never becomes a certificate (CLAUDE.md §1).
    """
    n = len(c)
    finite_lo = lb > -_INF
    finite_hi = ub < _INF
    # Directions live in the unit box, tightened to a half-line (or to {0}) by
    # whichever variable bounds are finite.
    d_lo = np.where(finite_lo, 0.0, -1.0)
    d_hi = np.where(finite_hi, 0.0, 1.0)
    if not np.any(d_lo < 0.0) and not np.any(d_hi > 0.0):
        return False  # compact box: the only recession direction is d = 0

    rows = [A]
    row_lo = [np.where(cl > -_INF, 0.0, -_INF)]
    row_hi = [np.where(cu < _INF, 0.0, _INF)]
    if Q is not None:
        rows.append(np.asarray(Q, dtype=np.float64).reshape(n, n))
        row_lo.append(np.zeros(n))
        row_hi.append(np.zeros(n))
    A_ray = np.vstack(rows) if rows[0].shape[0] or len(rows) > 1 else np.empty((0, n))
    cl_ray = np.concatenate(row_lo)
    cu_ray = np.concatenate(row_hi)

    res = _solve_core(
        np.asarray(c, dtype=np.float64),
        A_ray,
        cl_ray,
        cu_ray,
        d_lo,
        d_hi,
        np.zeros(n),
        opts,
    )
    if res.status != SolveStatus.OPTIMAL or res.objective is None:
        return False
    threshold = -_RAY_COST_TOL * max(1.0, float(np.max(np.abs(c))) if n else 1.0)
    return bool(res.objective < threshold)


def _settle_ambiguous_unbounded(
    result: LPResult,
    c: np.ndarray,
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: dict,
) -> LPResult:
    """Keep ``UNBOUNDED`` only when :func:`_certify_unbounded_ray` exhibits a ray.

    Reporting ``ERROR`` otherwise is the honest outcome: the engine could not
    decide, the caller degrades to the exact simplex, and no false certificate is
    produced (#940).
    """
    if result.status != SolveStatus.UNBOUNDED:
        return result
    if _certify_unbounded_ray(c, A, cl, cu, lb, ub, opts):
        return result
    logger.debug(
        "POUNCE exited with an ambiguous Ipopt code 3/4 and no improving recession "
        "direction exists, so the problem is not unbounded; reporting ERROR so the "
        "caller degrades to the exact simplex rather than certifying a false "
        "'unbounded' (#940)."
    )
    return LPResult(
        status=SolveStatus.ERROR, iterations=result.iterations, wall_time=result.wall_time
    )


def _snap_inverted_bounds(lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap tiny ``lb > ub`` inversions to their midpoint (in place-safe)."""
    inverted = lb > ub
    if np.any(inverted):
        tiny = inverted & ((lb - ub) <= _BOUND_SNAP_TOL * (1.0 + np.abs(ub)))
        if np.any(tiny):
            lb = lb.copy()
            ub = ub.copy()
            mid = 0.5 * (lb[tiny] + ub[tiny])
            lb[tiny] = mid
            ub[tiny] = mid
    return lb, ub


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

    Same semantics as :func:`discopt.solvers.lp_simplex.solve_lp`. ``bounds``
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

    # ---- validate dimensions (parity with lp_simplex) -------------------------
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
    lb, ub = _snap_inverted_bounds(lb, ub)

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

    # Shared baseline first, so an explicit caller option still wins; see
    # solvers.pounce_option_defaults for why these are not Ipopt's (#940).
    opts: dict[str, Any] = pounce_option_defaults()
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
        if slacks is not None and _is_infeasible_violation(slacks, cl, cu):
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
        if slacks is not None and _is_infeasible_violation(slacks, cl, cu):
            result.infeasibility_certificate = _build_certificate(slacks, n_ineq)

    # Phase-1 above settles the "was it really infeasible?" reading of an ambiguous
    # code-3/4 exit. This settles the other one: UNBOUNDED survives only if a ray
    # is actually exhibited, so the certificate is earned rather than inferred from
    # a numerical-failure code (#940).
    return _settle_ambiguous_unbounded(result, c_arr, A, cl, cu, lb, ub, opts)


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
    lb, ub = _snap_inverted_bounds(lb, ub)

    cl = b_arr.copy()
    cu = b_arr.copy()
    x0 = _interior_start(lb, ub)

    opts: dict[str, Any] = pounce_option_defaults()
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


def _is_infeasible_violation(slacks: Optional[np.ndarray], cl: np.ndarray, cu: np.ndarray) -> bool:
    """Whether the Phase-1 minimal total violation certifies infeasibility.

    The decision uses a scale-aware threshold rather than the bare absolute
    ``_FEAS_TOL``: the interior-point Phase-1 leaves a small residual per row, so
    a constant tolerance summed over all ``m`` rows raises false ``INFEASIBLE``
    verdicts on large or large-magnitude systems. Scale by the row count and the
    right-hand-side magnitude so only a genuine (data-significant) violation
    trips. Erring toward *not* certifying infeasibility is the safe direction —
    the caller then reports the prior status (e.g. iteration limit) rather than a
    wrong ``INFEASIBLE``.
    """
    if slacks is None:
        return False
    arr = np.asarray(slacks, dtype=np.float64)
    total = float(arr.sum())
    m = int(arr.size)
    # Use only genuinely finite RHS entries for the scale — the ±_INF sentinel
    # (1e20) is "finite" to np.isfinite and would blow the scale up.
    cl = np.asarray(cl, dtype=np.float64).ravel()
    cu = np.asarray(cu, dtype=np.float64).ravel()
    finite = np.concatenate(
        [cl[np.abs(cl) < _FINITE_BOUND_THRESHOLD], cu[np.abs(cu) < _FINITE_BOUND_THRESHOLD]]
    )
    rhs_scale = 1.0 + (float(np.max(np.abs(finite))) if finite.size else 0.0)
    return total > _FEAS_TOL * max(1.0, float(m)) * rhs_scale


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
