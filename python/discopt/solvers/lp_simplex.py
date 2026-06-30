"""Exact LP solver backed by the pure-Rust warm-started simplex.

Mirrors :func:`discopt.solvers.lp_highs.solve_lp` and
:func:`discopt.solvers.lp_pounce.solve_lp` in signature and return type, so it
is a drop-in alternative at call sites that need an LP solved to its **true
vertex optimum** (OBBT, issue #145).

Why a dedicated LP seam over :mod:`discopt.solvers.milp_simplex`? OBBT tightens a
variable's bound to the optimum of ``min``/``max x_i`` over the relaxation
polytope, which is sound *only when that LP is solved exactly*. POUNCE's
interior-point method returns the analytic center of the optimal face — an
objective that can be grossly wrong on an ill-conditioned LP (e.g. a 1e6 linking
coefficient) while still reporting ``OPTIMAL``, over-tightening a bound and
pruning the true optimum (issue #145). The Rust simplex reaches an exact vertex,
so its optimum is a rigorous bound — the same property HiGHS provides, but
self-hosted (no external HiGHS dependency).

This is a *pure* LP adapter: it delegates the matrix marshalling to
:func:`discopt.solvers.milp_simplex.solve_milp` with ``integrality=None`` and
re-wraps the :class:`MILPResult` as an :class:`LPResult`. There is no simplex
basis exposed across the binding, so ``LPResult.basis`` is always ``None`` and
warm-starting is a silent no-op (the ``warm_basis`` keyword is accepted for
signature compatibility and ignored).
"""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import numpy as np
import scipy.sparse as sp

from discopt.solvers import LPResult, SolveStatus

try:
    from discopt._rust import solve_milp_py  # noqa: F401

    SIMPLEX_AVAILABLE = True
except ImportError:
    SIMPLEX_AVAILABLE = False

# Largest bound violation that is snapped back onto the box as numerical noise
# (see ``solve_lp``). Comfortably above observed simplex round-off (~1e-4 on
# wide-range LPs) yet far below any meaningful constraint scale, so a genuine
# solver defect (a large off-box value) is left intact to surface in tests.
_BOUND_SNAP_TOL = 1e-3


def _dense_rows(A: Optional[Union[np.ndarray, sp.spmatrix]], n: int) -> np.ndarray:
    """Dense ``(m, n)`` view of a constraint block, or an empty ``(0, n)``."""
    if A is None:
        return np.zeros((0, n), dtype=np.float64)
    dense = cast("sp.spmatrix", A).toarray() if sp.issparse(A) else np.asarray(A, dtype=np.float64)
    return dense.reshape(-1, n)


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    time_limit: Optional[float] = None,
    warm_basis: Optional[object] = None,  # accepted for compatibility; ignored
    **_kwargs: Any,
) -> LPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq, bounds`` exactly.

    Returns an :class:`LPResult` whose ``objective`` is the simplex vertex
    optimum (a rigorous bound) **with the vertex duals**: ``dual_values`` are the
    row duals and ``reduced_costs`` the per-variable reduced costs, in the same
    sign/order convention as :func:`discopt.solvers.lp_highs.solve_lp` (validated
    equal to HiGHS to machine precision). Exposing them is what lets the
    dual-consuming seams (Benders subproblem, DBBT) run on the pure-Rust simplex
    instead of HiGHS (issue #356). On any non-optimal exit the status is
    propagated and ``objective`` is left ``None`` so callers that require an exact
    bound (OBBT) skip the tightening rather than trust an inexact value.

    The LP is marshalled to the engine's standard form ``A z = b`` with one slack
    per row — ``[0, +inf)`` for an inequality row, pinned ``[0, 0]`` for an
    equality row — and solved cold via the warm-startable Rust simplex (which
    equilibrates internally). The row duals come straight from the optimal basis
    (``y = B⁻ᵀ c_B``, exact at the vertex), and the reduced costs are
    ``c − A_ubᵀ y_ub − A_eqᵀ y_eq``.
    """
    from discopt._rust import solve_lp_warm_py

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    a_ub = _dense_rows(A_ub if (b_ub is not None and np.size(b_ub)) else None, n)
    a_eq = _dense_rows(A_eq if (b_eq is not None and np.size(b_eq)) else None, n)
    m_ub, m_eq = a_ub.shape[0], a_eq.shape[0]
    m = m_ub + m_eq
    b_vec = np.concatenate(
        [
            np.asarray(b_ub, dtype=np.float64).ravel() if m_ub else np.zeros(0),
            np.asarray(b_eq, dtype=np.float64).ravel() if m_eq else np.zeros(0),
        ]
    )

    # Standard form [A_ub | I_ub | 0 ; A_eq | 0 | I_eq] z = b, with slacks in
    # [0, +inf) for the inequality rows and pinned to [0, 0] for the equality rows.
    a_std = np.zeros((m, n + m), dtype=np.float64)
    if m_ub:
        a_std[:m_ub, :n] = a_ub
        a_std[:m_ub, n : n + m_ub] = np.eye(m_ub)
    if m_eq:
        a_std[m_ub:, :n] = a_eq
        a_std[m_ub:, n + m_ub :] = np.eye(m_eq)
    c_std = np.concatenate([c_arr, np.zeros(m)])
    if bounds is not None:
        lb = np.array([lo for lo, _ in bounds], dtype=np.float64)
        ub = np.array([hi for _, hi in bounds], dtype=np.float64)
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, 1e20, dtype=np.float64)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m_ub, 1e20), np.zeros(m_eq)])

    status, x_full, obj, _iters, _cs, _bv, dual, _ray = solve_lp_warm_py(
        np.ascontiguousarray(c_std),
        np.ascontiguousarray(a_std),
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
    )

    status_map = {
        "optimal": SolveStatus.OPTIMAL,
        "infeasible": SolveStatus.INFEASIBLE,
        "unbounded": SolveStatus.UNBOUNDED,
        "iter_limit": SolveStatus.ITERATION_LIMIT,
        "numerical": SolveStatus.ERROR,
    }
    st = status_map.get(status, SolveStatus.ERROR)
    if st != SolveStatus.OPTIMAL:
        return LPResult(status=st)

    x = np.asarray(x_full, dtype=np.float64)[:n].copy()
    # Snap small numerical bound violations onto the box. An LP optimum is a
    # vertex sitting on its active bounds; on some platforms (observed on
    # darwin/arm64 for genuinely wide-range coefficients) the scaled simplex can
    # return a component a hair outside its bound (e.g. x=-1.3e-4 at lb=0). The
    # variable bounds are hard box constraints, so projecting a *small* violation
    # back to the bound restores feasibility without changing the optimum. Only
    # near-bound violations are snapped; a large violation is left intact so a
    # genuine solver defect still surfaces rather than being masked.
    if bounds is not None:
        mm = min(len(x), len(bounds))
        lo = np.array([bounds[i][0] for i in range(mm)], dtype=np.float64)
        hi = np.array([bounds[i][1] for i in range(mm)], dtype=np.float64)
        xm = x[:mm]
        below = (xm < lo) & (xm >= lo - _BOUND_SNAP_TOL)
        above = (xm > hi) & (xm <= hi + _BOUND_SNAP_TOL)
        xm[below] = lo[below]
        xm[above] = hi[above]
        x[:mm] = xm

    # Row duals from the optimal basis (HiGHS row order: inequality rows then
    # equality rows) and the reduced costs c − Aᵀy. Attach only when finite; a
    # consumer that reads them (Benders/DBBT) then degrades gracefully on the rare
    # numerical exit rather than building a cut from a non-finite multiplier.
    y = np.asarray(dual, dtype=np.float64)
    dual_values = y if (y.shape[0] == m and np.all(np.isfinite(y))) else None
    reduced_costs = None
    if dual_values is not None:
        rc = c_arr.copy()
        if m_ub:
            rc = rc - a_ub.T @ y[:m_ub]
        if m_eq:
            rc = rc - a_eq.T @ y[m_ub:]
        if np.all(np.isfinite(rc)):
            reduced_costs = rc

    return LPResult(
        status=SolveStatus.OPTIMAL,
        x=x,
        objective=float(obj),
        dual_values=dual_values,
        reduced_costs=reduced_costs,
        basis=None,
    )


def solve_lp_batch(
    c: np.ndarray,
    A_ub: Union[np.ndarray, sp.spmatrix],
    instances: list[tuple[np.ndarray, list[tuple[float, float]]]],
    *,
    tol: float = 1e-9,
    max_iter: int = 100_000,
) -> list[LPResult]:
    """Solve many LPs ``min c^T x s.t. A_ub x <= b_ub, bounds`` that share ``c``
    and ``A_ub``, one per ``instances`` entry ``(b_ub, bounds)``.

    The shared constraint matrix is marshalled to standard form once and the
    Rust batch path computes the equilibration scaling a single time, reusing it
    for every instance and solving them in parallel. The result list is in input
    order; each is observationally identical to calling :func:`solve_lp` on that
    instance alone. This is the throughput path for re-solving an LP over many
    right-hand sides or bound boxes (the B&B / OBBT / scenario pattern).

    Raises :class:`SimplexBackendUnavailable` if the Rust binding is missing.
    """
    from discopt.solvers.milp_simplex import SimplexBackendUnavailable

    try:
        from discopt._rust import solve_lp_batch_py
    except ImportError as err:  # pragma: no cover - exercised via the selector
        raise SimplexBackendUnavailable(
            "discopt._rust.solve_lp_batch_py is unavailable; build the Rust extension"
        ) from err

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    a = (
        cast("sp.spmatrix", A_ub).toarray()
        if sp.issparse(A_ub)
        else np.asarray(A_ub, dtype=np.float64)
    ).reshape(-1, n)
    m = a.shape[0]

    # Standard form A_eq z = b with one slack per row: [A_ub | I] z = b.
    a_std = np.zeros((m, n + m), dtype=np.float64)
    a_std[:, :n] = a
    a_std[:, n:] = np.eye(m)
    c_std = np.concatenate([c_arr, np.zeros(m)])

    k = len(instances)
    b_stack = np.zeros((k, m), dtype=np.float64)
    lb_stack = np.zeros((k, n + m), dtype=np.float64)
    ub_stack = np.zeros((k, n + m), dtype=np.float64)
    for t, (b_ub, bounds) in enumerate(instances):
        b_stack[t, :] = np.asarray(b_ub, dtype=np.float64).ravel()
        if bounds is not None:
            lb_stack[t, :n] = [lo for lo, _ in bounds]
            ub_stack[t, :n] = [hi for _, hi in bounds]
        else:
            ub_stack[t, :n] = 1e20
        ub_stack[t, n:] = 1e20  # slacks in [0, inf)

    statuses, xs, objs = solve_lp_batch_py(
        np.ascontiguousarray(c_std),
        np.ascontiguousarray(a_std),
        np.ascontiguousarray(b_stack),
        np.ascontiguousarray(lb_stack),
        np.ascontiguousarray(ub_stack),
        tol,
        int(max_iter),
    )

    status_map = {
        "optimal": SolveStatus.OPTIMAL,
        "infeasible": SolveStatus.INFEASIBLE,
        "unbounded": SolveStatus.UNBOUNDED,
        "iter_limit": SolveStatus.ITERATION_LIMIT,
        "numerical": SolveStatus.ERROR,
    }
    results: list[LPResult] = []
    for t in range(k):
        st = status_map.get(statuses[t], SolveStatus.ERROR)
        if st != SolveStatus.OPTIMAL:
            results.append(LPResult(status=st))
            continue
        results.append(
            LPResult(
                status=SolveStatus.OPTIMAL,
                x=np.asarray(xs[t])[:n].copy(),
                objective=float(objs[t]),
                basis=None,
            )
        )
    return results
