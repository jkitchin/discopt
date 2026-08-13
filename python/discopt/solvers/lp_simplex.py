"""Exact LP solver backed by the pure-Rust warm-started simplex.

Mirrors :func:`discopt.solvers.lp_pounce.solve_lp` and
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


# Keywords of the shared matrix-LP contract (:mod:`lp_pounce`, :mod:`gurobi`) that
# this backend has no use for and legitimately ignores: an IPM warm-start point, an
# engine-specific options dict, a Farkas-certificate request (the simplex always
# reports its own status), a thread count (the solve is single-threaded). They are
# accepted so a caller can hand the same kwargs to any backend through
# :func:`discopt.solvers.lp_backend.get_lp_solver`.
_IGNORED_LP_KWARGS = frozenset({"x0", "options", "certificate", "threads"})


def _reject_unknown_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise on a keyword this backend neither honors nor knowingly ignores.

    ``solve_lp`` takes ``**_kwargs`` only for cross-backend signature
    compatibility, and a bare catch-all is a silent-wrong-answer seam: a caller
    passing ``lb=``/``ub=`` arrays (the spelling several other discopt LP helpers
    use) had them swallowed and got the *default* ``[0, 1e20]`` box back, with an
    ``OPTIMAL`` status over a box it never asked for (issue #937, side finding).
    ``Model.solve`` rejects unknown options loudly for exactly this reason; so does
    this. The known-inert contract keywords in :data:`_IGNORED_LP_KWARGS` still
    pass through silently.
    """
    unknown = sorted(set(kwargs) - _IGNORED_LP_KWARGS)
    if unknown:
        raise TypeError(
            f"lp_simplex.solve_lp() got unexpected keyword argument(s): "
            f"{', '.join(unknown)}. The variable box is set through `bounds=` "
            f"(a list of (lo, hi) pairs), not `lb=`/`ub=`."
        )


#: The LP layer's "no bound" sentinel. The Rust simplex tests openness as
#: ``ub >= INF`` / ``lb <= -INF`` with ``INF = 1e20``; anything at or beyond it is
#: unbounded on that side.
_LP_INF = 1e20


def _finite_box(lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate a modeling-layer box into the LP layer's sentinel convention.

    The two layers spell "no bound" differently: ``Model.continuous(ub=None)``
    stores **NaN**, while the simplex reads the sentinel ``±1e20``. A NaN that
    crosses untranslated is not merely unhelpful — it is read *both ways*, since
    every comparison against NaN is false. The ratio test's ``ub < INF`` calls a
    NaN upper bound open and steps to ``t = INF``; the unbounded-ray box-recession
    check's ``ub >= INF`` calls the same bound closed. Issue #1008: a Benders
    recourse LP over ``w ∈ [0, NaN]`` was walked to an unbounded ray the box could
    not certify, so the verdict depended on which guard you asked.

    NaN means unbounded on that side, and ``±inf`` (and any magnitude past the
    sentinel, which the simplex already treats as unbounded) is clamped onto it so
    the two readings agree. Finite bounds pass through untouched.
    """
    lo = np.where(np.isnan(lb), -_LP_INF, np.maximum(lb, -_LP_INF))
    hi = np.where(np.isnan(ub), _LP_INF, np.minimum(ub, _LP_INF))
    return lo, hi


def _dense_rows(A: Optional[Union[np.ndarray, sp.spmatrix]], n: int) -> np.ndarray:
    """Dense ``(m, n)`` view of a constraint block, or an empty ``(0, n)``."""
    if A is None:
        return np.zeros((0, n), dtype=np.float64)
    dense = cast("sp.spmatrix", A).toarray() if sp.issparse(A) else np.asarray(A, dtype=np.float64)
    return dense.reshape(-1, n)


def _sparse_rows(A: Optional[Union[np.ndarray, sp.spmatrix]], n: int) -> "sp.csr_matrix":
    """Sparse CSR ``(m, n)`` view of a constraint block, or an empty ``(0, n)``.

    Never densifies a sparse input — the whole point of the sparse marshaling in
    :func:`solve_lp`. A dense ndarray input is reshaped to ``(-1, n)`` (matching
    :func:`_dense_rows`) then converted to CSR.
    """
    if A is None:
        return sp.csr_matrix((0, n), dtype=np.float64)
    if sp.issparse(A):
        M = sp.csr_matrix(A)
        return M if M.shape[1] == n else M.reshape((-1, n))
    return sp.csr_matrix(np.asarray(A, dtype=np.float64).reshape(-1, n))


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    time_limit: Optional[float] = None,
    warm_basis: Optional[object] = None,  # accepted for compatibility; ignored
    max_iter: Optional[int] = None,
    **_kwargs: Any,
) -> LPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq, bounds`` exactly.

    Returns an :class:`LPResult` whose ``objective`` is the simplex vertex
    optimum (a rigorous bound) **with the vertex duals**: ``dual_values`` are the
    row duals and ``reduced_costs`` the per-variable reduced costs, in the same
    sign/order convention as :func:`discopt.solvers.lp_pounce.solve_lp` (validated
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

    ``max_iter`` caps the (cold) pivot count. It is *soundness-neutral*: within
    the cap the returned optimum is exactly the vertex optimum (a rigorous
    bound); if the cap is hit the status is ``ITERATION_LIMIT`` and no objective
    is returned, so a caller that wants a robust result must fall back (never a
    wrong bound). This lets a caller bound the rare cold-stall on a wide,
    ill-conditioned LP (the F3 multilinear vertex-hull LP: ``2^n`` λ columns)
    without waiting for the 100 000-pivot default.

    The variable box comes **only** from ``bounds``. An unrecognized keyword
    raises :class:`TypeError` rather than being swallowed by ``**_kwargs`` — see
    :func:`_reject_unknown_kwargs`.
    """
    _reject_unknown_kwargs(_kwargs)

    from discopt._rust import solve_lp_warm_csc_py

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]

    # Constraint blocks kept SPARSE. A dense standard form ``a_std`` here is
    # ``O(m*(n+m))`` — for a large lifted relaxation (qap's 85756-row McCormick LP:
    # ``a_std`` = 85756 x 107405 ~ 9.2e9 cells ~ 73 GB) it blows memory before the
    # solve. This oracle is the exact-LP seam the PSD/OBBT/DBBT paths call, so the
    # blowup surfaced there (~46 GB, issue: sparse-milp-plan T7). The CSC-native
    # ``solve_lp_warm_csc_py`` consumes the standard form directly with the slack
    # identity left implicit-sparse.
    a_ub = _sparse_rows(A_ub if (b_ub is not None and np.size(b_ub)) else None, n)
    a_eq = _sparse_rows(A_eq if (b_eq is not None and np.size(b_eq)) else None, n)
    m_ub, m_eq = a_ub.shape[0], a_eq.shape[0]
    m = m_ub + m_eq
    b_vec = np.concatenate(
        [
            np.asarray(b_ub, dtype=np.float64).ravel() if m_ub else np.zeros(0),
            np.asarray(b_eq, dtype=np.float64).ravel() if m_eq else np.zeros(0),
        ]
    )

    # Standard form [A_ub | I_ub | 0 ; A_eq | 0 | I_eq] z = b, built directly as
    # CSC. Structural columns first, then one slack per row: [0, +inf) for the
    # inequality rows and pinned to [0, 0] for the equality rows. ``sort_indices``
    # gives ascending row order per column, which the CSC-native simplex requires.
    a_struct = sp.vstack([a_ub, a_eq], format="csr") if m else sp.csr_matrix((0, n))
    if m > 0:
        a_std = sp.hstack(
            [a_struct, sp.identity(m, format="csc", dtype=np.float64)], format="csc"
        ).tocsc()
        a_std.sort_indices()
    else:
        a_std = sp.csc_matrix((0, n), dtype=np.float64)
    c_std = np.concatenate([c_arr, np.zeros(m)])
    if bounds is not None:
        # `None`/NaN on either side means "no bound" here (scipy's spelling and
        # the modeling layer's); `_finite_box` maps both onto the LP sentinel.
        lb, ub = _finite_box(
            np.array([lo for lo, _ in bounds], dtype=np.float64),
            np.array([hi for _, hi in bounds], dtype=np.float64),
        )
    else:
        lb = np.zeros(n, dtype=np.float64)
        ub = np.full(n, _LP_INF, dtype=np.float64)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m_ub, _LP_INF), np.zeros(m_eq)])

    _warm_kw: dict[str, Any] = {}
    if max_iter is not None:
        _warm_kw["max_iter"] = int(max_iter)
    from discopt import _timing

    with _timing.charge("rust"):
        status, x_full, obj, _iters, _cs, _bv, dual, _ray = solve_lp_warm_csc_py(
            np.ascontiguousarray(c_std),
            m,
            n + m,
            np.ascontiguousarray(a_std.indptr, dtype=np.int64),
            np.ascontiguousarray(a_std.indices, dtype=np.int64),
            np.ascontiguousarray(a_std.data, dtype=np.float64),
            np.ascontiguousarray(b_vec),
            np.ascontiguousarray(lb_std),
            np.ascontiguousarray(ub_std),
            None,  # no warm basis (cold solve)
            None,
            **_warm_kw,
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
            rc = rc - np.asarray(a_ub.T @ y[:m_ub]).ravel()
        if m_eq:
            rc = rc - np.asarray(a_eq.T @ y[m_ub:]).ravel()
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
            lb_stack[t, :n], ub_stack[t, :n] = _finite_box(
                np.array([lo for lo, _ in bounds], dtype=np.float64),
                np.array([hi for _, hi in bounds], dtype=np.float64),
            )
        else:
            ub_stack[t, :n] = _LP_INF
        ub_stack[t, n:] = _LP_INF  # slacks in [0, inf)

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
