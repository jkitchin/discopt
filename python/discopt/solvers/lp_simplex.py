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

from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import LPResult, SolveStatus

try:
    from discopt._rust import solve_milp_py  # noqa: F401

    SIMPLEX_AVAILABLE = True
except ImportError:
    SIMPLEX_AVAILABLE = False


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
    optimum (a rigorous bound). On any non-optimal exit the status is propagated
    and ``objective`` is left ``None`` so callers that require an exact bound
    (OBBT) skip the tightening rather than trust an inexact value.
    """
    from discopt.solvers.milp_simplex import solve_milp

    res = solve_milp(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=None,  # pure LP: no integer columns
        time_limit=time_limit,
    )

    if res.status != SolveStatus.OPTIMAL:
        return LPResult(status=res.status, wall_time=res.wall_time)

    # A pure LP solved to optimality has objective == bound == the true optimum;
    # ``bound`` is the rigorous dual value, so prefer it when present.
    obj = res.bound if res.bound is not None else res.objective
    return LPResult(
        status=SolveStatus.OPTIMAL,
        x=res.x,
        objective=float(obj) if obj is not None else None,
        basis=None,
        wall_time=res.wall_time,
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
    a = (A_ub.toarray() if sp.issparse(A_ub) else np.asarray(A_ub, dtype=np.float64)).reshape(-1, n)
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
