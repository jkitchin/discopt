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
