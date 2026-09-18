"""HiGHS convex-QP backend — the vertex solver POUNCE degrades to.

Why this module exists
----------------------

``_solve_qp_matrix`` guards every QP answer twice: the returned point must be
primal-feasible, and (for an interior-point backend that reports one) its final
KKT residual must be stationary. Those guards are what stop a stalled, drifted
"optimal" from becoming a certificate (#145). Until now the only QP backend was
POUNCE, so a point the guards rejected left the route with **nothing** — a
bounded, feasible, convex QP came back ``error`` (measured: ``min (w - 0.5)^2``
beside an unrelated row of activity ~1e14, where POUNCE's KKT residual is
9.2e-05 against a 1e-06 bar).

``QPResult.kkt_error`` has always documented the intended remedy — it is ``None``
"for vertex solvers like HiGHS that reach an exact optimum", so that a caller
"can degrade to a vertex solver instead of trusting a drifted objective". This is
that vertex solver.

Why this is not the rescue that was removed
-------------------------------------------

#359 removed a JAX QP IPM rescue from exactly this position, and the reason is
worth restating because it is the failure this module must not repeat: that
rescue "did not degrade gracefully, it degraded *past the guard*" — it re-solved
the QP and reported ``status="optimal"`` with ``bound=obj_val`` and ``gap=0`` on
nothing but its own internal convergence flag, with no feasibility check and no
stationarity check, in precisely the situation where a verified engine's answer
had just been thrown out.

This backend returns a plain :class:`QPResult` and is invoked **through**
``_solve_qp_matrix``, so the same feasibility gate that rejected POUNCE's point
judges this one. It issues no certificate of its own. It declines (``ValueError``)
rather than guessing whenever it is handed something it cannot decide exactly:
integer variables, or a Hessian that is not positive semidefinite.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import QPResult, SolveStatus

logger = logging.getLogger(__name__)

#: Eigenvalue floor for the convexity test. HiGHS's QP solver requires a positive
#: semidefinite Hessian; a slightly-negative eigenvalue is the symmetric part's
#: round-off, anything below this is a genuinely indefinite objective HiGHS cannot
#: solve to a global optimum and this module must refuse rather than mis-answer.
_PSD_EIG_TOL = -1e-8

_highspy: Any = None
try:  # pragma: no cover - import-time capability probe
    import highspy as _highspy_mod

    _highspy = _highspy_mod
    HIGHS_QP_AVAILABLE = hasattr(_highspy.Highs(), "passHessian")
except Exception:  # noqa: BLE001 - absence is a capability answer, not an error
    HIGHS_QP_AVAILABLE = False


def _hessian_is_psd(Q: np.ndarray) -> bool:
    """Whether ``0.5 xᵀQx`` is convex, i.e. ``sym(Q)`` is positive semidefinite."""
    if Q.size == 0:
        return True
    sym = 0.5 * (Q + Q.T)
    if not np.all(np.isfinite(sym)):
        return False
    try:
        return bool(np.min(np.linalg.eigvalsh(sym)) >= _PSD_EIG_TOL)
    except np.linalg.LinAlgError:
        return False


def _lower_triangle_csc(Q: np.ndarray):
    """``sym(Q)``'s lower triangle as HiGHS's triangular CSC triple."""
    sym = 0.5 * (Q + Q.T)
    tri = sp.csc_matrix(np.tril(sym))
    tri.eliminate_zeros()
    tri.sort_indices()
    return (
        np.asarray(tri.indptr, dtype=np.int32),
        np.asarray(tri.indices, dtype=np.int32),
        np.asarray(tri.data, dtype=np.float64),
    )


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
    **_kwargs,
) -> QPResult:
    """Solve ``min 0.5 xᵀQx + cᵀx`` s.t. linear constraints, via HiGHS.

    Same contract as :func:`discopt.solvers.qp_pounce.solve_qp`. ``kkt_error`` is
    left ``None``: HiGHS reaches a vertex/exact optimum rather than converging to
    one, so there is no residual to report and the caller's stationarity gate
    correctly skips it. The primal-feasibility gate still applies and is what
    actually accepts or rejects this point.

    Returns ``OPTIMAL`` or ``ERROR`` and nothing else -- see the note at the status
    mapping for why a raw HiGHS ``Infeasible``/``Unbounded`` must not become a
    certificate here.

    Raises
    ------
    ImportError
        If ``highspy`` is unavailable or too old to accept a Hessian.
    ValueError
        On inconsistent dimensions; on any integer-marked variable (HiGHS's QP
        solver is continuous-only); or on a Hessian that is not positive
        semidefinite, which HiGHS cannot solve to a *global* optimum. Refusing is
        the point: a nonconvex QP answered as though it were convex is a false
        certificate, and this backend exists to be a safe last resort.
    """
    if not HIGHS_QP_AVAILABLE or _highspy is None:
        raise ImportError("highspy with QP (passHessian) support is required for this backend")
    if integrality is not None and np.any(np.asarray(integrality) == 1):
        raise ValueError(
            "qp_highs.solve_qp is a continuous QP solver; integrality is not supported."
        )

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = int(c_arr.shape[0])
    Q_arr = np.asarray(Q, dtype=np.float64)
    if Q_arr.shape != (n, n):
        raise ValueError(f"Q has shape {Q_arr.shape} but c has {n} elements")
    if not _hessian_is_psd(Q_arr):
        raise ValueError(
            "HiGHS solves convex QPs only and this Hessian is not positive "
            "semidefinite; refusing rather than returning a local point as optimal."
        )

    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")
        xl = np.array([b[0] for b in bounds], dtype=np.float64)
        xu = np.array([b[1] for b in bounds], dtype=np.float64)
    else:
        xl = np.full(n, -np.inf)
        xu = np.full(n, np.inf)

    inf = _highspy.kHighsInf
    xl = np.where(np.isfinite(xl), xl, -inf)
    xu = np.where(np.isfinite(xu), xu, inf)

    # Rows, as HiGHS's two-sided form: `A_ub x <= b_ub` is (-inf, b], `A_eq x == b`
    # is [b, b].
    blocks, row_lo, row_hi = [], [], []
    if A_ub is not None and b_ub is not None and np.size(b_ub):
        a = sp.csr_matrix(A_ub, dtype=np.float64)
        blocks.append(a)
        bu = np.asarray(b_ub, dtype=np.float64).ravel()
        row_lo.append(np.full(a.shape[0], -inf))
        row_hi.append(bu)
    if A_eq is not None and b_eq is not None and np.size(b_eq):
        a = sp.csr_matrix(A_eq, dtype=np.float64)
        blocks.append(a)
        be = np.asarray(b_eq, dtype=np.float64).ravel()
        row_lo.append(be)
        row_hi.append(be)

    h = _highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit is not None and time_limit > 0:
        h.setOptionValue("time_limit", float(time_limit))

    h.addVars(n, xl, xu)
    h.changeColsCost(n, np.arange(n, dtype=np.int32), c_arr)
    if blocks:
        A = sp.vstack(blocks, format="csr")
        A.sort_indices()
        h.addRows(
            A.shape[0],
            np.concatenate(row_lo),
            np.concatenate(row_hi),
            int(A.nnz),
            np.asarray(A.indptr, dtype=np.int32),
            np.asarray(A.indices, dtype=np.int32),
            np.asarray(A.data, dtype=np.float64),
        )
    if np.any(Q_arr):
        start, index, value = _lower_triangle_csc(Q_arr)
        # ``_highspy`` is deliberately untyped (``Any``): the stubs' ``passHessian``
        # overloads do not cover the raw CSC triple this passes, and the call is
        # exercised directly by ``test_backend_solves_a_convex_qp_with_rows``.
        h.passHessian(n, int(value.size), _highspy.HessianFormat.kTriangular, start, index, value)

    h.run()

    model_status = h.getModelStatus()
    name = h.modelStatusToString(model_status)
    # ONLY ``Optimal`` becomes a verdict. Everything else -- including HiGHS's own
    # ``Infeasible`` and ``Unbounded`` -- becomes ``ERROR``.
    #
    # This is deliberate and load-bearing. ``_solve_qp_matrix`` turns a backend's
    # ``INFEASIBLE`` straight into ``status="infeasible"`` and its ``UNBOUNDED``
    # straight into ``status="unbounded"``, with no cross-check of its own: POUNCE
    # earns that by running the mandatory elastic Phase-1 check *inside* its own
    # ``solve_qp`` before it will return either (#1319), so what reaches the mapper
    # is already verified. A raw HiGHS label carries no such check, and passing one
    # through would be precisely the unverified certificate #1295/#1309/#1319 exist
    # to refuse.
    #
    # Nothing is lost by refusing. This backend runs only where the route was
    # already about to report ``error``, so mapping a non-optimal exit to ``ERROR``
    # leaves that outcome exactly as it is today. The one thing it adds is an
    # ``optimal`` whose point the caller's feasibility gate has verified -- which is
    # the whole job of a last-resort vertex solver.
    status = SolveStatus.OPTIMAL if name == "Optimal" else SolveStatus.ERROR
    if status is SolveStatus.ERROR and name != "Optimal":
        logger.debug("HiGHS QP exited %r; reported as ERROR rather than as a verdict", name)

    x = None
    objective = None
    dual_values = None
    reduced_costs = None
    if status == SolveStatus.OPTIMAL:
        sol = h.getSolution()
        x = np.asarray(sol.col_value, dtype=np.float64)
        if x.size != n:
            return QPResult(status=SolveStatus.ERROR, iterations=0)
        objective = float(h.getInfo().objective_function_value)
        if getattr(sol, "row_dual", None) is not None and blocks:
            dual_values = np.asarray(sol.row_dual, dtype=np.float64)
        if getattr(sol, "col_dual", None) is not None:
            reduced_costs = np.asarray(sol.col_dual, dtype=np.float64)

    return QPResult(
        status=status,
        x=x,
        objective=objective,
        bound=objective,
        gap=0.0 if status == SolveStatus.OPTIMAL else None,
        dual_values=dual_values,
        reduced_costs=reduced_costs,
        iterations=int(h.getInfo().simplex_iteration_count),
        # Deliberately None: a vertex solver has no residual to report, and the
        # caller's stationarity gate is documented to skip it for exactly this
        # kind of backend. The feasibility gate is unaffected.
        kkt_error=None,
    )
