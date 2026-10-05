"""POUNCE's dedicated convex LP/QP interior-point method (``lp-ipm`` / ``qp-ipm``).

The other POUNCE matrix backends (:mod:`discopt.solvers.lp_pounce`,
:mod:`discopt.solvers.qp_pounce`) hand an LP or QP to POUNCE's general
**NLP** engine — the filter line-search interior-point method — through
LP/QP-shaped callbacks. This module instead calls :func:`pounce.qp.solve_qp`,
the specialised convex solver (Mehrotra predictor-corrector with presolve) that
POUNCE's own ``solver_selection=lp-ipm`` / ``qp-ipm`` reach. It backs
``Model.solve(solver="pounce")`` (#1533), whose purpose is that a pure LP or a
convex QP is answered by that algorithm and by nothing else.

Both entry points follow the shared matrix contracts
(:func:`discopt.solvers.lp_simplex.solve_lp` / ``qp_pounce.solve_qp``): same
signature, same :class:`~discopt.solvers.LPResult` / ``QPResult`` with
HiGHS-convention duals, so ``solver._solve_lp_matrix`` and
``solver._solve_qp_matrix`` apply their primal-feasibility and KKT guards to the
returned point unchanged.

Certificates
------------
POUNCE's convex engine reports ``primal_infeasible`` and ``dual_infeasible``
(unbounded) from its own detection. Neither is turned into a certificate on that
say-so alone, by the same rule the NLP-engine routes follow (#1309, #940):

* ``primal_infeasible`` becomes ``INFEASIBLE`` only when the exact Rust simplex
  (:func:`lp_pounce._simplex_feasibility_verdict`) proves the linear system
  empty; otherwise ``ERROR``.
* ``dual_infeasible`` becomes ``UNBOUNDED`` only when an improving recession ray
  is verified exactly (:func:`lp_pounce._certify_unbounded_ray`) **and** the
  simplex exhibits a feasible point; otherwise ``ERROR``.

Both checks only *withhold* a verdict; neither can produce an answer the IPM did
not, so the route never silently becomes a simplex solve.

Bounds
------
A declared bound at or beyond :func:`lp_pounce.finite_bound_threshold` is handed
to the engine as infinite, exactly as on the other POUNCE routes. That includes
the ``±9.999e19`` box a column declared without bounds receives, which, kept
finite, costs the IPM several times the iterations. The caller passes
``relaxes_huge_bounds=True`` to the matrix route so an ``UNBOUNDED`` verdict over
such a relaxed box is not certified (#850).

Objective scale
---------------
The engine's stopping test is not invariant to the scale of the objective: the
same 2080-column production LP that stops at a primal residual of 5.5e-9 with its
costs in dollars stops at 4.9e-6 with the costs in cents (x100) -- 3.5x over the
per-row test ``_matrix_solution_feasible`` applies -- and the solve came back
``error`` (#1537). A QP ``1e8*x + x**2/2`` on ``[0, 10]`` ended in
``numerical_failure``. :func:`_solve` therefore hands the engine
``sigma*P, sigma*c`` with ``sigma`` the power of two that brings the largest
objective coefficient to at most 1 (:func:`objective_scale`), and maps the
answer back: ``x`` is unchanged, objective and every dual divide by ``sigma``.
A power of two makes both directions exact, and ``sigma = 1`` (no change at all)
for every objective whose coefficients are already at most 1.

The engine's ``tol`` is an **absolute** bound on its KKT residual (POUNCE
``QpOptions::tol``, default ``1e-8``; a scale-relative arm opens only below the
finite-precision floor). Dual infeasibility and complementarity are linear in
the objective, so a solve of ``sigma*objective`` stopped at ``tol`` is stopped at
``tol/sigma`` in the caller's units -- 8192x looser on ``3200*j**2``, where the
mapped-back complementarity of 6.6e-6 failed the #1384 stationarity guard and
the solve came back ``error``. The engine is therefore handed ``tol*sigma``
(:func:`engine_tol`): the caller-unit stopping test is the one asked for, and the
primal residual, which ``sigma`` does not touch, is held at least as tight --
down to a floor of ``16*eps`` (:data:`_ENGINE_TOL_FLOOR`), below which the scaled
problem (coefficients at most 1) cannot resolve its residual and the engine runs
to ``iteration_limit``. The floor only ever loosens the request; the #1384
stationarity guard downstream still judges every point the engine returns.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import LPResult, QPResult, SolveStatus
from discopt.solvers.lp_pounce import (
    _INF,
    PHASE1_FEASIBLE,
    PHASE1_INFEASIBLE,
    POUNCE_AVAILABLE,
    _certify_unbounded_ray,
    _simplex_feasibility_verdict,
    _stack_constraints,
    finite_bound_threshold,
)

#: Options the convex engine understands. Anything else is refused by
#: :func:`convex_engine_options` rather than dropped: the NLP engine's options
#: (``mu_strategy``, ``linear_solver``, ...) name machinery this algorithm does
#: not have, and silently ignoring one would leave the caller believing it was
#: set (the M6 rule ``Model.solve`` applies to unknown keyword names).
CONVEX_OPTION_KEYS: frozenset[str] = frozenset(
    {"tol", "max_iter", "max_wall_time", "print_level", "tau", "tau_max"}
)


class IndefiniteQPError(ValueError):
    """The QP's Hessian was not PROVED positive semidefinite.

    Raised by :func:`solve_qp` when :func:`certify_psd` cannot prove ``Q`` PSD, and
    when POUNCE's own PSD check refuses it. The ``solver="pounce"`` route catches it
    and solves the QP with the NLP engine instead, reporting a local result.
    """


#: Up to this many quadratically-active variables :func:`certify_psd` decides PSD
#: by the exact rational test with no budget. Rational entries grow during
#: elimination of a dense full-rank matrix, so the cost is steep: measured on dense
#: random ``A'A``, 0.16 s at n=30, 3.3 s at 60, 228 s at 150.
_EXACT_PSD_MAX_N = 30

#: Above :data:`_EXACT_PSD_MAX_N`, a Hessian with at most this many nonzeros per
#: active row (on average) is still decided exactly, by the sparse elimination
#: under :data:`_EXACT_PSD_UPDATE_BUDGET` (#1616). These are the matrices the
#: eigenvalue margin cannot prove: a singular PSD Hessian such as a graph Laplacian
#: (``sum (x[i+1]-x[i])**2``, ``lambda_min = 0`` exactly) fails any margin
#: ``lambda_min >= K*eps*||Q||``, and such Hessians are typically sparse with little
#: fill. Denser matrices go straight to the eigenvalue test, as before.
_EXACT_PSD_SPARSE_ROW_NNZ = 8

#: Deterministic work budget (exact rational multiply-subtract updates) for the
#: sparse exact elimination above :data:`_EXACT_PSD_MAX_N`. An operation count,
#: never a wall-clock limit, so the route a model takes does not depend on machine
#: load. Exhausting it means "not decided exactly", never "PSD".
_EXACT_PSD_UPDATE_BUDGET = 500_000

#: Bit-length cap on a rational entry in the budgeted elimination; past it the
#: elimination stops as undecided (entry growth is what makes elimination slow).
_EXACT_PSD_MAX_BITS = 2048

#: Above this many quadratically-active variables the dense eigenvalue test is not
#: run at all; only the sparse exact elimination can prove such a Hessian PSD.
_EIG_PSD_MAX_N = 4000


def _exact_psd(Q: np.ndarray, budget: Optional[int] = None) -> Optional[bool]:
    """Decide ``Q`` PSD exactly, by sparse symmetric elimination over the rationals.

    Every float is a dyadic rational, so ``Fraction`` represents ``Q`` exactly and
    the verdict carries no tolerance. A symmetric matrix is PSD iff elimination on a
    positive pivot leaves a PSD Schur complement; a negative diagonal, or a zero
    diagonal whose row is not zero, refutes it. Any positive diagonal is a valid
    pivot (a symmetric permutation preserves PSD-ness), so the pivot is the row with
    the fewest nonzeros (minimum degree, ties by index -- deterministic), which
    keeps sparse elimination sparse.

    Returns ``True``/``False`` when decided. With a ``budget`` (exact updates), also
    returns ``None`` -- undecided -- once the budget or the
    :data:`_EXACT_PSD_MAX_BITS` entry size is exhausted. ``budget=None`` is the
    unbounded exact test.
    """
    import heapq
    from fractions import Fraction

    n = Q.shape[0]
    rows: dict[int, dict[int, Fraction]] = {i: {} for i in range(n)}
    if sp.issparse(Q):
        coo = sp.coo_matrix(Q)
        for i, j, v in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
            if v != 0.0:
                rows[i][j] = Fraction(float(v))
    else:
        ii, jj = np.nonzero(Q)
        for i, j in zip(ii.tolist(), jj.tolist()):
            rows[i][j] = Fraction(float(Q[i, j]))

    def refuted(i: int) -> bool:
        r = rows[i]
        d = r.get(i, 0)
        return d < 0 or (d == 0 and bool(r))

    if any(refuted(i) for i in rows):
        return False
    heap = [(len(r), i) for i, r in rows.items() if r]
    heapq.heapify(heap)
    ops = 0
    while heap:
        deg, p = heapq.heappop(heap)
        if p not in rows or len(rows[p]) != deg:
            continue  # stale entry
        prow = rows.pop(p)
        piv = prow.pop(p)
        nbrs = list(prow.items())
        for i, a_ip in nbrs:
            ri = rows[i]
            del ri[p]
            f = a_ip / piv
            for j, a_pj in nbrs:
                v = ri.get(j, 0) - f * a_pj
                if v == 0:
                    ri.pop(j, None)
                    continue
                if budget is not None and (
                    v.numerator.bit_length() + v.denominator.bit_length() > _EXACT_PSD_MAX_BITS
                ):
                    return None
                ri[j] = v
            ops += len(nbrs)
            if budget is not None and ops > budget:
                return None
        for i, _ in nbrs:
            if refuted(i):
                return False
            if rows[i]:
                heapq.heappush(heap, (len(rows[i]), i))
            else:
                del rows[i]
    return True


def certify_psd(Q: np.ndarray) -> bool:
    """True only when ``Q`` (symmetrized) is PROVED positive semidefinite (#1533 review).

    POUNCE's own check accepts ``lam_min >= -1e-8 * max|Q_ij|``, which admits an
    indefinite Hessian whose negative curvature is small against its largest entry
    but not against the box (``-1e-9 x**2 + y**2`` on ``x in [-1e4, 1e4]``): the
    convex IPM then stops at a saddle and the route would certify it ``optimal``.
    So convexity is decided here, before the QP arm can label anything optimal.

    Rows and columns that are identically zero (variables appearing only linearly)
    are dropped first. Up to :data:`_EXACT_PSD_MAX_N` remaining variables the test
    is exact (:func:`_exact_psd`). Beyond that, a sparse Hessian (at most
    :data:`_EXACT_PSD_SPARSE_ROW_NNZ` nonzeros per row on average) is still decided
    exactly under a deterministic work budget, at any size (#1616) -- this proves
    singular PSD Hessians (graph Laplacians) that no floating-point margin can.
    Otherwise, or when that budget runs out, the computed minimum eigenvalue must
    clear the scale-carrying roundoff margin ``K * eps * ||Q||_2`` the repo already
    uses for this purpose (``solver._CONVEX_OBJ_PSD_EIG_ROUNDOFF_K``, #1397); a
    matrix that does not is treated as unproved, never as PSD.
    """
    if sp.issparse(Q):
        # #1619 A-22: the same test without ever forming the dense (n, n) matrix;
        # only the active block reaches the eigenvalue fallback, densified there.
        Qs = sp.csr_matrix(Q, dtype=np.float64)
        if not np.all(np.isfinite(Qs.data)):
            return False
        Ss = (0.5 * (Qs + Qs.T)).tocsr()
        Ss.eliminate_zeros()
        active = np.flatnonzero(np.diff(Ss.indptr) > 0)
        if active.size == 0:
            return True
        Ss = Ss[active][:, active]
        n = Ss.shape[0]
        if n <= _EXACT_PSD_MAX_N:
            return bool(_exact_psd(Ss))
        if Ss.nnz <= _EXACT_PSD_SPARSE_ROW_NNZ * n:
            exact = _exact_psd(Ss, budget=_EXACT_PSD_UPDATE_BUDGET)
            if exact is not None:
                return exact
        if n > _EIG_PSD_MAX_N:
            return False
        S = Ss.toarray()
    else:
        Q = np.asarray(Q, dtype=np.float64)
        if not np.all(np.isfinite(Q)):
            return False
        S = 0.5 * (Q + Q.T)
        active = np.flatnonzero(np.any(S != 0.0, axis=1))
        S = S[np.ix_(active, active)]
        if S.size == 0:
            return True
        n = S.shape[0]
        if n <= _EXACT_PSD_MAX_N:
            return bool(_exact_psd(S))
        if np.count_nonzero(S) <= _EXACT_PSD_SPARSE_ROW_NNZ * n:
            exact = _exact_psd(S, budget=_EXACT_PSD_UPDATE_BUDGET)
            if exact is not None:
                return exact
        if n > _EIG_PSD_MAX_N:
            return False
    from discopt.solver import _CONVEX_OBJ_PSD_EIG_ROUNDOFF_K

    eigs = np.linalg.eigvalsh(S)
    margin = _CONVEX_OBJ_PSD_EIG_ROUNDOFF_K * np.finfo(np.float64).eps * float(np.max(np.abs(eigs)))
    return bool(eigs.min() >= margin)


def convex_engine_options(options: Optional[dict]) -> dict:
    """Validate ``options`` for the convex engine and return a copy.

    Raises:
        ValueError: on a key outside :data:`CONVEX_OPTION_KEYS`.
    """
    opts = dict(options or {})
    unknown = sorted(k for k in opts if k not in CONVEX_OPTION_KEYS)
    # #1615 A-17/A-24: the convex engine's own knobs (``qp_crossover``,
    # ``qp_presolve``, ``qp_hsde``, ...) are POUNCE options file / CLI options that
    # ``pounce.qp.solve_qp`` exposes no argument for, so no discopt call can set
    # them. Say where they do work rather than lump them in with NLP options.
    cli_only = [k for k in unknown if k.startswith("qp_")]
    if cli_only:
        raise ValueError(
            f"pounce_options {cli_only} are options of POUNCE's convex QP engine "
            f"that only its command line / options file reaches: pounce.qp.solve_qp, "
            f"which solver='pounce' calls for this model, takes no such argument, so "
            f"the setting cannot be applied from discopt. Refused rather than "
            f"ignored. To use it, export the model and run the CLI: "
            f"m.to_nl('model.nl'), then `pounce model.nl {cli_only[0]}=...`. From "
            f"discopt that engine accepts {sorted(CONVEX_OPTION_KEYS)}."
        )
    if unknown:
        raise ValueError(
            f"pounce_options {unknown} are not options of POUNCE's convex LP/QP "
            f"interior-point method (lp-ipm / qp-ipm), which this model was routed "
            f"to. That engine accepts {sorted(CONVEX_OPTION_KEYS)}. Options such as "
            f"mu_strategy or linear_solver belong to the NLP engine, which "
            f"solver='pounce' uses for nonlinear and nonconvex models. Refused rather "
            f"than ignored, so a setting never appears to apply when it does not."
        )
    return opts


def _engine_box(
    bounds: Optional[List[Tuple[float, float]]], n: int, default_lb: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Box with every ``|b| >= finite_bound_threshold()`` mapped to ``±inf``."""
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")
        lb = np.array([b[0] for b in bounds], dtype=np.float64)
        ub = np.array([b[1] for b in bounds], dtype=np.float64)
    else:
        lb = np.full(n, default_lb, dtype=np.float64)
        ub = np.full(n, np.inf, dtype=np.float64)
    thr = finite_bound_threshold()
    lb = np.where(lb <= -thr, -np.inf, lb)
    ub = np.where(ub >= thr, np.inf, ub)
    return lb, ub


def _sentinel(v: np.ndarray) -> np.ndarray:
    """``±inf`` -> the ``±1e20`` sentinel the discopt LP helpers expect."""
    return np.asarray(np.clip(v, -_INF, _INF), dtype=np.float64)


def _print_trace(selector: str, res: Any) -> None:
    """The per-iteration convergence trace, in the spirit of Ipopt's log."""
    print(f"POUNCE {selector} (convex interior-point method)")
    print(
        f"{'iter':>4}  {'objective':>14}  {'inf_pr':>9}  {'inf_du':>9}  "
        f"{'mu':>9}  {'alpha_pr':>8}  {'alpha_du':>8}"
    )
    for it in res.iterates:
        print(
            f"{int(it['iter']):>4}  {float(it['objective']):>14.7e}  "
            f"{float(it['primal_infeasibility']):>9.2e}  "
            f"{float(it['dual_infeasibility']):>9.2e}  {float(it['mu']):>9.2e}  "
            f"{float(it['alpha_primal']):>8.2e}  {float(it['alpha_dual']):>8.2e}"
        )
    print(f"status: {res.status}   iterations: {int(res.iters)}   objective: {res.obj!r}")


def objective_scale(P, c: np.ndarray) -> float:
    """The power of two ``sigma <= 1`` that brings ``max(|P|, |c|)`` to at most 1.

    ``1.0`` when every objective coefficient is already at most 1 in magnitude,
    or the objective is zero. See the module docstring ("Objective scale").
    """
    mags = [float(np.max(np.abs(c))) if c.size else 0.0]
    if P is not None:
        data = P.data if sp.issparse(P) else np.asarray(P)
        if np.size(data):
            mags.append(float(np.max(np.abs(data))))
    m = max(mags)
    if not math.isfinite(m) or m <= 1.0:
        return 1.0
    return math.ldexp(1.0, -math.ceil(math.log2(m)))


#: POUNCE ``QpOptions::default().tol`` (``pounce-convex/src/ipm.rs``), which the
#: Python surface does not expose. ``test_engine_default_tol_is_mirrored`` pins it:
#: ``tol=None`` and ``tol=_ENGINE_DEFAULT_TOL`` must give the identical solve.
_ENGINE_DEFAULT_TOL = 1e-8


#: The smallest engine ``tol`` :func:`engine_tol` asks for. Measured 2026-10-04 on
#: ten solves (``test_1537_pounce_objective_scale.py``: the production LP with
#: costs x1e-2..x1e8, ``1e8*x + x**2/2`` bounded and free, ``k/2*j**2`` with
#: ``k`` = 6400, 1e6, 1e8): ``1*eps`` sends the x1e6 and x1e8 LPs to
#: ``iteration_limit``; ``4*eps`` to ``20*eps`` pass all ten; ``64*eps`` loses
#: ``k = 1e8`` to the stationarity guard. ``16*eps`` sits ~4x inside both edges.
_ENGINE_TOL_FLOOR = 16.0 * float(np.finfo(np.float64).eps)


def engine_tol(tol: Optional[float], sigma: float) -> Optional[float]:
    """The engine tolerance that keeps ``tol`` in the caller's units under ``sigma``.

    ``None`` (the engine default) passes through when ``sigma == 1``, so an
    unscaled solve is bit-for-bit what it was. See the module docstring.
    """
    if sigma == 1.0:
        return tol
    want = (_ENGINE_DEFAULT_TOL if tol is None else float(tol)) * sigma
    return max(want, _ENGINE_TOL_FLOOR)


def _unscale_result(res: Any, sigma: float) -> Any:
    """``res`` of the engine solve of ``sigma * objective``, in the caller's units.

    ``x`` and the primal residual do not depend on the objective; the objective,
    every multiplier, the dual residual and complementarity (``z * slack``) are
    linear in it. ``kkt_error`` is the max of the three residuals, so it is
    recomputed from the rescaled parts rather than divided as a whole.
    """
    if sigma == 1.0:
        return res
    inv = 1.0 / sigma  # exact: sigma is a power of two
    resid = dict(res.residuals or {})
    for key in ("dual_infeasibility", "complementarity"):
        if resid.get(key) is not None:
            resid[key] = float(resid[key]) * inv
    parts = [
        float(v)
        for k in ("primal_infeasibility", "dual_infeasibility", "complementarity")
        if (v := resid.get(k)) is not None
    ]
    if parts:
        kkt = max(parts)
    elif res.kkt_error is not None:
        kkt = float(res.kkt_error) * max(1.0, inv)  # no breakdown: the safe upper bound
    else:
        kkt = None
    if "kkt_error" in resid:
        resid["kkt_error"] = kkt

    def mult(v):
        return None if v is None else np.asarray(v, dtype=np.float64) * inv

    iterates = [
        {
            **it,
            **{
                k: float(it[k]) * inv
                for k in ("objective", "dual_infeasibility")
                if it.get(k) is not None
            },
        }
        for it in (res.iterates or [])
    ]
    return SimpleNamespace(
        status=res.status,
        success=getattr(res, "success", None),
        iters=res.iters,
        x=res.x,
        obj=None if res.obj is None else float(res.obj) * inv,
        y=mult(res.y),
        z=mult(res.z),
        z_lb=mult(res.z_lb),
        z_ub=mult(res.z_ub),
        kkt_error=kkt,
        residuals=resid,
        iterates=iterates,
        scaling_warning=getattr(res, "scaling_warning", None),
        objective_scale=sigma,
    )


def _solve(
    P: Optional[np.ndarray],
    c: np.ndarray,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    lb: np.ndarray,
    ub: np.ndarray,
    time_limit: Optional[float],
    options: Optional[dict],
    solve_report: bool = False,
    x0: Optional[np.ndarray] = None,
) -> Tuple[
    str,
    Any,
    float,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict],
]:
    """Run :func:`pounce.qp.solve_qp` once; map nothing yet.

    Returns ``(raw_status, result, wall, A, cl, cu, report)`` where ``A, cl, cu``
    is the stacked row system the certificate checks need (``None`` on an
    ``optimal`` status, which never reaches those checks -- #1619) and ``report`` is the
    ``pounce.solve-report/v1`` document when ``solve_report`` is set (#1534),
    else ``None``.
    """
    from pounce.qp import solve_qp as _pounce_solve_qp

    opts = convex_engine_options(options)
    n = len(c)
    print_level = int(opts.pop("print_level", 0) or 0)
    max_wall = opts.pop("max_wall_time", None)
    budgets = [float(t) for t in (time_limit, max_wall) if t is not None]
    limit = min(budgets) if budgets else None

    G = None if A_ub is None else (A_ub if sp.issparse(A_ub) else np.asarray(A_ub, float))
    A = None if A_eq is None else (A_eq if sp.issparse(A_eq) else np.asarray(A_eq, float))
    h = None if b_ub is None else np.asarray(b_ub, dtype=np.float64).ravel()
    b = None if b_eq is None else np.asarray(b_eq, dtype=np.float64).ravel()

    sigma = objective_scale(P, c)  # module docstring, "Objective scale" (#1537)
    P_eng = P if P is None or sigma == 1.0 else P * sigma
    c_eng = c if sigma == 1.0 else c * sigma

    started_unix_nanos = time.time_ns()
    t0 = time.perf_counter()
    try:
        res = _pounce_solve_qp(
            P=P_eng,
            c=c_eng,
            A=A,
            b=b,
            G=G,
            h=h,
            lb=lb,
            ub=ub,
            tol=engine_tol(opts.get("tol"), sigma),
            max_iter=opts.get("max_iter"),
            time_limit=limit,
            collect_iterates=print_level > 0 or solve_report,
            method="ipm",
            tau=opts.get("tau"),
            tau_max=opts.get("tau_max"),
            # #1615 B-01b: a primal start seeds the IPM; it does not change the
            # solution (the objective scale touches only the duals, not ``x``).
            warm_start=None if x0 is None else {"x": np.asarray(x0, dtype=np.float64)},
        )
    except ValueError as exc:
        # POUNCE's PSD guard: the convex engine refuses an indefinite P before
        # iterating. Re-raised as a distinct type so the route can tell it from
        # a malformed-input error, which must still propagate.
        if P is not None and "positive semidefinite" in str(exc):
            raise IndefiniteQPError(str(exc)) from exc
        raise
    wall = time.perf_counter() - t0
    res = _unscale_result(res, sigma)
    if print_level > 0:
        _print_trace("lp-ipm" if P is None else "qp-ipm", res)

    # The stacked DENSE row system is what the infeasible / unbounded certificate
    # checks in ``_verdict_status`` consume, and nothing else does. Build it only on
    # the status that reaches them (#1619): on an ``optimal`` solve it was an
    # (m, n) float64 copy of sparse input -- 64 MB on the 2000-row chain LP of
    # #1619 -- that was then thrown away.
    A_rows: Optional[np.ndarray] = None
    cl: Optional[np.ndarray] = None
    cu: Optional[np.ndarray] = None
    if res.status != "optimal":
        A_rows, cl, cu = _stack_constraints(A_ub, b_ub, A_eq, b_eq, n)
    n_rows = (0 if A_ub is None or b_ub is None else int(A_ub.shape[0])) + (
        0 if A_eq is None or b_eq is None else int(A_eq.shape[0])
    )
    report = None
    if solve_report:
        from discopt.solvers._pounce_report import convex_report

        report = convex_report(
            res,
            wall_time=wall,
            n_constraints=n_rows,
            started_unix_nanos=started_unix_nanos,
        )
    return res.status, res, wall, A_rows, cl, cu, report


def _verdict_status(
    raw: str,
    c: np.ndarray,
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    Q: Optional[np.ndarray],
) -> Tuple[SolveStatus, str]:
    """Map a non-optimal engine status, verifying any certificate it implies.

    Returns ``(status, reason)``; ``reason`` names the engine's own status and,
    when a verdict is withheld, the check that withheld it (#1618 A-09).
    """
    lbs, ubs = _sentinel(lb), _sentinel(ub)
    if raw == "primal_infeasible":
        if _simplex_feasibility_verdict(A, cl, cu, lbs, ubs) == PHASE1_INFEASIBLE:
            return SolveStatus.INFEASIBLE, ""
        return SolveStatus.ERROR, (
            "the engine reported 'primal_infeasible', but the exact simplex did not "
            "prove the constraint system empty, so 'infeasible' is not certified"
        )
    if raw == "dual_infeasible":
        from discopt.solvers import pounce_option_defaults

        ray = _certify_unbounded_ray(c, A, cl, cu, lbs, ubs, pounce_option_defaults(), Q=Q)
        if not ray:
            return SolveStatus.ERROR, (
                "the engine reported 'dual_infeasible' (objective unbounded), but no "
                "improving recession ray passed the exact-arithmetic check, so "
                "'unbounded' is not certified"
            )
        if _simplex_feasibility_verdict(A, cl, cu, lbs, ubs) == PHASE1_FEASIBLE:
            return SolveStatus.UNBOUNDED, ""
        return SolveStatus.ERROR, (
            "the engine reported 'dual_infeasible' and an improving recession ray was "
            "verified, but the exact simplex did not exhibit a feasible point, so "
            "'unbounded' is not certified"
        )
    if raw == "time_limit":
        return SolveStatus.TIME_LIMIT, ""
    if raw in ("iteration_limit", "optimal_inaccurate"):
        # Reached for ``optimal_inaccurate`` only from :func:`solve_lp`: the LP matrix
        # route publishes an ``optimal`` point's objective as its bound with no
        # certificate of its own, so a point the engine could not converge to ``tol``
        # is not handed to it. :func:`solve_qp` passes it on to its #1596 certificate.
        return SolveStatus.ITERATION_LIMIT, f"the engine reported {raw!r}"
    return SolveStatus.ERROR, f"the engine reported {raw!r}"


def _kkt_parts(res: Any, n_ub: int) -> Tuple[np.ndarray, np.ndarray]:
    """HiGHS-convention row duals and reduced costs from a POUNCE ``QpResult``.

    POUNCE's Lagrangian is ``½x'Px + c'x + y'(Ax-b) + z'(Gx-h) - z_lb'(x-lb) +
    z_ub'(x-ub)``; HiGHS reports ``∂obj/∂rhs``, which is ``-z`` for the ``G``
    (``A_ub``) rows and ``-y`` for the equality rows, stacked inequality rows
    first. Reduced costs are ``z_lb - z_ub`` (``c + P x - A'y_h``).
    """
    z = np.asarray(res.z, dtype=np.float64).ravel()[:n_ub]
    y = np.asarray(res.y, dtype=np.float64).ravel()
    dual = -np.concatenate([z, y])
    rc = np.asarray(res.z_lb, dtype=np.float64) - np.asarray(res.z_ub, dtype=np.float64)
    return dual, rc


def _onto_box(x, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """The engine's point projected onto its own simple bounds (#1537).

    The interior-point method's final iterate can sit a rounding error outside a
    bound it was handed: ``min 1e8*x + x**2/2`` over ``x >= 0`` returned
    ``x = -1.2e-16`` and published the objective ``-1.2e-8``, below the true
    optimum 0, at a point outside the declared box. The box is the one constraint
    that can be met exactly, so the point is clipped onto it (``lb``/``ub`` are the
    engine's: a bound it treated as infinite is not imposed). Nothing is trusted
    from the clip -- the route's feasibility guard and certificate run on the
    clipped point.
    """
    return np.asarray(np.clip(np.asarray(x, dtype=np.float64), lb, ub), dtype=np.float64)


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    warm_basis: Optional[object] = None,
    time_limit: Optional[float] = None,
    options: Optional[dict] = None,
    solve_report: bool = False,
) -> LPResult:
    """Solve ``min c'x`` s.t. ``A_ub x <= b_ub``, ``A_eq x = b_eq`` with POUNCE's lp-ipm.

    ``bounds`` default to ``(0, +inf)`` (the shared LP contract). ``warm_basis``
    is accepted for signature compatibility and ignored: an IPM has no basis.
    ``options`` must be a subset of :data:`CONVEX_OPTION_KEYS`.
    ``solve_report=True`` attaches the solve's ``pounce.solve-report/v1``
    document as ``LPResult.solve_report`` (#1534), built by
    :func:`discopt.solvers._pounce_report.convex_report`.
    """
    del warm_basis
    if not POUNCE_AVAILABLE:
        raise ImportError("pounce is required. Install it with:\n  pip install pounce-solver")
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    lb, ub = _engine_box(bounds, n, default_lb=0.0)
    raw, res, wall, A, cl, cu, report = _solve(
        None, c_arr, A_ub, b_ub, A_eq, b_eq, lb, ub, time_limit, options, solve_report
    )
    iters = int(res.iters)
    if raw != "optimal":
        assert A is not None and cl is not None and cu is not None
        status, why = _verdict_status(raw, c_arr, A, cl, cu, lb, ub, None)
        return LPResult(
            status=status,
            iterations=iters,
            wall_time=wall,
            ray_verified=True if status == SolveStatus.UNBOUNDED else None,
            solve_report=report,
            message=why,
        )
    n_ub = 0 if A_ub is None else int(A_ub.shape[0])
    dual, rc = _kkt_parts(res, n_ub)
    x = _onto_box(res.x, lb, ub)
    return LPResult(
        status=SolveStatus.OPTIMAL,
        x=x,
        objective=float(c_arr @ x),
        dual_values=dual,
        reduced_costs=rc,
        rc_absum=np.abs(np.asarray(res.z_lb)) + np.abs(np.asarray(res.z_ub)),
        iterations=iters,
        wall_time=wall,
        solve_report=report,
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
    options: Optional[dict] = None,
    solve_report: bool = False,
    warm_start: Optional[np.ndarray] = None,
) -> QPResult:
    """Solve ``min ½x'Qx + c'x`` s.t. linear rows with POUNCE's qp-ipm.

    ``bounds`` default to free variables (the shared QP contract).
    ``solve_report=True`` attaches the solve's ``pounce.solve-report/v1``
    document as ``QPResult.solve_report`` (#1534). ``warm_start`` is a primal
    point of length ``n`` that seeds the interior-point iteration (#1615 B-01b);
    it never changes the answer.

    Raises:
        IndefiniteQPError: ``Q`` is not PSD (POUNCE's own check).
        ValueError: any integer-marked variable, or an unknown option.
    """
    del gap_tolerance  # a continuous QP has no gap to close
    if not POUNCE_AVAILABLE:
        raise ImportError("pounce is required. Install it with:\n  pip install pounce-solver")
    if integrality is not None and np.any(np.asarray(integrality) == 1):
        raise ValueError("POUNCE's convex QP IPM is continuous; integrality is not supported.")
    # #1619 A-22: a sparse Q stays sparse (POUNCE takes scipy.sparse P).
    Q_arr: Any = sp.csr_matrix(Q) if sp.issparse(Q) else np.asarray(Q, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    if Q_arr.shape != (n, n):
        raise ValueError(f"Q has shape {Q_arr.shape} but c has {n} elements")
    if not certify_psd(Q_arr):
        raise IndefiniteQPError(
            "discopt could not prove the QP Hessian positive semidefinite, so the "
            "convex QP IPM may not certify its answer."
        )
    lb, ub = _engine_box(bounds, n, default_lb=-np.inf)
    x0 = None
    if warm_start is not None:
        x0 = np.asarray(warm_start, dtype=np.float64).ravel()
        if x0.shape != (n,):
            # POUNCE silently ignores a mismatched start; refuse it here instead.
            raise ValueError(f"warm_start has {x0.size} entries for a QP with {n} variables")
    raw, res, wall, A, cl, cu, report = _solve(
        Q_arr, c_arr, A_ub, b_ub, A_eq, b_eq, lb, ub, time_limit, options, solve_report, x0
    )
    iters = int(res.iters)
    # ``optimal_inaccurate`` is the engine's own ``optimal`` iterate re-judged on its
    # normalized KKT measure (POUNCE gh#984) and found above ``tol`` -- a converged
    # point, not an iteration cap. It goes to the caller as a candidate, exactly as
    # ``optimal`` does: on this route the engine's label never certified anything.
    # ``solver._solve_qp_matrix`` refuses it unless it is primal feasible and passes
    # the #1384 stationarity guard, and publishes a bound only when the #1596
    # certificate, which charges the unconverged complementarity, closes the gap;
    # otherwise it is an uncertified ``feasible`` point. Mapping it to
    # ``ITERATION_LIMIT`` dropped the point and named a cap the engine never hit:
    # the 1e-6-scaled row of #1617 stopped after 17 iterations as
    # ``optimal_inaccurate`` under pounce ``main`` and as ``optimal`` under the
    # 0.12.0 wheel, with bit-identical iterates.
    if raw not in ("optimal", "optimal_inaccurate"):
        assert A is not None and cl is not None and cu is not None
        # The ray certificate stacks Q under the (already dense) rows; it runs
        # only on this non-optimal path, so densifying a sparse Q here is the
        # same trade the rows make (#1619).
        Q_dense = Q_arr.toarray() if sp.issparse(Q_arr) else Q_arr
        status, why = _verdict_status(raw, c_arr, A, cl, cu, lb, ub, Q_dense)
        return QPResult(
            status=status,
            iterations=iters,
            wall_time=wall,
            solve_report=report,
            message=why,
        )
    n_ub = 0 if A_ub is None else int(A_ub.shape[0])
    dual, rc = _kkt_parts(res, n_ub)
    x = _onto_box(res.x, lb, ub)
    return QPResult(
        status=SolveStatus.OPTIMAL,
        x=x,
        objective=float(0.5 * x @ (Q_arr @ x) + c_arr @ x),
        dual_values=dual,
        reduced_costs=rc,
        iterations=iters,
        wall_time=wall,
        kkt_error=res.kkt_error,
        solve_report=report,
        message="" if raw == "optimal" else f"the engine reported {raw!r}",
    )
