"""HiGHS matrix-form MILP backend for the OA / GDP-LOA **master** (issue #1060).

Why this exists when #356 removed HiGHS
=======================================
#356 took HiGHS off the *per-node LP* path of the spatial branch-and-bound, where
the in-house Rust simplex gives a rigorous, warm-startable bound and HiGHS did
not. That decision is untouched here: this module is never in the ``"auto"``
fallback order and never solves a spatial-B&B node. It is an **opt-in master
engine** (``milp_solver="highs"``) for the MIP-NLP family, whose master is a
plain MILP whose only product is a dual bound and an integer point.

The measured reason (#1060, ``rsyn0840m`` master, n=280, 1029 rows, 104 binaries,
published master optimum -482.206969, root LP -2778.1802164922506 — identical to
HiGHS's to 4.09e-12, so the *relaxation* is right and only the search differs):

===============================================  ============  ==========
root loop                                        bound         gap closed
===============================================  ============  ==========
in-house driver, knapsack cover + GMI (today)      -2778.18         0.0%
  + MIR and aggregation c-MIR                      -2536.13        10.5%
  + probing-derived implied bounds                 -1912.49        37.7%
  all of the above, iterated to tailing-off        -1685.31        47.6%
HiGHS, all families, node 0                         -801.81        86.1%
===============================================  ============  ==========

HiGHS then finishes that master in **92 nodes / 0.42 s**; the in-house driver was
still at a -802.77 bound after 655 003 nodes and 60 s, *even when seeded with the
exact optimum* — so the deficiency is dual, not primal, and it is a cut-arsenal
gap (clique, flow cover, tableau-aggregated c-MIR, path) that a root loop built
from the families in-tree provably cannot close: iterating those to tailing-off
plateaus at 47.6%.

Soundness contract
==================
``bound`` is HiGHS's ``mip_dual_bound`` and nothing else. Reading the incumbent as
a bound would inflate the OA global LB past the true optimum and falsely certify
optimality — the failure :class:`~discopt.solvers.MILPResult` documents. When
HiGHS reports no usable dual bound, ``bound`` is ``None``; it is never synthesized.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus
from discopt.solvers.lp_milp_highs import MILP_PRESOLVE_RULE_OFF
from discopt.solvers.milp_simplex import _INF, BoundList, _marshal_col_bounds

logger = logging.getLogger(__name__)

# #1634: every master solve here switches off the same HiGHS presolve rule the
# pure-MILP route does (:data:`~discopt.solvers.lp_milp_highs.MILP_PRESOLVE_RULE_OFF`,
# rule 13, ParallelRowsAndCols). Measured on the #1634 generator written with explicit
# slacks (``A x + s = b``) plus a convex quadratic, so it takes the convex-MINLP route
# with OA on this master: 35/1200 false certificates on main (seed 181 at cost scale
# 1e-3 certified 0.02082 against an enumerated 0.00801). The same masters through this
# backend directly: 35/1200 false bounds with HiGHS defaults, 1/1200 with rule 13 off.
#: Row/bound slack the #1634 cross-check grants a point before it counts as verified,
#: relative to ``max(1, |rhs|, |a| @ |x|)``. A verified point is only ever used to
#: LOWER a bound, so a loose value errs toward a weaker bound, never a false one.
_VERIFY_REL = 1e-6
#: Integrality slack for the same check (CLAUDE.md: integrality 1e-5).
_VERIFY_INT = 1e-5


class HighsBackendUnavailable(ImportError):
    """``highspy`` is not installed, so the HiGHS master backend cannot be used."""


def _require_highspy():
    try:
        import highspy
    except ImportError as err:  # pragma: no cover - exercised via the selector
        raise HighsBackendUnavailable(
            "milp_solver='highs' needs the optional HiGHS backend: pip install highspy"
        ) from err
    return highspy


def _to_highs_inf(arr: np.ndarray, highspy) -> np.ndarray:
    """Map discopt's ``1e20`` open-bound sentinel onto HiGHS's own infinity.

    discopt treats ``|v| >= 1e20`` as unbounded (CLAUDE.md), HiGHS uses ``1e30``.
    Handing HiGHS a literal ``1e20`` would make an open bound a *finite* one and
    silently change the model, so the sentinel is translated rather than passed.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out >= _INF] = highspy.kHighsInf
    out[out <= -_INF] = -highspy.kHighsInf
    return out


def _highs_matrix_window(h, highspy) -> tuple[float, float]:
    """HiGHS's ``(small_matrix_value, large_matrix_value)`` -- the window it accepts.

    An entry at or below the small value is DROPPED and reported as ``kWarning``;
    one at or above the large value is refused outright. Both are read from the live
    solver rather than hardcoded, so a caller that moves either option stays in sync
    with the row preparation below instead of silently disagreeing with it.
    """
    out = []
    for key in ("small_matrix_value", "large_matrix_value"):
        st, val = h.getOptionValue(key)
        if st != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS would not report {key} (status {st})")
        out.append(float(val))
    return out[0], out[1]


def _violates_pending(a: np.ndarray, r: float, x: np.ndarray) -> bool:
    """Whether ``x`` violates the pending lazy row ``a @ x <= r`` (#1658).

    The threshold is the smaller of a ``1e-9`` relative test and OA's separator
    test (``1e-6 * min(1, |a|_inf)``, ``oa.collect_new_lazy_cuts``), so every
    point the separator would call violated is declined here too. With the
    relative test alone, a row of norm 1e-5 left a window -- violations between
    1e-11 and 1e-9 -- in which neither the decline nor the separator acted
    (review of #1673). Declining more is always safe: the point is offered again
    by the rebuilt tree, where the row is present.
    """
    a_norm = float(np.max(np.abs(a))) if a.size else 0.0
    tol = min(1e-9 * max(1.0, abs(r)), 1e-6 * min(1.0, a_norm))
    return float(np.dot(a, x)) > r + tol


def _prepare_cut_row(
    coeffs: np.ndarray,
    rhs: float,
    lb: np.ndarray,
    ub: np.ndarray,
    small_tol: float,
    large_tol: float,
    *,
    lift_to_unit: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Make ``coeffs @ x <= rhs`` safe to hand :meth:`highspy.Highs.addRow`.

    ``lift_to_unit`` (#1658) additionally lifts a row whose largest coefficient is
    below 1 into [1, 2) by a power of two. Only the LP/NLP-BB lazy rows ask for it
    (see the comment at the lift); the whole-model fitting that serves the OA,
    GOA and GDP masters leaves it off, so those masters are unchanged.

    HiGHS silently discards any matrix entry with ``|value| <= small_matrix_value``
    and reports the discard as ``kWarning`` -- neither an error nor a clean add.
    Both readings are wrong: treating it as a rejection killed every ``squfl``
    instance under ``lp_nlp_bb`` (#1066; ``squfl025-040`` died in 1.0 s), and
    treating it as a clean add would let HiGHS *change the cut* behind the
    separator's back. So the row is fitted to HiGHS's window here:

    1. **Scale.** An OA cut is an inequality, so multiplying it by any ``s > 0``
       leaves it mathematically identical. ``squfl025-040``'s first cut spans
       ``2.05e-19`` to ``114`` -- 21 orders -- but HiGHS's window is 24 wide, so the
       whole row fits once lifted. ``s`` is rounded down to a power of two, which
       makes the scaled row *bit-for-bit* the original one rescaled: no coefficient
       is perturbed, so there is no question of a rounding-tightened cut.
    2. **Drop what still will not fit**, on the valid side only: with ``rhs``
       untouched when ``a_j * x_j >= 0`` over the whole column box (removing a
       non-negative term can only make the row easier to satisfy), otherwise with
       ``rhs`` loosened by the most that term can contribute. Either way the result
       is *implied by* the original row -- every point the cut kept, it keeps.
    3. **Refuse loudly** when neither applies, rather than ship a row that is not
       implied by the cut.

    The unboundedness test is on the *bound* (``>= _INF``), never on the product:
    ``1e-9 * 1e20`` is an unremarkable ``1e11``, which is exactly how an open bound
    sneaks past a finiteness check (CLAUDE.md).
    """
    nz = np.flatnonzero(coeffs)
    if nz.size == 0:
        return nz, coeffs[nz], rhs

    a_abs = np.abs(coeffs[nz])
    a_max = float(np.max(a_abs))
    # The largest scale the big end tolerates, i.e. without bringing it within a
    # decade of the value HiGHS refuses. Scale UP only: scaling down would push
    # entries that fit today below the threshold.
    scale_cap = large_tol / (10.0 * a_max)
    # Only terms this scale can actually lift clear of the drop threshold get a
    # say in how far to scale. A term too small to be rescued is dropped below no
    # matter what, so letting it set the scale buys nothing and costs the whole
    # row its conditioning -- measured on ``squfl020-150`` (#1066): a perspective
    # row at reference ``z=1.2e-16`` spans 3e-32 (its ``q*z**2`` term) to 1, the
    # 3e-32 drove ``headroom`` to 46, and ``2**46 * 3e-32 = 2.1e-18`` was still
    # under ``small_matrix_value`` and dropped -- so the row was scaled by
    # 7.04e13 to rescue a term it discarded, and what reached the master was a
    # redundant ``s_k >= 0`` carrying a 7.04e13 coefficient.
    if a_max >= large_tol:
        # The big end is one HiGHS REFUSES (kError), not drops: scale the row DOWN,
        # by a power of two so it stays the same inequality bit-for-bit. Measured on
        # ``clay0303hfsg`` with rows scaled by up to 1e6: OA cut coefficients reached
        # 1.6e22 and the whole master was rejected. Whatever this pushes under
        # ``small_tol`` is handled below like any other tiny term -- dropped on its
        # valid side, or refused.
        scale = 2.0 ** math.floor(math.log2(scale_cap))
        coeffs = coeffs * scale
        rhs *= scale
    else:
        liftable = a_abs[a_abs * scale_cap > small_tol]
        # ``a_max`` itself always qualifies (``a_max * scale_cap`` is
        # ``large_tol/10``, far above ``small_tol``), so this is never empty.
        a_min = float(np.min(liftable))
        room = min(10.0 * small_tol / a_min, scale_cap)
        headroom = math.floor(math.log2(room))
        # #1658: a row whose largest coefficient is below 1 is also lifted until it
        # is in [1, 2). HiGHS judges a row against an ABSOLUTE feasibility
        # tolerance, so on a row with coefficients ~5.5e-5 (flay03m, rows scaled by
        # 10^U(-6,6)) a master point violating an OA cut by 1.2e-2 per unit of
        # coefficient read as feasible (6.6e-7 raw), and LP/NLP-BB stopped 3.3e-4
        # short of the optimum. Exact for the same reason as above, and capped by
        # the same ``scale_cap``.
        if lift_to_unit and a_max < 1.0:
            headroom = max(
                headroom,
                min(math.ceil(-math.log2(a_max)), math.floor(math.log2(scale_cap))),
            )
        if headroom > 0:
            scale = 2.0**headroom
            coeffs = coeffs * scale
            rhs *= scale

    tiny = (coeffs != 0.0) & (np.abs(coeffs) <= small_tol)
    if not tiny.any():
        nz = np.flatnonzero(coeffs)
        return nz, coeffs[nz], rhs

    a = coeffs[tiny]
    lo, hi = lb[tiny], ub[tiny]
    # ``a_j * x_j >= 0`` everywhere: a positive coefficient on a non-negative
    # column, or a negative one on a non-positive column. Those drop for free.
    free = np.where(a > 0.0, lo >= 0.0, hi <= 0.0)
    # Otherwise rhs must absorb the term's worst contribution, which needs the
    # bound on the binding side to be finite.
    binding_open = np.where(a > 0.0, lo <= -_INF, hi >= _INF)
    stuck = ~free & binding_open
    if stuck.any():
        bad = np.flatnonzero(tiny)[stuck][:5]
        raise ValueError(
            "cannot add this lazy cut to the HiGHS master: even rescaled it has "
            f"coefficients at or below HiGHS's small_matrix_value ({small_tol:g}) on "
            f"columns that are unbounded on the binding side (columns {bad.tolist()}), "
            "so HiGHS would drop those terms and no finite right-hand side can "
            "compensate. The cut would silently stop being valid. Bound those "
            "columns, or separate a cut that does not touch them."
        )
    absorb = ~free
    if absorb.any():
        aa, lo_a, hi_a = a[absorb], lo[absorb], hi[absorb]
        # One ulp up, because the loosening can be smaller than the ulp of ``rhs``
        # and round straight back off it -- which would leave the row a hair
        # TIGHTER than the one that is provably implied. Rounding the slack up is
        # always the safe direction: it can only weaken the cut.
        rhs = float(np.nextafter(rhs + float(np.sum(np.maximum(-aa * lo_a, -aa * hi_a))), np.inf))

    kept = coeffs.copy()
    kept[tiny] = 0.0
    nz = np.flatnonzero(kept)
    return nz, kept[nz], rhs


def _fit_rows_to_window(
    a: sp.csr_matrix,
    rhs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    small_tol: float,
    large_tol: float,
    equality: bool,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Apply :func:`_prepare_cut_row` to every row with an entry HiGHS would not take.

    ``passModel`` drops ``|a_ij| <= small_matrix_value`` and answers ``kWarning``,
    and refuses the model (``kError``) on ``|a_ij| >= large_matrix_value``, exactly
    as ``addRow`` does, so a whole master built from OA cuts needs the same
    fitting the lazy path already gets. Rows with no such entry are returned
    bit-for-bit, so every master that HiGHS accepted before is unchanged.

    An equality row may only be *scaled* (power of two: exact); it has no valid
    side to drop a term on, so one that still holds a tiny entry after scaling is
    refused rather than silently perturbed.
    """
    if a.nnz == 0:
        return a, rhs
    misfit_row = np.zeros(a.shape[0], dtype=bool)
    rows_of_nz = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    a_abs = np.abs(a.data)
    misfit_row[rows_of_nz[(a_abs <= small_tol) | (a_abs >= large_tol)]] = True
    if not misfit_row.any():
        return a, rhs
    a = a.tolil(copy=True)
    rhs = rhs.copy()
    for i in np.flatnonzero(misfit_row):
        dense = a.getrow(i).toarray().ravel()
        nnz_before = int(np.count_nonzero(dense))
        idx, vals, new_rhs = _prepare_cut_row(dense, float(rhs[i]), lb, ub, small_tol, large_tol)
        if equality and idx.size != nnz_before:
            raise ValueError(
                f"equality row {i} of the HiGHS master has coefficients at or below "
                f"HiGHS's small_matrix_value ({small_tol:g}) that no exact rescaling "
                "lifts clear of it; HiGHS would drop them and the row would change."
            )
        row = np.zeros(a.shape[1])
        row[idx] = vals
        a[i, :] = row
        rhs[i] = new_rhs
    return a.tocsr(), rhs


def _stack_rows(
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    n: int,
    highs_inf: float,
    window: Optional[tuple[np.ndarray, np.ndarray, float, float]] = None,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Stack ``A_ub x <= b_ub`` and ``A_eq x == b_eq`` into HiGHS's row-range form.

    ``window=(lb, ub, small_tol, large_tol)`` fits every row to HiGHS's coefficient
    window first (:func:`_fit_rows_to_window`); without it rows pass as given.
    """
    blocks, lowers, uppers = [], [], []
    if A_ub is not None and b_ub is not None:
        a = sp.csr_matrix(A_ub, dtype=np.float64)
        rhs = np.asarray(b_ub, dtype=np.float64).ravel()
        if a.shape[1] != n or a.shape[0] != rhs.shape[0]:
            raise ValueError(f"A_ub {a.shape} inconsistent with c ({n},) / b_ub {rhs.shape}")
        if window is not None:
            a, rhs = _fit_rows_to_window(a, rhs, *window, equality=False)
        blocks.append(a)
        lowers.append(np.full(a.shape[0], -highs_inf))
        uppers.append(rhs)
    if A_eq is not None and b_eq is not None:
        a = sp.csr_matrix(A_eq, dtype=np.float64)
        rhs = np.asarray(b_eq, dtype=np.float64).ravel()
        if a.shape[1] != n or a.shape[0] != rhs.shape[0]:
            raise ValueError(f"A_eq {a.shape} inconsistent with c ({n},) / b_eq {rhs.shape}")
        if window is not None:
            a, rhs = _fit_rows_to_window(a, rhs, *window, equality=True)
        blocks.append(a)
        lowers.append(rhs)
        uppers.append(rhs)
    if not blocks:
        return sp.csc_matrix((0, n), dtype=np.float64), np.zeros(0), np.zeros(0)
    return (
        sp.vstack(blocks, format="csc"),
        np.concatenate(lowers),
        np.concatenate(uppers),
    )


#: HiGHS terminal status -> discopt status. Anything absent is an ERROR: a status
#: this module has not reasoned about must not be silently read as a clean exit.
def _status_map(highspy) -> dict:
    ms = highspy.HighsModelStatus
    return {
        ms.kOptimal: SolveStatus.OPTIMAL,
        ms.kInfeasible: SolveStatus.INFEASIBLE,
        ms.kUnbounded: SolveStatus.UNBOUNDED,
        ms.kUnboundedOrInfeasible: SolveStatus.UNBOUNDED,
        ms.kTimeLimit: SolveStatus.TIME_LIMIT,
        ms.kIterationLimit: SolveStatus.ITERATION_LIMIT,
        ms.kObjectiveBound: SolveStatus.CUTOFF,
        ms.kObjectiveTarget: SolveStatus.CUTOFF,
        ms.kSolutionLimit: SolveStatus.ITERATION_LIMIT,
        ms.kInterrupt: SolveStatus.ITERATION_LIMIT,
    }


def solve_milp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[BoundList] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 1_000_000,
    mip_start: Optional[np.ndarray] = None,
    confirm_bound_from: Optional[float] = None,
) -> MILPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq`` with HiGHS.

    Signature-compatible with :func:`discopt.solvers.milp_simplex.solve_milp`, so
    it drops into ``get_milp_solver``'s contract. ``bound`` is HiGHS's dual bound,
    held against a second, presolve-free HiGHS solve (#1634,
    :func:`_cross_check_presolve`) before it is returned.

    ``confirm_bound_from`` (#1658) lets a caller that will only *certify* a bound
    at or above some level skip the cross-solve below it: when the primary is
    not ``infeasible`` and its bound is below ``confirm_bound_from``, the primary
    is returned unconfirmed, marked ``callback_stats["presolve_cross_check"]
    ["confirmed"] = False``. Such a bound is HiGHS's word only (the #1634 panel
    measured 1/1200 false bounds with presolve on), so the caller must not treat it
    as certified. ``None`` (the default) cross-checks every claim, as before.
    """
    t0 = time.time()
    kw: dict[str, Any] = dict(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=integrality,
        gap_tolerance=gap_tolerance,
        mip_start=mip_start,
    )
    primary = _solve_milp_once(time_limit=time_limit, presolve=True, **kw)
    if (
        confirm_bound_from is not None
        and primary.status != SolveStatus.INFEASIBLE
        and primary.bound is not None
        and float(primary.bound) < float(confirm_bound_from)
    ):
        primary.callback_stats = {
            **(primary.callback_stats or {}),
            "presolve_cross_check": {
                "ran": False,
                "confirmed": False,
                "primary_status": primary.status.value,
                "confirm_bound_from": float(confirm_bound_from),
            },
        }
        return primary
    res = _cross_check_presolve(primary, kw, time_limit, t0)
    diag = (res.callback_stats or {}).get("presolve_cross_check")
    if isinstance(diag, dict):
        diag["confirmed"] = True
    return res


def _solve_milp_once(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    bounds: Optional[BoundList],
    integrality: Optional[np.ndarray],
    time_limit: Optional[float],
    gap_tolerance: float,
    mip_start: Optional[np.ndarray],
    presolve: bool,
) -> MILPResult:
    """One HiGHS MILP solve; ``presolve=False`` is the #1634 cross-solve, not an option."""
    highspy = _require_highspy()
    t0 = time.time()

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    lb, ub = _marshal_col_bounds(bounds, n)
    h = highspy.Highs()
    small_tol, large_tol = _highs_matrix_window(h, highspy)
    a_csc, row_lower, row_upper = _stack_rows(
        A_ub, b_ub, A_eq, b_eq, n, highspy.kHighsInf, window=(lb, ub, small_tol, large_tol)
    )

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = a_csc.shape[0]
    lp.col_cost_ = c_arr
    lp.col_lower_ = _to_highs_inf(lb, highspy)
    lp.col_upper_ = _to_highs_inf(ub, highspy)
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = n
    lp.a_matrix_.num_row_ = a_csc.shape[0]
    lp.a_matrix_.start_ = a_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = a_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = a_csc.data
    if integrality is not None:
        int_mask = np.asarray(integrality).ravel().astype(bool)
        if int_mask.shape[0] != n:
            raise ValueError(f"integrality has {int_mask.shape[0]} entries but c has {n}")
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if f else highspy.HighsVarType.kContinuous
            for f in int_mask
        ]

    # Every option write is checked: a rejected option would silently leave the
    # solve on a different configuration than the one reported (CLAUDE.md §6/§7).
    opts: list[tuple[str, object]] = [
        ("output_flag", False),
        ("mip_rel_gap", float(gap_tolerance)),
        # #1634: HiGHS's parallel-column presolve fixes integers on an absolute
        # cost-tie test that bad coefficient ratios defeat; see the constant.
        ("presolve_rule_off", int(MILP_PRESOLVE_RULE_OFF)),
    ]
    if not presolve:
        opts.append(("presolve", "off"))
    if time_limit is not None:
        opts.append(("time_limit", float(time_limit)))
    for key, val in opts:
        st = h.setOptionValue(key, val)
        if st != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {key}={val!r} (status {st})")
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected the master model")

    if mip_start is not None:
        seed = np.asarray(mip_start, dtype=np.float64).ravel()
        if seed.shape[0] != n:
            raise ValueError(
                f"mip_start has {seed.shape[0]} entries but the master has {n} columns"
            )
        sol = highspy.HighsSolution()
        sol.col_value = seed
        # A rejected start is not fatal -- it is a warm-start hint, and HiGHS
        # validates it itself -- but it must not pass unnoticed either.
        h.setSolution(sol)

    h.run()
    wall = time.time() - t0

    model_status = h.getModelStatus()
    status = _status_map(highspy).get(model_status, SolveStatus.ERROR)
    info = h.getInfo()

    x = None
    objective = None
    if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
        x = np.asarray(h.getSolution().col_value, dtype=np.float64).ravel()[:n]
        objective = float(c_arr @ x)

    # SOUNDNESS: the dual bound comes from HiGHS and is never synthesized from the
    # incumbent. `mip_dual_bound` is +/-inf before the root LP finishes.
    bound = None
    raw_bound = float(info.mip_dual_bound)
    # #1614 D-26: with no integer column HiGHS runs its LP solver, never the MIP
    # one, and ``mip_dual_bound`` keeps a stale FINITE 0.0. Read as a bound it
    # capped an optimal LP's bound at 0 (a continuous Benders master then never
    # closed) and, on an LP stopped by a limit, would have claimed 0 as a lower
    # bound on an LP whose optimum is below it. An LP's bound is its optimum,
    # which the OPTIMAL arm below supplies; otherwise there is none.
    has_integer = integrality is not None and bool(np.any(np.asarray(integrality).ravel()))
    if has_integer and np.isfinite(raw_bound):
        bound = raw_bound
    if status == SolveStatus.OPTIMAL and objective is not None:
        # HiGHS may report a dual bound a hair above the incumbent at its own
        # gap tolerance; a bound is only a bound if it does not cross.
        bound = objective if bound is None else min(bound, objective)

    gap = None
    if np.isfinite(info.mip_gap):
        gap = float(info.mip_gap)

    return MILPResult(
        status=status,
        x=x,
        objective=objective,
        bound=bound,
        gap=gap,
        node_count=int(info.mip_node_count),
        iterations=int(info.simplex_iteration_count),
        wall_time=wall,
    )


def _verified_objective(kw: dict, x: Optional[np.ndarray]) -> Optional[float]:
    """``c @ x`` when ``x`` satisfies the ORIGINAL master (rows, bounds, integrality).

    #1634: the point is checked against the caller's arrays, not against the model
    HiGHS was handed (which ``_stack_rows`` may have windowed), so a verified point is
    a point of the master the caller asked about.
    """
    if x is None:
        return None
    c_arr = np.asarray(kw["c"], dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    if x.shape[0] != n or not np.all(np.isfinite(x)):
        return None
    lb, ub = _marshal_col_bounds(kw["bounds"], n)
    lo_ok = (lb <= -_INF) | (x >= lb - _VERIFY_REL * np.maximum(1.0, np.abs(lb)))
    hi_ok = (ub >= _INF) | (x <= ub + _VERIFY_REL * np.maximum(1.0, np.abs(ub)))
    if not (np.all(lo_ok) and np.all(hi_ok)):
        return None
    if kw["integrality"] is not None:
        mask = np.asarray(kw["integrality"]).ravel().astype(bool)
        if np.any(np.abs(x[mask] - np.round(x[mask])) > _VERIFY_INT):
            return None
    for key_a, key_b, eq in (("A_ub", "b_ub", False), ("A_eq", "b_eq", True)):
        a = kw[key_a]
        if a is None:
            continue
        a = sp.csr_matrix(a)
        if a.shape[0] == 0:
            continue
        rhs = np.asarray(kw[key_b], dtype=np.float64).ravel()
        act = a @ x
        scale = np.maximum(np.maximum(1.0, np.abs(rhs)), abs(a) @ np.abs(x))
        viol = np.abs(act - rhs) if eq else act - rhs
        if np.any(viol > _VERIFY_REL * scale):
            return None
    return float(c_arr @ x)


def _cross_check_presolve(
    primary: MILPResult, kw: dict, time_limit: Optional[float], t0: float
) -> MILPResult:
    """#1634: hold a HiGHS master's bound against a second, presolve-free HiGHS solve.

    An OA / GDP master's ``bound`` becomes the global lower bound, so a master solve
    that prunes its own optimum is a FALSE certificate on a convex MINLP. HiGHS's MIP
    presolve and tree decide on ABSOLUTE tolerances that a badly scaled master
    defeats. Rule 13 is off (:data:`~discopt.solvers.lp_milp_highs.MILP_PRESOLVE_RULE_OFF`),
    but measured on the
    #1634 slack panel through this backend that still leaves 1/1200 false bounds with
    presolve on, and 6/1200 with presolve off entirely; the two configurations never
    failed on the same instance. So:

    * the published bound is ``min`` of the two configurations' bounds -- valid
      whenever either one is -- and never above a VERIFIED point of either solve
      (:func:`_verified_objective`): a verified point below a bound refutes it;
    * the returned incumbent is the better verified point (the primary's own when
      neither verifies, as before);
    * ``infeasible`` stands only when the cross-solve agrees; a verified point
      refutes it;
    * a claim the cross-solve cannot confirm (no budget left, or no bound of its
      own) is withdrawn -- ``bound=None`` -- rather than standing on the absence of
      evidence (#1309).

    ``OPTIMAL`` is kept only when both solves were optimal and the published bound
    still closes the gap at ``gap_tolerance``; otherwise the result is an uncertified
    ``ITERATION_LIMIT`` carrying the (valid) bound. Diagnostics are in
    ``callback_stats["presolve_cross_check"]``.
    """
    infeasible = primary.status == SolveStatus.INFEASIBLE
    if not infeasible and primary.bound is None:
        return primary  # no claim to check
    diag: dict[str, object] = {"ran": False, "primary_status": primary.status.value}
    primary.callback_stats = {**(primary.callback_stats or {}), "presolve_cross_check": diag}

    def _withdraw(why: str) -> MILPResult:
        diag["withdrawn"] = why
        logger.warning("HiGHS master: claim withdrawn: %s (#1634)", why)
        primary.bound = None
        primary.gap = None
        if infeasible:
            primary.status, primary.x, primary.objective = SolveStatus.ERROR, None, None
        elif primary.status == SolveStatus.OPTIMAL:
            primary.status = SolveStatus.ITERATION_LIMIT
        return primary

    remaining = None
    if time_limit is not None:
        remaining = float(time_limit) - (time.time() - t0)
        if remaining <= 0.0:
            return _withdraw("no time budget left for the presolve-free cross-solve")
    cross = _solve_milp_once(time_limit=remaining, presolve=False, **kw)
    diag.update(ran=True, cross_status=cross.status.value, cross_bound=cross.bound)
    primary.node_count += cross.node_count
    primary.iterations += cross.iterations
    primary.wall_time = time.time() - t0

    points = []
    for r in (primary, cross):
        obj = _verified_objective(kw, r.x)
        if obj is not None:
            points.append((obj, r.x))
    best = min(points, key=lambda p: p[0]) if points else None
    diag["verified_points"] = len(points)

    if infeasible:
        if cross.status == SolveStatus.INFEASIBLE:
            return primary
        if best is not None:
            diag["refuted"] = f"infeasible, but a verified point of objective {best[0]:.12g}"
            logger.warning("HiGHS master: infeasible refuted by a presolve-free solve (#1634)")
            primary.status = SolveStatus.ITERATION_LIMIT
            primary.objective, primary.x = best[0], np.asarray(best[1], dtype=np.float64)
            primary.bound = primary.gap = None
            return primary
        return _withdraw(
            f"the presolve-free cross-solve did not confirm infeasible ({cross.status.value})"
        )

    if cross.bound is None:
        return _withdraw(f"the presolve-free cross-solve has no bound ({cross.status.value})")
    assert primary.bound is not None
    bound = min(float(primary.bound), float(cross.bound))
    if best is not None and best[0] < bound:
        diag["refuted"] = f"bound {bound:.12g} above a verified point's objective {best[0]:.12g}"
        bound = best[0]
    if bound < float(primary.bound):
        diag["bound_lowered"] = float(primary.bound) - bound
        # An ulp-level lowering is still applied, but at WARNING it fired on nearly
        # every master once the decomposition masters moved to HiGHS (#1614 D-26).
        material = float(primary.bound) - bound > 1e-9 * max(1.0, abs(bound))
        logger.log(
            logging.WARNING if material else logging.DEBUG,
            "HiGHS master: bound %.12g lowered to %.12g by the presolve-free cross-solve (#1634)",
            primary.bound,
            bound,
        )
    if best is not None:
        primary.objective, primary.x = best[0], np.asarray(best[1], dtype=np.float64)
    primary.bound = bound
    if primary.objective is not None:
        gap = max(0.0, primary.objective - bound)
        denom = max(abs(primary.objective), abs(bound), 1e-10)
        primary.gap = gap / denom
        closed = gap <= 1e-9 or gap / denom <= float(kw["gap_tolerance"])
    else:
        closed = False
    both_optimal = primary.status == SolveStatus.OPTIMAL and cross.status == SolveStatus.OPTIMAL
    if primary.status == SolveStatus.OPTIMAL and not (both_optimal and closed):
        primary.status = SolveStatus.ITERATION_LIMIT
    return primary


def _cross_check_lazy_master(
    h,
    highspy,
    status: SolveStatus,
    bound: Optional[float],
    time_left: Optional[float],
    objective: Optional[float] = None,
) -> tuple[SolveStatus, Optional[float], dict]:
    """#1634 for the lazy-cut master: re-solve its FINAL model with presolve off.

    The cross-solve's incumbent is not separated, so it is never returned as the
    master's point; only its verdict and bound are used, as in
    :func:`_cross_check_presolve`: ``min`` of the two bounds, ``infeasible`` only if
    both agree, and a claim the cross-solve cannot confirm is withdrawn.

    ``OPTIMAL`` survives a lowered bound when the gap against ``objective`` (the
    master's returned incumbent) is still closed at the master's ``mip_rel_gap``:
    two B&B runs stopping at different points inside that gap is not a refutation.
    """
    infeasible = status == SolveStatus.INFEASIBLE
    diag: dict[str, object] = {"ran": False, "primary_status": status.value}
    if not infeasible and bound is None:
        return status, bound, diag
    if time_left is not None and time_left <= 0.0:
        diag["withdrawn"] = "no time budget left for the presolve-free cross-solve"
        if infeasible:
            return SolveStatus.ERROR, None, diag
        if status == SolveStatus.OPTIMAL:
            status = SolveStatus.ITERATION_LIMIT
        return status, None, diag
    g = highspy.Highs()
    st, rel_gap = h.getOptionValue("mip_rel_gap")
    if st != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS would not report mip_rel_gap (status {st})")
    opts: list[tuple[str, object]] = [
        ("output_flag", False),
        ("presolve", "off"),
        ("presolve_rule_off", int(MILP_PRESOLVE_RULE_OFF)),
        ("mip_rel_gap", float(rel_gap)),
    ]
    if time_left is not None:
        opts.append(("time_limit", float(time_left)))
    for key, val in opts:
        if g.setOptionValue(key, val) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {key}={val!r}")
    if g.passModel(h.getLp()) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected the master model for the #1634 cross-solve")
    g.run()
    cstatus = _status_map(highspy).get(g.getModelStatus(), SolveStatus.ERROR)
    raw = float(g.getInfo().mip_dual_bound)
    cbound = raw if np.isfinite(raw) else None
    diag.update(ran=True, cross_status=cstatus.value, cross_bound=cbound)
    if infeasible:
        if cstatus == SolveStatus.INFEASIBLE:
            return status, bound, diag
        diag["withdrawn"] = f"cross-solve did not confirm infeasible ({cstatus.value})"
        return SolveStatus.ERROR, None, diag
    if cbound is None:
        diag["withdrawn"] = f"cross-solve has no bound ({cstatus.value})"
        if status == SolveStatus.OPTIMAL:
            status = SolveStatus.ITERATION_LIMIT
        return status, None, diag
    assert bound is not None
    if cbound < bound:
        diag["bound_lowered"] = bound - cbound
        logger.warning(
            "HiGHS lazy master: bound %.12g lowered to %.12g by the presolve-free "
            "cross-solve (#1634)",
            bound,
            cbound,
        )
        bound = cbound
        if status == SolveStatus.OPTIMAL and not _gap_closed(objective, bound, float(rel_gap)):
            status = SolveStatus.ITERATION_LIMIT
    return status, bound, diag


def _gap_closed(objective: Optional[float], bound: float, rel_gap: float) -> bool:
    """Whether ``bound`` still closes the gap to ``objective`` at ``rel_gap`` (#1634)."""
    if objective is None or not np.isfinite(objective):
        return False
    gap = max(0.0, float(objective) - float(bound))
    return gap <= 1e-9 or gap / max(abs(float(objective)), abs(float(bound)), 1e-10) <= rel_gap


def solve_milp_with_lazy_cuts(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[BoundList] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 1_000_000,
    lazy_callback=None,
    node_callback=None,
    terminate_callback=None,
    terminate_poll_s: float = 1.0,
    mip_start: Optional[np.ndarray] = None,
) -> MILPResult:
    """LP/NLP-BB master on HiGHS: separate at integer-feasible nodes, restart on a cut.

    Interface-compatible with
    :func:`discopt.solvers.milp_simplex.solve_milp_with_lazy_cuts`, so
    ``solve_lp_nlp_bb`` dispatches to it unchanged.

    **Why this is a restart loop and not a true single tree.** Quesada-Grossmann
    wants to *inject* a row from inside the tree. HiGHS 1.12 declares
    ``kCallbackMipDefineLazyConstraints`` but its callback *input* struct exposes
    only ``user_interrupt`` / ``setSolution`` / ``repairSolution`` — there is no
    field through which a row can be handed back, so row injection is genuinely
    unavailable and pretending otherwise would silently drop the caller's cuts.
    What HiGHS does give is ``kCallbackMipImprovingSolution``: a hook at every
    integer-feasible incumbent, which is exactly the QG separation trigger. So the
    tree is rebuilt whenever a cut is actually needed, and only then — strictly
    fewer restarts than multi-tree OA, which restarts at every master *optimum*
    whether or not a cut was required.

    That is affordable precisely because of the measurement that motivates this
    module: HiGHS solves the ``rsyn0840m`` master in 92 nodes / 0.42 s, so a
    handful of restarts costs a few seconds where the in-house driver had not
    finished one tree in 60 s.

    **Soundness.** The returned incumbent is only ever a point the separator
    *accepted*. A point that triggered a cut is a point the OA cuts exclude, so
    reporting HiGHS's own last incumbent could hand back a MINLP-infeasible
    solution on a time-limited exit; the accepted point is tracked separately for
    that reason. ``bound`` is HiGHS's dual bound on the master, which only ever
    gains valid cuts, so it stays a valid lower bound throughout.

    ``callback_stats["mipsol_calls"] == 0`` means the separator never ran — NOT
    that it accepted everything (CLAUDE.md §6).

    ``terminate_callback`` is consulted at two kinds of instant: **at each
    restart**, after a tree has finished and produced cuts; and **inside the
    tree**, through ``kCallbackMipInterrupt``. The snapshot is the running
    ``callback_stats`` plus ``context`` (``"restart"`` or ``"interrupt"``),
    ``elapsed`` and ``dual_bound`` — the master's current dual bound, ``None``
    when HiGHS has none yet. Returning true stops the search and sets
    ``callback_stats["terminated"]``.

    Returning the string ``"abandon"`` (#1658 review B2) stops it the same way and
    also says the caller is handing its remaining budget elsewhere: the #1634
    presolve-free cross-solve then gets no time, so the bound it would have
    confirmed is withdrawn (``callback_stats["abandoned"]``). Without it the
    cross-solve ran on the route's *full* budget after the #1066 guard had
    already handed over: on ``rsyn0815m03m`` it ran from 17.0 s to 26.2 s of a
    30 s limit, the fallback got 4.2 s, and the solve ended at 36.3 s. A
    convergence stop (plain ``True``) keeps the cross-solve, which is what
    confirms its certificate.

    The in-tree poll is what makes a progress budget honest: restarts alone are
    not a clock. On ``rsyn0820m02m`` the master separates rarely enough that a
    restart-only hook had nothing to judge at the checkpoint and abandoned a run
    that certifies 2 s later. But HiGHS fires that callback about **3000 times a
    second** (measured), which is neither affordable to answer in Python nor a
    sensible sampling rate for a trend, so it is answered at most once every
    ``terminate_poll_s`` seconds. ``last_poll`` is reset by a restart too, since
    a restart *is* a consultation -- so the two arms compose into one guarantee:
    the hook is consulted about every ``terminate_poll_s``, whatever the restart
    cadence. (Frequent restarts therefore mean the interrupt arm rarely fires,
    which is the hook being consulted MORE often than promised, not a dead clock.
    Measured in ``scripts/entry_poll_clock_cadence.py``.)

    Size a budget window for **~2x ``terminate_poll_s``, not 1x**: HiGHS offers
    the interrupt when it reaches one, not on a timer, so a gap is the interval
    plus the time to the next callback. Measured worst case over five arms at
    ``terminate_poll_s=1.0`` was 1.76 s -- a caller that read the interval as a
    bound and sized a 2 s window would get two samples where it planned for
    four.

    The hook can only ever give budget *back*: ``time_limit`` is still enforced
    through the HiGHS option on every run, so nothing here can overrun it.
    """
    highspy = _require_highspy()
    t0 = time.time()

    if lazy_callback is None:
        raise ValueError(
            "solve_milp_with_lazy_cuts requires lazy_callback; use solve_milp for a "
            "plain MILP solve"
        )
    if node_callback is not None:
        raise NotImplementedError(
            "the HiGHS lazy-cut backend has no MIPNODE equivalent: it separates only "
            "at integer-feasible incumbents, so node_callback (fractional user cuts) "
            "cannot be honoured. Use milp_solver='gurobi' for that."
        )

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    lb, ub = _marshal_col_bounds(bounds, n)
    h = highspy.Highs()
    small_tol, large_tol = _highs_matrix_window(h, highspy)
    a_csc, row_lower, row_upper = _stack_rows(
        A_ub, b_ub, A_eq, b_eq, n, highspy.kHighsInf, window=(lb, ub, small_tol, large_tol)
    )

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = a_csc.shape[0]
    lp.col_cost_ = c_arr
    lp.col_lower_ = _to_highs_inf(lb, highspy)
    lp.col_upper_ = _to_highs_inf(ub, highspy)
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = n
    lp.a_matrix_.num_row_ = a_csc.shape[0]
    lp.a_matrix_.start_ = a_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = a_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = a_csc.data
    if integrality is not None:
        int_mask = np.asarray(integrality).ravel().astype(bool)
        if int_mask.shape[0] != n:
            raise ValueError(f"integrality has {int_mask.shape[0]} entries but c has {n}")
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if f else highspy.HighsVarType.kContinuous
            for f in int_mask
        ]

    for key, val in [
        ("output_flag", False),
        ("mip_rel_gap", float(gap_tolerance)),
        ("presolve_rule_off", int(MILP_PRESOLVE_RULE_OFF)),  # #1634
    ]:
        if h.setOptionValue(key, val) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {key}={val!r}")
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected the master model")

    if mip_start is not None:
        seed = np.asarray(mip_start, dtype=np.float64).ravel()
        if seed.shape[0] != n:
            raise ValueError(
                f"mip_start has {seed.shape[0]} entries but the master has {n} columns"
            )
        sol = highspy.HighsSolution()
        sol.col_value = seed
        h.setSolution(sol)

    counts = {
        "mipsol_calls": 0,
        "mipnode_calls": 0,
        "lazy_cuts": 0,
        "node_cuts": 0,
        "restarts": 0,
        # #1658: improving solutions HiGHS offered from a stale tree that
        # violate a pending row, and so were not judged -- see ``_callback``.
        "stale_offers": 0,
        # #1658: final tree solutions no callback had judged, separated after the
        # tree finished -- see the loop below.
        "final_offers": 0,
        # How many times the hook was actually asked. Zero with a hook installed
        # is "it never got a look in", NOT "it kept saying continue" (§6).
        "terminate_polls": 0,
    }
    terminated = False
    terminate_context: Optional[str] = None
    stop = [False]
    last_poll = [t0]
    poll_interval = float(terminate_poll_s)
    if poll_interval < 0.0:
        raise ValueError(f"terminate_poll_s must be non-negative, got {terminate_poll_s!r}")
    pending: list[tuple[np.ndarray, float]] = []
    # The accepted incumbent is tracked separately from HiGHS's own: HiGHS's best
    # solution may be a point the separator VETOED, and returning that as the OA
    # incumbent would report an infeasible point as feasible (CLAUDE.md §1). Only
    # a point the separator declined to cut is eligible.
    best: list[Optional[np.ndarray]] = [None]
    best_obj: list[Optional[float]] = [None]

    abandoned = [False]

    def _consult(context: str, dual_bound, elapsed: float) -> bool:
        """Ask the caller's hook whether to stop. Never swallows (CLAUDE.md §7)."""
        snapshot: dict[str, object] = dict(counts)
        snapshot["context"] = context
        snapshot["elapsed"] = elapsed
        snapshot["dual_bound"] = dual_bound
        counts["terminate_polls"] += 1
        answer = terminate_callback(snapshot)
        if isinstance(answer, str) and answer == "abandon":
            if context != "final":
                abandoned[0] = True
            return True
        return bool(answer)

    def _callback(callback_type, message, data_out, data_in, user_data):
        if callback_type == highspy.cb.HighsCallbackType.kCallbackMipInterrupt:
            # HiGHS hands every callback the SAME input struct, so the flag the
            # separator sets to request a restart is still set when the rebuilt
            # tree's first interrupt arrives -- and interrupts it before it has
            # done anything. Measured: with this branch not writing the flag, the
            # toy model below dropped from 7 restarts / optimal 9.0 to 1 restart /
            # feasible 15.0. Every path through here states the flag it wants.
            if stop[0]:
                data_in.user_interrupt = True
                return
            # Fires thousands of times a second, so it is answered on an interval.
            now = time.time()
            if now - last_poll[0] < poll_interval:
                data_in.user_interrupt = False
                return
            last_poll[0] = now
            raw_lb = float(data_out.mip_dual_bound)
            stop[0] = _consult("interrupt", raw_lb if np.isfinite(raw_lb) else None, now - t0)
            data_in.user_interrupt = stop[0]
            return

        x = np.asarray(data_out.mip_solution, dtype=np.float64).ravel()[:n]
        counts["mipsol_calls"] += 1
        if pending and any(_violates_pending(a, r, x) for a, r in pending):
            # #1658: this tree is already stale. A cut was requested and HiGHS has
            # not honoured the interrupt yet, so it keeps offering improving
            # solutions of a model that lacks the pending rows. One that violates
            # a pending row must not be judged: a separator that reports only rows
            # it has not emitted before (OA's) finds nothing new and accepts it.
            # Measured on the #1658 n=20 big-M portfolio: a point with master
            # objective 0.0058 and true objective 0.0301 (optimum 0.0104) was
            # accepted, became HiGHS's incumbent, capped the master's dual bound
            # at 0.0058 and left LP/NLP-BB uncertified. An offer that satisfies
            # every pending row is judged as usual -- it is checked against the
            # whole row set the separator has emitted.
            counts["stale_offers"] += 1
            data_in.user_interrupt = True
            return
        # CLAUDE.md §7: a separator that raises must crash the solve, not be read
        # as "this point is fine" -- an accepted point becomes the OA incumbent.
        raw = lazy_callback(x)
        rows = (
            []
            if raw is None
            else [(np.asarray(a, dtype=np.float64).ravel(), float(r)) for a, r in raw]
        )
        if rows:
            pending.extend(rows)
            counts["lazy_cuts"] += len(rows)
            data_in.user_interrupt = True
        else:
            obj = float(c_arr @ x)
            if best_obj[0] is None or obj < best_obj[0]:
                best[0], best_obj[0] = x.copy(), obj

    h.setCallback(_callback, None)
    h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
    if terminate_callback is not None:
        h.startCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)

    status = SolveStatus.ERROR
    bound = None
    nodes = 0
    iters = 0
    while True:
        remaining = None
        if time_limit is not None:
            remaining = float(time_limit) - (time.time() - t0)
            if remaining <= 0.0:
                status = SolveStatus.TIME_LIMIT
                break
            if h.setOptionValue("time_limit", remaining) != highspy.HighsStatus.kOk:
                raise RuntimeError("HiGHS rejected the remaining time limit")
        pending.clear()
        h.run()

        info = h.getInfo()
        nodes += int(info.mip_node_count)
        iters += int(info.simplex_iteration_count)
        raw_bound = float(info.mip_dual_bound)
        if np.isfinite(raw_bound):
            bound = raw_bound if bound is None else max(bound, raw_bound)

        if stop[0]:
            # The hook interrupted the tree. HiGHS reports that as kInterrupt,
            # which is true but says nothing about who asked; record who.
            terminated = True
            terminate_context = "interrupt"
            status = _status_map(highspy).get(h.getModelStatus(), SolveStatus.ERROR)
            break

        if not pending:
            # #1658 (review of #1673): HiGHS does not route every solution through
            # kCallbackMipImprovingSolution -- one found in presolve or postsolve
            # is never offered. A tree can then finish "optimal" on a point the
            # separator never saw, and the loop ends with its bound stuck below the
            # incumbent: tls2 with rows scaled by 10^U(-6,6), under #1667's
            # presolve rules, finished optimal at 4.3 with the separator's only
            # accepted point at 5.3 (optimum 5.3). The final solution is therefore
            # separated here whenever it beats the best accepted point; cuts
            # rebuild the tree as usual, no cuts make it the accepted point.
            final_info = h.getInfo()
            if final_info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
                x_fin = np.asarray(h.getSolution().col_value, dtype=np.float64).ravel()[:n]
                obj_fin = float(c_arr @ x_fin)
                if best_obj[0] is None or obj_fin < best_obj[0] - 1e-9 * max(1.0, abs(best_obj[0])):
                    counts["final_offers"] += 1
                    raw_fin = lazy_callback(x_fin)
                    fin_rows = (
                        []
                        if raw_fin is None
                        else [
                            (np.asarray(a, dtype=np.float64).ravel(), float(r)) for a, r in raw_fin
                        ]
                    )
                    if fin_rows:
                        pending.extend(fin_rows)
                        counts["lazy_cuts"] += len(fin_rows)
                    else:
                        best[0], best_obj[0] = x_fin.copy(), obj_fin
        if not pending:
            status = _status_map(highspy).get(h.getModelStatus(), SolveStatus.ERROR)
            if terminate_callback is not None:
                # The tree finished on its own, so there is nothing left to
                # interrupt -- but this is the ONLY moment at which a master that
                # converges inside its FINAL tree can be observed at all. The
                # separator falls silent exactly when the incumbent becomes good,
                # so that convergence arrives with no restart left to carry a
                # check-in, and the hook never sees the certificate it exists to
                # detect. The answer is deliberately ignored: reporting "stopped
                # early" for a tree that ran to completion would be false.
                _consult("final", bound, time.time() - t0)
            break

        # A cut was requested, so this tree is stale: append the rows and rebuild.
        counts["restarts"] += 1
        if terminate_callback is not None:
            last_poll[0] = time.time()
            if _consult("restart", bound, last_poll[0] - t0):
                terminated = True
                terminate_context = "restart"
                status = _status_map(highspy).get(h.getModelStatus(), SolveStatus.ERROR)
                if status == SolveStatus.OPTIMAL:
                    # The tree that just finished was solved to optimality, but its
                    # rows are stale -- the separator vetoed its incumbent. Calling
                    # that "optimal" would hand the caller a certificate for a
                    # master that is missing the cut we are about to not add.
                    status = SolveStatus.TIME_LIMIT
                break
        for coeffs, rhs in pending:
            if coeffs.shape[0] != n:
                raise ValueError(f"lazy cut has {coeffs.shape[0]} coefficients, expected {n}")
            idx, vals, row_rhs = _prepare_cut_row(
                coeffs, float(rhs), lb, ub, small_tol, large_tol, lift_to_unit=True
            )
            if idx.size == 0:
                # Nothing survived, so there is no row to add. Adding nothing would
                # restart an identical tree forever on a point the separator keeps
                # vetoing, so refuse instead of spinning.
                raise ValueError(
                    "the separator returned a lazy cut with no coefficient HiGHS will "
                    f"accept (small_matrix_value {small_tol:g}); there is no row to "
                    "add and the master would not change."
                )
            st = h.addRow(-highspy.kHighsInf, row_rhs, int(idx.size), idx.astype(np.int32), vals)
            if st != highspy.HighsStatus.kOk:
                raise RuntimeError(
                    f"HiGHS rejected a lazy cut row (status {st}): {idx.size} nonzeros, "
                    f"|coef| in [{np.min(np.abs(vals)):.3g}, {np.max(np.abs(vals)):.3g}], "
                    f"rhs {row_rhs:.6g}"
                )

    h.stopCallback(highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution)
    if terminate_callback is not None:
        h.stopCallback(highspy.cb.HighsCallbackType.kCallbackMipInterrupt)
    h.clearCallbacks()

    x_out, obj_out = best[0], best_obj[0]
    if bound is not None and obj_out is not None and bound > obj_out:
        # Only possible at HiGHS's own gap tolerance; a bound that crosses the
        # incumbent is not a bound.
        bound = obj_out

    # #1634: the master's bound is the OA lower bound; hold it against a presolve-free
    # solve of the final master before it leaves this function.
    time_left = None if time_limit is None else float(time_limit) - (time.time() - t0)
    if abandoned[0]:
        # The caller stopped this search to spend its budget elsewhere (see the
        # docstring): the cross-solve gets none, and the unconfirmed bound is
        # withdrawn rather than published.
        time_left = 0.0
    status, bound, cross_diag = _cross_check_lazy_master(
        h, highspy, status, bound, time_left, objective=obj_out
    )

    return MILPResult(
        status=status,
        x=x_out,
        objective=obj_out,
        bound=bound,
        node_count=nodes,
        iterations=iters,
        wall_time=time.time() - t0,
        callback_stats={
            **counts,
            "terminated": terminated,
            "terminate_context": terminate_context,
            "abandoned": bool(abandoned[0]),
            "presolve_cross_check": cross_diag,
        },
    )
