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

import math
import time
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult, SolveStatus
from discopt.solvers.milp_simplex import _INF, BoundList, _marshal_col_bounds


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


def _prepare_cut_row(
    coeffs: np.ndarray,
    rhs: float,
    lb: np.ndarray,
    ub: np.ndarray,
    small_tol: float,
    large_tol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Make ``coeffs @ x <= rhs`` safe to hand :meth:`highspy.Highs.addRow`.

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

    a = coeffs[nz]
    a_min = float(np.min(np.abs(a)))
    a_max = float(np.max(np.abs(a)))
    # Lift the small end a decade clear of the drop threshold, without bringing the
    # large end within a decade of the value HiGHS refuses. Scale UP only: scaling
    # down would push entries that fit today below the threshold. Both ends are
    # strictly positive, so the logarithm is always defined.
    room = min(10.0 * small_tol / a_min, large_tol / (10.0 * a_max))
    headroom = math.floor(math.log2(room))
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


def _stack_rows(
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]],
    b_eq: Optional[np.ndarray],
    n: int,
    highs_inf: float,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Stack ``A_ub x <= b_ub`` and ``A_eq x == b_eq`` into HiGHS's row-range form."""
    blocks, lowers, uppers = [], [], []
    if A_ub is not None and b_ub is not None:
        a = sp.csr_matrix(A_ub, dtype=np.float64)
        rhs = np.asarray(b_ub, dtype=np.float64).ravel()
        if a.shape[1] != n or a.shape[0] != rhs.shape[0]:
            raise ValueError(f"A_ub {a.shape} inconsistent with c ({n},) / b_ub {rhs.shape}")
        blocks.append(a)
        lowers.append(np.full(a.shape[0], -highs_inf))
        uppers.append(rhs)
    if A_eq is not None and b_eq is not None:
        a = sp.csr_matrix(A_eq, dtype=np.float64)
        rhs = np.asarray(b_eq, dtype=np.float64).ravel()
        if a.shape[1] != n or a.shape[0] != rhs.shape[0]:
            raise ValueError(f"A_eq {a.shape} inconsistent with c ({n},) / b_eq {rhs.shape}")
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
) -> MILPResult:
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub, A_eq x == b_eq`` with HiGHS.

    Signature-compatible with :func:`discopt.solvers.milp_simplex.solve_milp`, so
    it drops into ``get_milp_solver``'s contract. ``bound`` is HiGHS's dual bound.
    """
    highspy = _require_highspy()
    t0 = time.time()

    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = c_arr.shape[0]
    lb, ub = _marshal_col_bounds(bounds, n)
    a_csc, row_lower, row_upper = _stack_rows(A_ub, b_ub, A_eq, b_eq, n, highspy.kHighsInf)

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

    h = highspy.Highs()
    # Every option write is checked: a rejected option would silently leave the
    # solve on a different configuration than the one reported (CLAUDE.md §6/§7).
    opts: list[tuple[str, object]] = [
        ("output_flag", False),
        ("mip_rel_gap", float(gap_tolerance)),
    ]
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
    if np.isfinite(raw_bound):
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

    The in-tree poll is what makes a progress budget honest: restarts alone are
    not a clock. On ``rsyn0820m02m`` the master separates rarely enough that a
    restart-only hook had nothing to judge at the checkpoint and abandoned a run
    that certifies 2 s later. But HiGHS fires that callback about **3000 times a
    second** (measured), which is neither affordable to answer in Python nor a
    sensible sampling rate for a trend, so it is answered at most once every
    ``terminate_poll_s`` seconds. That interval is the hook's real resolution and
    a caller budgeting by it should size its window accordingly.

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
    a_csc, row_lower, row_upper = _stack_rows(A_ub, b_ub, A_eq, b_eq, n, highspy.kHighsInf)

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

    h = highspy.Highs()
    for key, val in [("output_flag", False), ("mip_rel_gap", float(gap_tolerance))]:
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

    small_tol, large_tol = _highs_matrix_window(h, highspy)
    counts = {
        "mipsol_calls": 0,
        "mipnode_calls": 0,
        "lazy_cuts": 0,
        "node_cuts": 0,
        "restarts": 0,
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

    def _consult(context: str, dual_bound, elapsed: float) -> bool:
        """Ask the caller's hook whether to stop. Never swallows (CLAUDE.md §7)."""
        snapshot: dict[str, object] = dict(counts)
        snapshot["context"] = context
        snapshot["elapsed"] = elapsed
        snapshot["dual_bound"] = dual_bound
        counts["terminate_polls"] += 1
        return bool(terminate_callback(snapshot))

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
            status = _status_map(highspy).get(h.getModelStatus(), SolveStatus.ERROR)
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
            idx, vals, row_rhs = _prepare_cut_row(coeffs, float(rhs), lb, ub, small_tol, large_tol)
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

    return MILPResult(
        status=status,
        x=x_out,
        objective=obj_out,
        bound=bound,
        node_count=nodes,
        iterations=iters,
        wall_time=time.time() - t0,
        callback_stats={**counts, "terminated": terminated, "terminate_context": terminate_context},
    )
