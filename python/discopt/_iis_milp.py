"""Matrix-form IIS engine for linear models, with integrality as an element.

:func:`discopt.infeasibility.compute_iis` filters ``Constraint`` objects and
whole-variable bounds by re-solving the model, which cannot express the element
that dominates the size of a MIP's infeasible subsystem: the **integrality of a
single column**. (A discopt ``Variable`` carries one ``var_type`` for all its
elements.) For a linear model this module works on the extracted matrix instead,
with one persistent HiGHS model whose row bounds, column bounds and per-column
integrality are toggled in place for each probe.

Elements (Guieu & Chinneck 1999; the MIPcc27 competition uses the same set):

* a **constraint** (all rows of one ``model._constraints`` entry);
* the **integrality** of one integer or binary column;
* a finite **bound** of a column that is *not* held integer. The bounds of a
  column whose integrality is kept always apply and are not elements -- for a
  binary they *are* its domain. Once a column's integrality is relaxed its
  bounds become ordinary elements (a relaxed binary keeps ``[0, 1]`` only if
  those bounds are themselves kept).

Algorithm: deletion filtering (Chinneck & Dravnieks 1991) over the order
integrality -> constraints -> bounds. Before filtering integralities one at a
time, the LP relaxation is tested (Guieu & Chinneck's first step): if it is
already infeasible, every integrality is dropped in a single probe.

Soundness contract (same as ``compute_iis``): an element is dropped only on a
**proven** infeasible status; a probe that hits its time limit keeps the element
and clears ``proven_irreducible``. "Proven" means HiGHS returned ``Infeasible``
AND a presolve-free re-solve agreed -- the #1634 cross-check discopt's own HiGHS
MILP route requires before an infeasibility claim may stand.

Certificates come for free. A kept element ``e`` was kept because the active set
minus ``e`` was feasible, and the probe returned a point ``x``. Elements are only
ever removed afterwards, so ``x`` satisfies the *final* subsystem minus ``e`` as
well. Each such point is re-checked arithmetically against the final subsystem
(every other kept element within ``cert_tol``, ``e`` violated by more than
``cert_tol``) before it is reported; a point that fails the check is dropped,
never "repaired".
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model

_INF = 1e20  # discopt's unbounded-bound sentinel

#: A solve is not started with less wall time than this left in its budget.
_MIN_SOLVE_S = 0.05
#: Share of ``total_time_limit`` the presolve-free confirmation of the base set
#: (checkpoint 0) may use ...
_BASE_CONFIRM_FRACTION = 0.25
#: ... saturating at this ceiling (#1153): a bigger budget must buy filtering. The
#: slowest presolve-free base proof measured on MIPcc27's smallset is 107 s
#: (neos-1461051; dives 65 s, every other instance <= 12 s).
_BASE_CONFIRM_CAP_S = 300.0
#: Wall-clock grace past a run's own budget before the interrupt callback stops
#: it. HiGHS normally honours ``time_limit`` itself; this catches the overruns
#: measured on presolve-free re-solves (7-10x a 30 s limit on ``enlight4``).
_GRACE_S = 0.5


class _OutOfTime(Exception):
    """The overall budget leaves no time for another solve."""


@dataclass
class MilpIIS:
    """Result of :func:`milp_iis`, in flat-column / constraint-index terms."""

    constraints: list[int]  # indices into model._constraints
    bounds: list[tuple[int, str]]  # (flat column, "lower"|"upper")
    integrality: list[int]  # flat columns
    certificates: dict[tuple, np.ndarray] = field(default_factory=dict)
    n_solves: int = 0
    proven_irreducible: bool = True
    n_cross_checks: int = 0  # infeasibility claims re-solved with presolve off (#1634)
    n_refuted: int = 0  # ... that the presolve-free solve found feasible
    n_cross_inconclusive: int = 0  # ... that the presolve-free solve could not decide
    confirm: str = "final"  # where the #1634 confirmation ran: "each" probe or the "final" set
    fallback: bool = False  # "final" confirmation failed; the result is a full "each" rerun
    complete: bool = True  # False: the overall budget stopped filtering (anytime result)
    n_interrupted: int = 0  # solves stopped by the wall-clock guard
    n_repairs: int = 0  # "feasible" verdicts checked for an exact-integer witness
    n_artifacts: int = 0  # ... whose point did not survive rounding (verdict -> undecided)
    sat_rows: Optional[int] = None  # constraints in the SAT-core candidate (None: not used)
    sat_conflicts: int = 0  # CDCL conflicts spent building the candidate
    order: str = "ICB"  # phase order (and "-r": reversed element order) that produced it
    # Portfolio runs only: one entry per member, in portfolio order, with keys
    # order / outcome ("ok" | "timeout" | "abandoned") / size / complete / wall,
    # and a timed-out member's own TimeoutError message.
    members: Optional[list] = None


class _Engine:
    """One HiGHS model; elements are toggled by changing bounds/integrality."""

    def __init__(
        self,
        A,
        row_lo,
        row_hi,
        col_lb,
        col_ub,
        is_int,
        row_groups,
        probe_tl,
        confirm_each=True,
        deadline: Optional[float] = None,
        reserve_confirm: bool = False,
    ):
        import highspy

        self.hs = highspy
        self.A = A.tocsr()
        self.row_lo, self.row_hi = row_lo, row_hi
        self.col_lb, self.col_ub = col_lb, col_ub
        self.is_int = is_int
        self.row_groups = row_groups  # constraint index -> np.ndarray of row ids
        self.probe_tl = probe_tl
        self.n_solves = 0
        self.n_cross_checks = 0  # infeasibility claims re-solved with presolve off
        self.n_refuted = 0  # ... of which the presolve-free solve found feasible
        self.n_cross_inconclusive = 0  # ... of which the presolve-free solve was undecided
        self.confirm_each = confirm_each
        # ``time.perf_counter()`` instant after which no solve may run (None: no
        # overall budget), and the wall instant the current run must stop by.
        self.deadline = deadline
        # "final" mode: hold back time for confirming the current subsystem
        # (two solves), sized from recently observed solve times.
        self.reserve_confirm = reserve_confirm
        self._recent: deque[float] = deque(maxlen=20)  # wall seconds of recent solves
        self._stop = math.inf
        self.n_interrupted = 0
        self.n_repairs = 0
        self.n_artifacts = 0
        # Held-integer columns and their declared box in the current configuration.
        self._held = np.zeros(0, dtype=np.int64)
        self._cfg: tuple[np.ndarray, ...] = ()
        n, m = col_lb.size, row_lo.size
        inf = highspy.kHighsInf
        self._inf = inf
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("output_flag", False)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n, m
        lp.col_cost_ = np.zeros(n)
        lp.col_lower_ = np.where(col_lb <= -_INF, -inf, col_lb)
        lp.col_upper_ = np.where(col_ub >= _INF, inf, col_ub)
        lp.row_lower_ = np.where(row_lo <= -_INF, -inf, row_lo)
        lp.row_upper_ = np.where(row_hi >= _INF, inf, row_hi)
        csc = A.tocsc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = csc.data.astype(np.float64)
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous for b in is_int
        ]
        h.passModel(lp)
        self.h = h

        # Wall-clock guard on every run: HiGHS's own ``time_limit`` was measured
        # overrunning 7-10x on presolve-free re-solves, so a budget is enforced
        # here too. An interrupted run is a non-proof status (inconclusive).
        def _guard(event) -> None:
            # Set the flag both ways: HiGHS keeps a raised interrupt across runs.
            event.interrupt(time.perf_counter() > self._stop)

        h.cbMipInterrupt.subscribe(_guard)
        h.cbSimplexInterrupt.subscribe(_guard)
        h.cbIpmInterrupt.subscribe(_guard)

    def _configure(
        self,
        active_cons: set[int],
        active_bounds: set[tuple[int, str]],
        active_int: set[int],
    ) -> None:
        """Load one probe's rows, bounds and integrality into the HiGHS model."""
        h, inf = self.h, self._inf
        n, m = self.col_lb.size, self.row_lo.size
        rlo = np.full(m, -inf)
        rhi = np.full(m, inf)
        for ci in active_cons:
            rows = self.row_groups[ci]
            rlo[rows] = np.where(self.row_lo[rows] <= -_INF, -inf, self.row_lo[rows])
            rhi[rows] = np.where(self.row_hi[rows] >= _INF, inf, self.row_hi[rows])
        lb = np.full(n, -inf)
        ub = np.full(n, inf)
        integ = np.zeros(n, dtype=bool)
        for j in active_int:
            integ[j] = True
        # A column held integer keeps its declared bounds unconditionally.
        held = np.flatnonzero(integ)
        lb[held] = self.col_lb[held]
        ub[held] = self.col_ub[held]
        for j, side in active_bounds:
            if integ[j]:
                continue
            if side == "lower":
                lb[j] = self.col_lb[j]
            else:
                ub[j] = self.col_ub[j]
        lb = np.where(lb <= -_INF, -inf, lb)
        ub = np.where(ub >= _INF, inf, ub)
        self._held = held
        self._cfg = (rlo, rhi, lb, ub)  # what this probe imposes, for witness checks
        idx_m = np.arange(m, dtype=np.int32)
        idx_n = np.arange(n, dtype=np.int32)
        if m:
            h.changeRowsBounds(m, idx_m, rlo, rhi)
        if n:
            h.changeColsBounds(n, idx_n, lb, ub)
            vt = self.hs.HighsVarType
            h.changeColsIntegrality(
                n, idx_n, np.array([vt.kInteger if b else vt.kContinuous for b in integ])
            )

    def _solve(self, presolve: str, stop_at: Optional[float] = None):
        """Solve the configured probe from scratch; returns the HiGHS model status.

        The run's budget is the per-probe limit, cut to ``self.deadline`` and
        ``stop_at``; raises :class:`_OutOfTime` when that leaves nothing.
        """
        h = self.h
        now = time.perf_counter()
        stop = now + float(self.probe_tl)
        hard = self.deadline
        if hard is not None and self.reserve_confirm:
            # Two confirmation solves, each about as long as recent ones, with a
            # 25% margin. (Each may still overrun its own budget by up to
            # ``_GRACE_S`` before the guard stops it.)
            hard -= 2.5 * max(self._recent, default=0.0)
        for lim in (hard, stop_at):
            if lim is not None:
                stop = min(stop, lim)
        if stop - now < _MIN_SOLVE_S:
            raise _OutOfTime
        h.clearSolver()
        # HiGHS's run clock is cumulative over this object's lifetime and LP
        # solves compare it with ``time_limit``, so a relative budget made every
        # LP solve return "Time limit reached" (0 iterations) once the engine had
        # run longer than one budget. Offset by the clock; the wall guard below
        # enforces the real budget whichever clock HiGHS consults.
        h.setOptionValue("time_limit", h.getRunTime() + (stop - now))
        h.setOptionValue("presolve", presolve)
        self.n_solves += 1
        self._stop = stop + _GRACE_S
        try:
            h.run()
        finally:
            self._stop = math.inf
            self._recent.append(time.perf_counter() - now)
        status = h.getModelStatus()
        if status == self.hs.HighsModelStatus.kInterrupt:
            self.n_interrupted += 1
        return status

    def _point(self) -> np.ndarray:
        return np.array(self.h.getSolution().col_value, dtype=np.float64)

    def _witness(self, stop_at: Optional[float] = None) -> Optional[np.ndarray]:
        """An exact-integer feasible point behind the ``Optimal`` status just
        returned, or ``None`` if the point is a tolerance artifact.

        HiGHS accepts integers within ``mip_feasibility_tolerance``; with large
        coefficients that slack hides real row violations (``neos-3135526-osun``:
        |x_j - round(x_j)| ~ 1e-10 times coefficients up to 5e5 satisfied rows
        that are violated by 1.4e-5 once the integers are rounded). So a
        "feasible" verdict must survive repair: fix every held integer at its
        rounded value and re-solve the continuous part as an LP with presolve
        off. Its point has exact integers and is what the caller gets (and what
        becomes a certificate). Only integrality is made exact: rows are judged
        by the MIP verdict's own standard (``mip_feasibility_tolerance``), so the
        repair removes integrality slack and nothing else. A stricter row
        tolerance rejected ordinary points: at 1e-9 all 203 of ``dell``'s repairs
        (rows met to 9.4e-8), at the LP's 1e-7 seven more (rows met to 1.7e-7, a
        point HiGHS's MIP accepted). osun's artifacts fail at 1e-9, 1e-7 and 1e-6
        alike.
        """
        x = self._point()
        held = self._held
        if held.size == 0:
            return x
        self.n_repairs += 1
        h, vt = self.h, self.hs.HighsVarType
        xr = np.round(x[held])
        mip_tol = h.getOptionValue("mip_feasibility_tolerance")
        mip_tol = mip_tol[1] if isinstance(mip_tol, tuple) else mip_tol
        # Rounding often breaks nothing: then the rounded point itself is the
        # exact-integer witness, checked here by arithmetic. (The repair LP can
        # misjudge exactly this case: on dell an integral point meeting its rows
        # to 1.7e-7 had its integers-fixed LP declared Infeasible even at 1e-5,
        # presumably through HiGHS's internal scaling.)
        y0 = x.copy()
        y0[held] = xr
        rlo, rhi, lb, ub = self._cfg
        act = self.A @ y0
        with np.errstate(invalid="ignore"):
            viol = max(
                float(np.max(np.where(np.isfinite(rlo), rlo - act, 0.0), initial=0.0)),
                float(np.max(np.where(np.isfinite(rhi), act - rhi, 0.0), initial=0.0)),
                float(np.max(np.where(np.isfinite(lb), lb - y0, 0.0), initial=0.0)),
                float(np.max(np.where(np.isfinite(ub), y0 - ub, 0.0), initial=0.0)),
            )
        if viol <= mip_tol:
            return y0
        idx = held.astype(np.int32)
        h.changeColsBounds(idx.size, idx, xr, xr)
        h.changeColsIntegrality(idx.size, idx, np.array([vt.kContinuous] * idx.size))
        lp_tol = h.getOptionValue("primal_feasibility_tolerance")
        # (status, value) tuples in some highspy versions
        lp_tol = lp_tol[1] if isinstance(lp_tol, tuple) else lp_tol
        h.setOptionValue("primal_feasibility_tolerance", mip_tol)
        try:
            ok = self._solve("off", stop_at) == self.hs.HighsModelStatus.kOptimal
            y = self._point() if ok else None
        finally:
            h.setOptionValue("primal_feasibility_tolerance", lp_tol)
            lo = np.where(self.col_lb[held] <= -_INF, -self._inf, self.col_lb[held])
            hi = np.where(self.col_ub[held] >= _INF, self._inf, self.col_ub[held])
            h.changeColsBounds(idx.size, idx, lo, hi)
            h.changeColsIntegrality(idx.size, idx, np.array([vt.kInteger] * idx.size))
        if y is None:
            self.n_artifacts += 1
            return None
        y[held] = xr  # exact integers, not the LP's representation of them
        return y

    def probe(
        self,
        active_cons: set[int],
        active_bounds: set[tuple[int, str]],
        active_int: set[int],
    ) -> tuple[Optional[bool], Optional[np.ndarray]]:
        """``(True, None)`` infeasible; ``(False, x)`` feasible at ``x``;
        ``(None, None)`` inconclusive (time limit or another non-proof status).

        With ``confirm_each`` (the #1634 policy, applied per probe) an
        ``Infeasible`` verdict stands only when a presolve-free re-solve agrees.
        Without it a presolve-on ``Infeasible`` is acted on directly and the
        caller MUST confirm the final subsystem with :meth:`confirm_infeasible`.
        """
        self._configure(active_cons, active_bounds, active_int)
        MS = self.hs.HighsModelStatus
        ms = self._solve("choose")
        if ms == MS.kOptimal:
            x = self._witness()
            return (False, x) if x is not None else (None, None)
        if ms not in (MS.kInfeasible, MS.kUnboundedOrInfeasible):
            return None, None
        if ms == MS.kInfeasible and not self.confirm_each:
            return True, None
        # An infeasibility claim is what lets the filter DROP an element, so it
        # stands only when a presolve-free solve agrees -- the #1634 policy of
        # discopt's own HiGHS MILP route, where presolve-on and presolve-off each
        # produced false claims on a 1200-instance panel but never on the same
        # instance. (``kUnboundedOrInfeasible`` cannot be "unbounded" under a zero
        # objective; the presolve-free solve decides it, in either mode.)
        self.n_cross_checks += 1
        ms2 = self._solve("off")
        if ms2 == MS.kInfeasible:
            return True, None
        if ms2 == MS.kOptimal:
            x = self._witness()
            if x is not None:
                if ms == MS.kInfeasible:
                    self.n_refuted += 1
                return False, x
        # Undecided: presolve-off neither agreed nor produced an exact-integer
        # witness against the presolve-on claim. Never acted on.
        self.n_cross_inconclusive += 1
        return None, None

    def confirm_infeasible(
        self,
        active_cons: set[int],
        active_bounds: set[tuple[int, str]],
        active_int: set[int],
        stop_at: Optional[float] = None,
        presolve: tuple[str, ...] = ("choose", "off"),
    ) -> Optional[bool]:
        """``True`` iff presolve-on AND presolve-off both return ``Infeasible``;
        ``False`` if either finds an exact-integer feasible point (an earlier
        verdict was wrong); ``None`` if undecided (time limit, interrupt, or a
        "feasible" status whose point is a tolerance artifact). ``presolve``
        restricts the check to configurations not already proven for this set."""
        self._configure(active_cons, active_bounds, active_int)
        MS = self.hs.HighsModelStatus
        all_infeasible = True
        for p in presolve:
            st = self._solve(p, stop_at)
            if st == MS.kOptimal and self._witness(stop_at) is not None:
                return False
            all_infeasible &= st == MS.kInfeasible
        return True if all_infeasible else None


def _check_certificates(pending: dict, eng: _Engine, cons, bounds, ints, tol) -> dict:
    """The points in ``pending`` that certify their element: ``x`` satisfies the
    final subsystem minus ``key`` (every other row, applied bound and integrality
    within ``tol``) and violates ``key`` by more than ``tol``.

    The final set's masks are built once and each point is checked with one
    sparse product and vector operations. Rebuilding the set per point cost
    O(|IIS|) Python work per certificate: on dijle's 6,282-element anytime set
    that ran past the deadline and a portfolio member's result was discarded.
    """
    m, n = eng.row_lo.size, eng.col_lb.size
    row_kept = np.zeros(m, dtype=bool)
    for c in cons:
        row_kept[eng.row_groups[c]] = True
    held = np.zeros(n, dtype=bool)
    held[list(ints)] = True
    # A held column's box applies; so do the kept bounds of the other columns.
    lbm = held.copy()
    ubm = held.copy()
    for j, side in bounds:
        (lbm if side == "lower" else ubm)[j] = True
    lbm &= eng.col_lb > -_INF
    ubm &= eng.col_ub < _INF
    lo_f = eng.row_lo > -_INF
    hi_f = eng.row_hi < _INF
    out = {}
    for key, x in pending.items():
        act = eng.A @ x
        rv = np.maximum(
            np.where(lo_f, eng.row_lo - act, 0.0), np.where(hi_f, act - eng.row_hi, 0.0)
        )
        rv = np.maximum(rv, 0.0)
        lbv = np.where(lbm, eng.col_lb - x, 0.0)
        ubv = np.where(ubm, x - eng.col_ub, 0.0)
        iv = np.where(held, np.abs(x - np.round(x)), 0.0)
        kind = key[0]
        rows_ok = row_kept
        if kind == "constraint":
            own = eng.row_groups[key[1]]
            rows_ok = row_kept.copy()
            rows_ok[own] = False
            e = float(np.max(rv[own], initial=0.0))
        elif kind == "integrality":
            j = key[1]
            iv[j] = 0.0  # its box still applies (held -> lbm/ubm), as when relaxed
            e = abs(float(x[j] - np.round(x[j])))
        else:
            j, side = key[1], key[2]
            if side == "lower":
                lbv[j] = 0.0
                e = float(eng.col_lb[j] - x[j])
            else:
                ubv[j] = 0.0
                e = float(x[j] - eng.col_ub[j])
        rest = max(
            float(np.max(np.where(rows_ok, rv, 0.0), initial=0.0)),
            float(np.max(lbv, initial=0.0)),
            float(np.max(ubv, initial=0.0)),
            float(np.max(iv, initial=0.0)),
        )
        if rest <= tol and e > tol:
            out[key] = x
    return out


#: CDCL conflict budgets for the clause-core candidate: the first UNSAT proof and
#: each core-refinement call. Where the candidate pays off (smallset's bf*, ssa,
#: jnh, aim) the first proof took 18-50 conflicts and a whole refinement ~200;
#: pigeonhole (hole9/hole10), where it cannot pay off (the core is every row),
#: needs ~70k. 10k is 200x the largest measured need and gives up on pigeonhole
#: in ~2 s instead of spending the stage's whole time share.
_SAT_FIRST_CONFLICTS = 10_000
_SAT_REFINE_CONFLICTS = 2_000
#: Conflicts per solver call between wall-clock checks. A conflict budget alone
#: does not bound time: PHP(10,9) ran > 280 s inside 200k conflicts, and on
#: smallset's hole9/hole10 the candidate stage overran a 60 s total_time_limit
#: until the process was killed. Learnt clauses persist across calls, so slicing
#: loses no search progress. 1000 was too coarse once PB rows are encoded: on a
#: 4x36 market-split model one slice outlasted a 0.2 s stage budget by 0.6 s
#: and took the base proof's time; at 100 the overshoot is ~35 ms (50: no better).
_SAT_SLICE_CONFLICTS = 100
#: Wall-clock ceiling of the candidate stage; with an overall budget it gets
#: ``_SAT_STAGE_FRACTION`` of it, never more than the ceiling (#1153).
_SAT_STAGE_CAP_S = 30.0
_SAT_STAGE_FRACTION = 0.10
#: Encoding-size ceiling: past it the stage proposes nothing. The largest
#: measured on the smallset is ~103k clauses (jjliu_clique).
_SAT_MAX_CLAUSES = 2_000_000


def _pb_rows(cols, vals, lo, hi) -> Optional[list[tuple[list[int], list[int], int]]]:
    """A linear row over binaries as ``sum w_i * l_i >= K`` constraints.

    ``w_i`` are positive integers and ``l_i`` DIMACS literals of column ``c``
    (``c + 1``, negated for a negative coefficient: ``a x = |a| (1 - x) - |a|``).
    Each finite side gives one constraint; the lhs is integral, so a fractional
    ``K`` rounds up. ``None`` when a coefficient is not integral.
    """
    if not bool(np.all(np.abs(vals - np.round(vals)) <= 1e-9)):
        return None
    out = []
    for sign, side in ((1, lo), (-1, hi)):
        if (sign == 1 and side <= -_INF) or (sign == -1 and side >= _INF):
            continue
        a = sign * np.round(vals).astype(np.int64)
        k = sign * float(side)
        K = int(round(k)) if abs(k - round(k)) <= 1e-9 else int(math.ceil(k))
        w: list[int] = []
        lits: list[int] = []
        for c, ai in zip(cols, a):
            if ai > 0:
                w.append(int(ai))
                lits.append(int(c) + 1)
            elif ai < 0:
                w.append(int(-ai))
                lits.append(-(int(c) + 1))
                K += int(-ai)
        out.append((w, lits, K))
    return out


def _encode_geq(w: list[int], lits: list[int], K: int, guard: int, nv: int, clauses: list):
    """Clauses for ``guard -> sum w_i l_i >= K`` as a reduced BDD (Een & Sorensson
    2006): node ``(i, k)`` means "the suffix from ``i`` reaches ``k``", with
    ``node -> ite(l_i, (i+1, k - w_i), (i+1, k))``; the constraint is monotone, so
    also ``node -> (i+1, k - w_i)``. ``k <= 0`` is true, ``k`` above the suffix's
    total is false. Built layer by layer, without recursion. Returns the new
    variable count.
    """
    order = sorted(range(len(w)), key=lambda i: -w[i])  # large weights first
    w = [w[i] for i in order]
    lits = [lits[i] for i in order]
    m = len(w)
    suf = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suf[i] = suf[i + 1] + w[i]
    if K <= 0:
        return nv
    if K > suf[0]:
        clauses.append([-guard])
        return nv
    layers: list[set[int]] = [{K}]
    for i in range(m):
        nxt = set()
        for k in layers[i]:
            for k2 in (k - w[i], k):
                if 0 < k2 <= suf[i + 1]:
                    nxt.add(k2)
        layers.append(nxt)
    var: dict[tuple[int, int], int] = {}
    for i in range(m - 1, -1, -1):
        for k in layers[i]:
            nv += 1
            v = var[(i, k)] = nv
            kh, kl = k - w[i], k
            hi = True if kh <= 0 else var[(i + 1, kh)] if kh <= suf[i + 1] else False
            lo = True if kl <= 0 else var[(i + 1, kl)] if kl <= suf[i + 1] else False
            if hi is False:
                clauses.append([-v])
            elif hi is not True:
                clauses.append([-v, hi])
            if lo is False:
                clauses.append([-v, lits[i]])
            elif lo is not True:
                clauses.append([-v, lits[i], lo])
    clauses.append([-guard, var[(0, K)]])
    return nv


def _binary_core(A, row_lo, row_hi, col_lb, col_ub, is_int, row_groups, stop_at):
    """A small unsatisfiable set of constraints of a pure-binary model.

    Applies when every column is binary on [0, 1] and every row has integral
    coefficients; otherwise ``None``. Each row becomes CNF guarded by its
    constraint's selector literal: one BDD per side (:func:`_encode_geq`), which
    is a single clause for a clause row and a sequential counter for a
    cardinality row. One CDCL solve under all selectors gives a core, which is
    shrunk by deletion with core refinement (each UNSAT trial replaces the
    remaining set by its own core). A budget-limited call keeps its element, so
    the result is always an unsatisfiable set; it is a *proposal* -- the caller
    still proves it infeasible with HiGHS before using it.

    Returns ``(constraint indices or None, conflicts spent)``. Measured before
    building (MIPcc27 smallset, 60 s): on clause models the engine returns IISs
    at or below Gurobi's reference size in 4-21 s (bf2670-001: 399 elements /
    1,913 s -> 145 / 5.7 s); with cardinality/PB rows, neos-4344877-stour
    1005 (incomplete) -> 126 in 2.2 s, neos-807454 TimeoutError -> 65,
    normalized-3_4_13 1639 (incomplete) -> 254.
    """
    n = col_lb.size
    if n == 0 or not bool(np.all(is_int & (col_lb == 0.0) & (col_ub == 1.0))):
        return None, 0
    groups = sorted(row_groups)
    nv = n + len(groups)
    clauses: list[list[int]] = []
    for gi, ci in enumerate(groups):
        for r in row_groups[ci]:
            s0, e0 = A.indptr[r], A.indptr[r + 1]
            rows = _pb_rows(A.indices[s0:e0], A.data[s0:e0], row_lo[r], row_hi[r])
            if rows is None:
                return None, 0
            for w, lits, K in rows:
                nv = _encode_geq(w, lits, K, n + 1 + gi, nv, clauses)
                if len(clauses) > _SAT_MAX_CLAUSES:
                    return None, 0
    from discopt._rust import PySatSolver

    sat = PySatSolver(nv)
    sel = [n + 1 + gi for gi in range(len(groups))]
    for c in clauses:
        sat.add_clause(c)

    def out_of_time() -> bool:
        return stop_at is not None and time.perf_counter() >= stop_at

    def solve_sliced(assumptions: list[int], conflicts: int) -> Optional[bool]:
        """``sat.solve`` in slices, stopping at ``stop_at``: None when undecided."""
        spent0 = sat.stats()[0]
        while True:
            verdict: Optional[bool] = sat.solve(assumptions, _SAT_SLICE_CONFLICTS)
            if verdict is not None:
                return verdict
            if out_of_time() or sat.stats()[0] - spent0 >= conflicts:
                return None

    if solve_sliced(sel, _SAT_FIRST_CONFLICTS) is not False:
        return None, sat.stats()[0]  # SAT, or no proof in budget/time: no candidate
    rem = {abs(lit) - (n + 1) for lit in sat.core()}
    kept: set[int] = set()
    for gi in sorted(rem):
        if out_of_time():
            break
        if gi not in rem:
            continue
        trial = rem - {gi}
        verdict = solve_sliced([sel[j] for j in trial | kept], _SAT_REFINE_CONFLICTS)
        if verdict is False:
            core = {abs(lit) - (n + 1) for lit in sat.core()}
            kept &= core
            rem = core - kept
        else:  # needed (SAT) or undecided within budget: keep it
            kept.add(gi)
            rem.discard(gi)
    return {groups[gi] for gi in rem | kept}, sat.stats()[0]


def milp_iis(
    model: "Model",
    *,
    include_bounds: bool = True,
    time_limit: float = 30.0,
    cert_tol: float = 1e-6,
    max_batch: Optional[int] = None,
    confirm: str = "final",
    total_time_limit: Optional[float] = None,
    sat_candidate: bool = True,
    order: str = "ICB",
    portfolio: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
) -> MilpIIS:
    """IIS of a linear (LP/MILP) ``model`` with integrality as an element.

    ``max_batch`` caps the adaptive deletion batch (see :func:`_batch_filter`);
    ``None`` lets it grow without limit, ``1`` is the classic one-at-a-time
    deletion filter (kept for A/B measurement and as the reference behaviour).

    ``confirm`` places the #1634 presolve-free confirmation. ``"final"`` (the
    default) acts on the presolve-on verdict and then requires the FINISHED
    subsystem to be ``Infeasible`` under both presolve-on and presolve-off; if
    it is not, the whole computation is redone with ``"each"``
    (``fallback=True``). ``"each"`` re-solves every infeasibility verdict before
    acting on it. Either way the returned subsystem's infeasibility is agreed by
    both configurations -- the claim #1634 protects -- so the two modes differ
    only in when a wrong intermediate verdict is caught.

    ``"each"`` is kept as a documented strict opt-out, not a stalled
    alternative. It is not the default because on MIPcc27's smallset
    (interleaved A/B, 300 s/instance) ``"final"`` produced 0/32 invalid outputs,
    identical sizes on all 27 instances both modes solved, a median wall ratio
    of 0.59, 5 more instances solved and 0 lost, with 0 fallbacks. The two
    configurations DO disagree on ill-conditioned models: on
    ``neos-3135526-osun`` (timed out in both arms of that A/B, so not counted
    there; coefficients up to 5e5) presolve-on returned ``Infeasible`` where
    presolve-off returned a point satisfying the set to 2e-9 -- but only through
    integrality slack: rounding its near-integer columns leaves a row violated
    by 1.4e-5. Which verdict is right is tolerance-dependent, not established.
    The final check saw the disagreement and fell back, which never acts on it
    either way. What would change the default:
    fallbacks becoming common on some model class (each costs a full ``"each"``
    rerun), which ``MilpIIS.fallback`` / ``IISResult.confirm_fallback`` report.
    Each kept member's necessity rests on its arithmetic certificate where one
    exists; a member whose probe point fails the arithmetic re-check has none,
    and its necessity rests on HiGHS's feasible verdict for that probe (see
    ``IISResult.fully_certified``).

    ``total_time_limit`` (seconds) bounds the whole computation. Every solve is
    cut to the time left and stopped by a wall-clock interrupt if HiGHS overruns
    it. When the budget runs out filtering stops and an *anytime* result is
    returned (``complete=False``, ``proven_irreducible=False``): a subsystem that
    is still proven infeasible under both configurations but may contain
    undecided members. With ``"final"`` the filter stops early enough
    (two recent solve durations) to confirm it, and the latest confirmed
    subsystem is kept as a fallback: the base set, confirmed presolve-free right
    after the base proof (one extra solve, capped at ``_BASE_CONFIRM_FRACTION``
    of the budget, at most ``_BASE_CONFIRM_CAP_S``), then the set at the end of
    each completed phase. If no subsystem can be confirmed in time,
    :class:`TimeoutError` is raised rather than an unconfirmed claim returned.

    ``sat_candidate``: on a pure-binary model with integral coefficients, start
    from a small unsatisfiable constraint set found by discopt's CDCL solver
    (:func:`_binary_core`). It is only a proposal: it is used if HiGHS proves it
    infeasible (the base probe), otherwise the full model is used, and the
    returned subsystem is confirmed and certified exactly as without it.

    Raises ``NotImplementedError`` (from the linear extractor) for a model with a
    nonlinear or non-algebraic constraint, ``ValueError`` when the full model is
    not proven infeasible, and ``TimeoutError`` as above.
    """
    t_start = time.perf_counter()
    import scipy.sparse as sp

    from discopt.decomposition._linear import extract_linear
    from discopt.modeling.core import VarType

    lin = extract_linear(model)
    A = sp.csr_matrix(lin.A)
    b = np.asarray(lin.b, dtype=np.float64)
    m = lin.m
    row_lo = np.full(m, -_INF)
    row_hi = np.full(m, _INF)
    for r, s in enumerate(lin.sense):
        if s in ("<=", "=="):
            row_hi[r] = b[r]
        if s in (">=", "=="):
            row_lo[r] = b[r]
    rows_of: dict[int, list[int]] = {}
    for r, src in enumerate(lin.row_source):
        rows_of.setdefault(src, []).append(r)
    row_groups = {k: np.asarray(v, dtype=np.int64) for k, v in rows_of.items()}

    lbs, ubs, ints = [], [], []
    for v in model._variables:
        lbs.append(np.broadcast_to(np.asarray(v.lb, dtype=np.float64), v.shape).reshape(-1))
        ubs.append(np.broadcast_to(np.asarray(v.ub, dtype=np.float64), v.shape).reshape(-1))
        ints.append(np.full(v.size, v.var_type in (VarType.INTEGER, VarType.BINARY)))
    col_lb = np.concatenate(lbs) if lbs else np.zeros(0)
    col_ub = np.concatenate(ubs) if ubs else np.zeros(0)
    is_int = np.concatenate(ints) if ints else np.zeros(0, dtype=bool)
    if col_lb.size != lin.n:
        raise AssertionError(f"column count mismatch: {col_lb.size} vs extractor {lin.n}")
    arrays = (A, row_lo, row_hi, row_groups, col_lb, col_ub, is_int)
    kwargs: dict[str, Any] = dict(
        include_bounds=include_bounds,
        time_limit=time_limit,
        cert_tol=cert_tol,
        max_batch=max_batch,
        confirm=confirm,
        sat_candidate=sat_candidate,
    )
    if portfolio is not None:
        return _portfolio_iis(arrays, kwargs, portfolio, workers, t_start, total_time_limit)
    return _milp_iis_arrays(
        arrays, **kwargs, order=order, total_time_limit=total_time_limit, t_start=t_start
    )


def _parse_order(order: str) -> tuple[str, bool, bool]:
    """``"ICB"`` -> (``"ICB"``, reverse=False, sat=True).

    A permutation of ``"ICB"`` (phase order), optionally followed by ``"-"`` and
    flags: ``r`` visits elements in reverse, ``n`` skips the SAT candidate. E.g.
    ``"CIB-r"``, ``"ICB-n"``, ``"ICB-rn"``.
    """
    phases, _, flags = order.partition("-")
    if (
        sorted(phases) != ["B", "C", "I"]
        or not set(flags) <= {"r", "n"}
        or len(set(flags)) != len(flags)
        or ("-" in order and not flags)
    ):
        raise ValueError(
            "order must be a permutation of 'ICB' with optional '-' flags "
            f"('r' reverse, 'n' no SAT candidate), got {order!r}"
        )
    return phases, "r" in flags, "n" not in flags


def _milp_iis_arrays(
    arrays: tuple,
    *,
    include_bounds: bool,
    time_limit: float,
    cert_tol: float,
    max_batch: Optional[int],
    confirm: str,
    sat_candidate: bool,
    order: str,
    total_time_limit: Optional[float],
    t_start: Optional[float] = None,
) -> MilpIIS:
    """:func:`milp_iis` on the extracted matrix, with one deletion order."""
    if t_start is None:
        t_start = time.perf_counter()
    deadline = None if total_time_limit is None else t_start + float(total_time_limit)
    phase_order, reverse, order_sat = _parse_order(order)
    sat_candidate = sat_candidate and order_sat
    A, row_lo, row_hi, row_groups, col_lb, col_ub, is_int = arrays

    def _ord(c: list) -> list:
        return c[::-1] if reverse else c

    if confirm not in ("each", "final"):
        raise ValueError(f"confirm must be 'each' or 'final', got {confirm!r}")
    # Under "final" the filter must stop early enough to confirm what it has: the
    # engine holds back two recent-solve durations (``reserve_confirm``).
    eng = _Engine(
        A,
        row_lo,
        row_hi,
        col_lb,
        col_ub,
        is_int,
        row_groups,
        time_limit,
        confirm == "each",
        deadline=deadline,
        reserve_confirm=confirm == "final",
    )
    cons = set(row_groups)
    ints_active = set(int(j) for j in np.flatnonzero(is_int))
    # Finite bounds, all applied at the start. They are elements (filtered in
    # step 3) only for columns not held integer -- the engine applies a held
    # column's box regardless -- and only when include_bounds is set.
    finite_bounds: list[tuple[int, str]] = []
    for j in range(col_lb.size):
        if col_lb[j] > -_INF:
            finite_bounds.append((j, "lower"))
        if col_ub[j] < _INF:
            finite_bounds.append((j, "upper"))
    bounds = set(finite_bounds)
    proven = True
    pending: dict[tuple, np.ndarray] = {}

    sat_rows: Optional[int] = None
    sat_conflicts = 0
    if sat_candidate:
        # The clause-core stage gets a share of an overall budget that SATURATES
        # at a constant ceiling (#1153): where it pays off it needs seconds, so a
        # bigger budget must buy filtering, not a longer give-up on pigeonhole.
        sat_budget = (
            _SAT_STAGE_CAP_S
            if deadline is None
            else min(_SAT_STAGE_FRACTION * float(total_time_limit or 0.0), _SAT_STAGE_CAP_S)
        )
        sat_stop = t_start + sat_budget
        cand, sat_conflicts = _binary_core(
            A, row_lo, row_hi, col_lb, col_ub, is_int, row_groups, sat_stop
        )
        if cand is not None and len(cand) < len(cons):
            cons = set(cand)
            sat_rows = len(cand)

    # The base proof is mandatory, so under an overall budget it may use all of
    # it rather than one probe's ``time_limit`` (no-ip-65059 needs > 30 s).
    eng.probe_tl = math.inf if deadline is not None else time_limit
    try:
        base, _ = eng.probe(cons, bounds, ints_active)
        if base is not True and sat_rows is not None:
            # HiGHS did not prove the SAT candidate infeasible: discard it and
            # take the base proof on the full model, exactly as without it.
            cons = set(row_groups)
            sat_rows = None
            base, _ = eng.probe(cons, bounds, ints_active)
    except _OutOfTime:
        raise TimeoutError(
            "compute_iis: total_time_limit ran out before the model was proven infeasible"
        ) from None
    eng.probe_tl = time_limit
    if base is None and eng.deadline is not None and time.perf_counter() >= eng.deadline:
        raise TimeoutError(
            "compute_iis: total_time_limit ran out before the model was proven infeasible"
        )
    if base is not True:
        if base is False:
            raise ValueError(
                "compute_iis: the model is feasible — there is no infeasibility to explain."
            )
        raise ValueError(
            "compute_iis: could not prove the model infeasible within the time limit; "
            "an IIS is only defined for an infeasible model."
        )

    cap = max_batch if max_batch is not None else 1 << 30
    if cap < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")

    def batch_filter(cands: list, active: set, key) -> None:
        """Adaptive batch deletion (SCIP 10's dynamic batching of the deletion
        filter; Guieu & Chinneck's one-at-a-time filter is ``cap == 1``).

        Remove ``cands[i : i + size]`` from ``active``. If the rest is proven
        infeasible the whole batch is redundant: keep it removed and double the
        size. Otherwise restore it and halve the size; at size 1 the element is
        kept, with the probe's point as its certificate. An inconclusive probe
        counts as "not proven", so nothing is ever dropped without a proof, and
        a kept element decided by an inconclusive single probe clears
        ``proven_irreducible``. Worst case (every element needed) ~ n + log n
        probes; when few are needed ~ k log(n / k).
        """
        nonlocal proven
        i, size = 0, 1
        while i < len(cands):
            batch = cands[i : i + size]
            for e in batch:
                active.discard(e)
            try:
                res, x = eng.probe(cons, bounds, ints_active)
            except _OutOfTime:
                for e in batch:  # undecided: back into the subsystem
                    active.add(e)
                raise
            if res is True:
                i += len(batch)
                size = min(2 * size, cap)
                continue
            for e in batch:
                active.add(e)
            if len(batch) > 1:
                size = max(1, len(batch) // 2)
                continue
            if res is None:
                proven = False
            else:
                # ``probe`` returns a point with every feasible verdict.
                assert x is not None
                pending[key(batch[0])] = x
            i += 1

    def phase_integrality() -> None:
        # Guieu & Chinneck: if the LP relaxation is infeasible, all integrality
        # goes in one probe.
        if not ints_active:
            return
        res, _ = eng.probe(cons, bounds, set())
        if res is True:
            ints_active.clear()
        else:
            batch_filter(_ord(sorted(ints_active)), ints_active, lambda j: ("integrality", j))

    def phase_constraints() -> None:
        batch_filter(_ord(sorted(cons)), cons, lambda ci: ("constraint", ci))

    def phase_bounds() -> None:
        # Bounds of columns not held integer (a held column's box is not an element).
        if include_bounds:
            batch_filter(
                _ord([jb for jb in finite_bounds if jb[0] not in ints_active]),
                bounds,
                lambda jb: ("bound", jb[0], jb[1]),
            )

    def redo_each() -> MilpIIS:
        # A presolve-on verdict the filter acted on did not survive: the set is
        # not infeasible under both configurations. Redo everything with
        # per-probe confirmation rather than patch a set built on a bad verdict.
        left = None if deadline is None else deadline - time.perf_counter()
        if left is not None and left < _MIN_SOLVE_S:
            raise TimeoutError(
                "compute_iis: a presolve-on verdict was refuted and total_time_limit "
                "leaves no time to redo the filter with per-probe confirmation"
            )
        redo = _milp_iis_arrays(
            arrays,
            include_bounds=include_bounds,
            time_limit=time_limit,
            cert_tol=cert_tol,
            max_batch=max_batch,
            confirm="each",
            sat_candidate=sat_candidate,
            order=order,
            total_time_limit=left,
        )
        redo.n_solves += eng.n_solves
        redo.n_interrupted += eng.n_interrupted
        redo.confirm = "final"
        redo.fallback = True
        return redo

    # "final": the latest subsystem confirmed infeasible under both
    # configurations, kept as the anytime fallback. ("each" confirms every
    # verdict, so its current subsystem is always confirmed.)
    checkpoint: Optional[tuple[set, set, set]] = None
    if confirm == "final" and deadline is not None:
        # Checkpoint 0: the base set, proven presolve-on just above, confirmed
        # presolve-off too. Without it a budget that runs out before the first
        # phase ends has nothing confirmed to return (TimeoutError), although
        # the base set is a valid -- if large -- infeasible subsystem. Capped so
        # a slow presolve-free solve cannot eat the filter's budget.
        base_budget = min(
            _BASE_CONFIRM_FRACTION * float(total_time_limit or 0.0), _BASE_CONFIRM_CAP_S
        )
        base_stop = min(deadline, time.perf_counter() + base_budget)
        try:
            verdict0 = eng.confirm_infeasible(
                cons, bounds, ints_active, stop_at=base_stop, presolve=("off",)
            )
        except _OutOfTime:
            verdict0 = None
        if verdict0 is True:
            checkpoint = (set(cons), set(bounds), set(ints_active))
        elif verdict0 is False:
            return redo_each()

    complete = True
    try:
        by_name = {"I": phase_integrality, "C": phase_constraints, "B": phase_bounds}
        phases = [by_name[c] for c in phase_order]
        for phase in phases:
            phase()
            if confirm == "final" and phase is not phases[-1]:
                verdict = eng.confirm_infeasible(cons, bounds, ints_active)
                if verdict is False:
                    return redo_each()
                if verdict is True:
                    checkpoint = (set(cons), set(bounds), set(ints_active))
    except _OutOfTime:
        complete = False
        proven = False

    if confirm == "final":
        eng.reserve_confirm = False  # the reserve is for this confirmation
        try:
            verdict = eng.confirm_infeasible(cons, bounds, ints_active)
        except _OutOfTime:
            verdict = None
        if verdict is False and complete:
            return redo_each()
        if verdict is not True:
            # The current set is not confirmed: fall back to the last confirmed one.
            if checkpoint is None:
                if verdict is None and deadline is None:
                    return redo_each()  # no budget pressure: just confirm per probe
                raise TimeoutError(
                    "compute_iis: no subsystem could be confirmed infeasible under "
                    "both presolve configurations within total_time_limit"
                )
            cons, bounds, ints_active = (set(c) for c in checkpoint)
            complete = False
            proven = False

    # Bounds the final subsystem APPLIES (beyond held-integer boxes). With
    # include_bounds=False that is every finite bound, but none is an element.
    applied_bounds = sorted((j, s) for j, s in bounds if j not in ints_active)
    kept_bounds = applied_bounds if include_bounds else []
    certs = _check_certificates(pending, eng, cons, applied_bounds, ints_active, cert_tol)
    return MilpIIS(
        constraints=sorted(cons),
        bounds=kept_bounds,
        integrality=sorted(ints_active),
        certificates=certs,
        n_solves=eng.n_solves,
        proven_irreducible=proven,
        n_cross_checks=eng.n_cross_checks,
        n_refuted=eng.n_refuted,
        n_cross_inconclusive=eng.n_cross_inconclusive,
        confirm=confirm,
        complete=complete,
        n_interrupted=eng.n_interrupted,
        n_repairs=eng.n_repairs,
        n_artifacts=eng.n_artifacts,
        sat_rows=sat_rows,
        sat_conflicts=sat_conflicts,
        order=order,
    )


#: Deletion orders run by default when a portfolio is requested. Each won on part
#: of MIPcc27's smallset in the ordering sweep (``ICB`` sorted / ``CIB`` sorted /
#: ``ICB`` reversed; no order dominated, sizes differed by up to 40%).
DEFAULT_PORTFOLIO: tuple[str, ...] = ("ICB", "CIB", "ICB-r", "CIB-r")
#: Wall time held back from the members' budget for process start-up and for
#: returning their results, capped at this fraction of the budget.
_PORTFOLIO_MARGIN_S = 2.0
_PORTFOLIO_MARGIN_FRACTION = 0.05


def _iis_size(res: MilpIIS) -> int:
    return len(res.constraints) + len(res.bounds) + len(res.integrality)


def _portfolio_child() -> None:
    """Entry point of one portfolio member process.

    Reads a pickled job from stdin, runs one deletion order and pickles its
    outcome to the job's result file: ``("ok", MilpIIS)``, ``("timeout",
    message)`` for a :class:`TimeoutError` (this order confirmed nothing in time
    -- a result, not a failure) or ``("error", exception)`` for anything else,
    which the parent re-raises.
    """
    import os
    import pickle
    import sys
    import threading

    job = pickle.load(sys.stdin.buffer)

    def _orphan_watch() -> None:
        # The parent keeps our stdin open until it is done with us; EOF means it
        # closed it or died. Do not outlive it (a killed caller left members
        # running at 100% CPU).
        # Raw fd, not sys.stdin: a daemon thread blocked in a buffered read
        # holds its lock and aborts interpreter shutdown.
        fd = sys.stdin.fileno()
        while os.read(fd, 4096):
            pass
        os._exit(3)

    threading.Thread(target=_orphan_watch, daemon=True).start()
    deadline_wall = job["deadline_wall"]
    total = None if deadline_wall is None else max(0.0, deadline_wall - time.time())
    try:
        res = _milp_iis_arrays(
            job["arrays"], **job["kwargs"], order=job["order"], total_time_limit=total
        )
        out: tuple = ("ok", res)
    except TimeoutError as e:
        out = ("timeout", str(e))
    except Exception as e:  # forwarded: the parent re-raises it
        out = ("error", e)
    with open(job["result_path"], "wb") as f:
        pickle.dump(out, f)


def _portfolio_iis(
    arrays: tuple,
    kwargs: dict,
    portfolio: Sequence[str],
    workers: Optional[int],
    t_start: float,
    total_time_limit: Optional[float],
    grace: float = math.inf,
) -> MilpIIS:
    """Run one deletion filter per order in ``portfolio``, each in its own process,
    against one shared budget, and return the smallest complete result.

    Each member is an ordinary :func:`milp_iis` run, so each result is sound on its
    own (confirmed infeasible under both presolve configurations, certificates
    checked); the portfolio only chooses among them. Choice: the smallest
    *complete* result, ties to the earlier order in ``portfolio``; if no member
    completed, the smallest anytime result. :class:`TimeoutError` only when every
    member timed out; any other member exception is re-raised.

    Members are fresh ``python -c`` interpreters fed a pickled job, not
    ``multiprocessing`` children: a spawned child re-imports the caller's
    ``__main__``, which would re-run an unguarded script once per member.

    ``grace``: once the first member completes at elapsed time ``t``, the others
    may run until ``grace * t`` (and never past the budget); then they are
    terminated and recorded as ``"abandoned"``. ``inf`` waits for every member.
    """
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    orders = list(portfolio)
    if not orders:
        raise ValueError("portfolio must name at least one order")
    if len(set(orders)) != len(orders):
        raise ValueError(f"portfolio has duplicate orders: {orders}")
    for o in orders:
        _parse_order(o)
    if kwargs["confirm"] not in ("each", "final"):
        raise ValueError(f"confirm must be 'each' or 'final', got {kwargs['confirm']!r}")
    n_workers = min(len(orders), os.cpu_count() or 1) if workers is None else int(workers)
    if n_workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if not grace >= 1.0:
        raise ValueError(f"grace must be >= 1, got {grace}")

    # perf_counter is per-process; members get an absolute wall-clock deadline.
    wall0 = time.time() - (time.perf_counter() - t_start)
    hard_end = None if total_time_limit is None else wall0 + float(total_time_limit)
    member_deadline = None
    if total_time_limit is not None:
        margin = min(_PORTFOLIO_MARGIN_S, _PORTFOLIO_MARGIN_FRACTION * float(total_time_limit))
        member_deadline = wall0 + float(total_time_limit) - margin
    boot = (
        f"import sys; sys.path[:0] = {list(sys.path)!r}; "
        "from discopt._iis_milp import _portfolio_child; _portfolio_child()"
    )

    todo = list(range(len(orders)))
    # index -> (process, start wall time, result path, stderr file)
    running: dict[int, tuple[Any, float, str, Any]] = {}
    results: list[Optional[MilpIIS]] = [None] * len(orders)
    members: list[dict] = [
        {"order": o, "outcome": "not started", "size": None, "complete": None, "wall": None}
        for o in orders
    ]
    cutoff = hard_end
    with tempfile.TemporaryDirectory(prefix="discopt_iis_portfolio_") as tmp:
        try:
            while todo or running:
                now = time.time()
                while todo and len(running) < n_workers and (cutoff is None or now < cutoff):
                    i = todo.pop(0)
                    result_path = os.path.join(tmp, f"member{i}.pkl")
                    err = open(os.path.join(tmp, f"member{i}.err"), "w+b")
                    proc = subprocess.Popen(
                        [sys.executable, "-c", boot],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=err,
                    )
                    assert proc.stdin is not None
                    job = dict(
                        arrays=arrays,
                        kwargs=kwargs,
                        order=orders[i],
                        deadline_wall=member_deadline,
                        result_path=result_path,
                    )
                    pickle.dump(job, proc.stdin)
                    # Kept open: the member exits on EOF, so it cannot outlive us.
                    proc.stdin.flush()
                    running[i] = (proc, time.time(), result_path, err)
                if not running:
                    break
                finished = [i for i, r in running.items() if r[0].poll() is not None]
                if not finished:
                    if cutoff is not None and time.time() >= cutoff:
                        break  # the rest are abandoned below
                    time.sleep(0.02)
                    continue
                for i in finished:
                    proc, t0, result_path, err = running.pop(i)
                    assert proc.stdin is not None
                    proc.stdin.close()
                    members[i]["wall"] = round(time.time() - t0, 3)
                    if proc.returncode != 0 or not os.path.exists(result_path):
                        err.seek(0)
                        tail = err.read().decode(errors="replace").strip()[-2000:]
                        err.close()
                        raise RuntimeError(
                            f"compute_iis portfolio member {orders[i]!r} exited without a "
                            f"result (exit code {proc.returncode}):\n{tail}"
                        )
                    err.close()
                    with open(result_path, "rb") as f:
                        kind, payload = pickle.load(f)
                    if kind == "error":
                        raise payload
                    if kind == "timeout":
                        members[i].update(outcome="timeout", message=payload)
                        continue
                    results[i] = payload
                    members[i].update(
                        outcome="ok", size=_iis_size(payload), complete=payload.complete
                    )
                    if payload.complete and math.isfinite(grace):
                        grace_end = wall0 + grace * (time.time() - wall0)
                        cutoff = grace_end if cutoff is None else min(cutoff, grace_end)
        finally:
            for i, (proc, t0, _, err) in running.items():
                proc.kill()
                proc.wait()
                assert proc.stdin is not None
                proc.stdin.close()
                err.close()
                members[i].update(outcome="abandoned", wall=round(time.time() - t0, 3))

    done = [(i, r) for i, r in enumerate(results) if r is not None]
    if not done:
        why = "; ".join(f"{mb['order']}: {mb.get('message') or mb['outcome']}" for mb in members)
        raise TimeoutError(
            "compute_iis: no portfolio member confirmed a subsystem infeasible "
            f"within total_time_limit ({why})"
        )
    pool = [(i, r) for i, r in done if r.complete] or done
    _, best = min(pool, key=lambda ir: (_iis_size(ir[1]), ir[0]))
    best.members = members
    return best
