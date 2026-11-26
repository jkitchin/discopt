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
import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from discopt import _iis_backend as _bk
from discopt._optional_highs import wrap_callback

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
#: Rolling checkpoints (anytime answers inside a phase): a presolve-free
#: confirmation is attempted after a successful deletion once the subsystem has
#: shrunk by this factor since the last attempt ...
_ROLL_SHRINK = 0.9
#: ... while all attempts together have used at most this share of the elapsed
#: time (0.10 starved them: on MIPcc27's dijle one confirmation of a row-phase set
#: costs 15-26 s while a probe costs 32 ms, so a 10% share early in the run never
#: paid for one -- iis_generalint_strategy.md R2) ...
_ROLL_SHARE = 0.25
#: ... each with at least this much, 4x the slowest successful confirmation so
#: far, and whatever the share leaves unspent (never more than checkpoint 0's
#: saturated budget). Measured on MIPcc27
#: (exp14, 300 s): intermediate sets confirm presolve-free in 0.1-2.5 s early in
#: a phase (dijle's down to 1417 of 3136 held integralities, all <= 0.6 s) and not
#: in 10 s late in it; the final sets of dijle / dingo / dives not in 90 s.
#: Falsified on the way: those integrality-phase sets are not smaller answers --
#: each relaxed integrality turns its two bounds into elements (dijle's 1417-held
#: set counts 8001 against the base set's 6282) -- hence the size gate in
#: ``roll``.
_ROLL_MIN_S = 2.0
#: How long a portfolio keeps its other members running once one completes at
#: elapsed time t: ``(grace, (lo, hi))`` gives ``clamp((grace - 1) * t, lo, hi)``
#: extra seconds, ``(inf, None)`` waits for every member.
#:
#: ``"fast"`` (the default) trades a few elements for minutes. On MIPcc27 every
#: later member that beat the first finisher did so within 31 s of it on the
#: 6-order sweep; at 600 s the clamped wait exited 100-200 s earlier on
#: neos859080 / mod008inf / hole10 / jjliu_clique with the same sizes, except
#: jjliu_clique 277 vs 266 (its ICB member completed ~90 s after ICB-r).
#: ``"all"`` is for the smallest set regardless of time.
PORTFOLIO_WAITS: dict[str, tuple[float, Optional[tuple[float, float]]]] = {
    "fast": (2.0, (10.0, 30.0)),
    "all": (math.inf, None),
}
#: Wall-clock grace past a run's own budget before the interrupt callback stops
#: it. HiGHS normally honours ``time_limit`` itself; this catches the overruns
#: measured on presolve-free re-solves (7-10x a 30 s limit on ``enlight4``).
#: Route-specific since P3.3: only the HiGHS backend (and ``_iis_reduce``'s HiGHS
#: solve) installs the guard; the Rust driver polls its own deadline (measured: a
#: 0.5 s budget returns in <= 0.58 s on enlight 8x8..14x14, presolve on and off).
_GRACE_S = _bk.GRACE_S  # the HiGHS route's overrun grace (single source: _iis_backend)


class _Refuted(Exception):
    """A presolve-free confirmation found an exact-integer point for a set the
    filter holds as proven: an earlier verdict was wrong, redo per probe."""


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
    # Constraints in the lattice candidate (:func:`_lattice_candidate`; None: not
    # used). The name is from the mod-2 candidate it generalizes.
    parity_rows: Optional[int] = None
    n_lattice: int = 0  # probes proven infeasible by the exact lattice test, no solve
    n_exact: int = 0  # confirmations decided exactly when presolve-off could not
    ladder_size: Optional[int] = None  # elements left by the oracle ladder (None: not run)
    ladder_calls: int = 0  # oracle calls the ladder made
    ladder_rejected: bool = False  # ladder's set not confirmed: its drops were discarded
    n_rolling: int = 0  # rolling checkpoints confirmed during the filter (anytime answers)
    n_reduced_confirmed: int = 0  # confirmations decided via the exact reduction
    n_exact_confirmed: int = 0
    n_exact_refuted: int = 0
    n_swept: int = 0  # elements the structural sweep dropped with no solve
    # Sweeps undone because they would have dropped an element holding a
    # certificate (a contradiction that only tolerances can produce).
    n_sweep_reverted: int = 0
    order: str = "ICB"  # phase order (and "-r": reversed element order) that produced it
    # The answer's post-pass (:func:`_polish`): elements the forced-violation
    # repair tried / certified, add-one candidates tried and moves adopted.
    n_forced: int = 0
    n_forced_certified: int = 0
    n_add_one_tried: int = 0
    n_add_one_moves: int = 0
    n_relaxed: int = 0  # integralities the anytime relaxation (_relax_integrality) traded away
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
        engine: str = "rust",
    ):
        from discopt._iis_backend import make_backend

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
        # Slowest successful two-route confirmation so far (seconds). Sizes the
        # reserve for the final confirmation and the rolling budgets: on general-
        # integer models confirmations cost 100-1000x a probe, so probe times
        # alone under-reserved them (dijle: final confirmation left 1.7 s).
        self.confirm_ref = 0.0
        # The last set a presolve-on solve proved infeasible (cons, bounds, ints).
        # The filter's current set is that set whenever the last batch it could
        # not drop was put back, so its final confirmation needs only the
        # presolve-free route (:meth:`proven_on`).
        self._proven_on: Optional[tuple[frozenset, frozenset, frozenset]] = None
        self._stop = math.inf
        self.n_interrupted = 0
        self.n_repairs = 0
        self.n_exact = 0  # exact (rational) decisions attempted as the second route
        self.n_presolve = 0  # presolve-only solves (ladder rung 0)
        self.n_exact_confirmed = 0  # ... that proved the set infeasible
        self.n_reduced = 0  # second-route attempts through the exact reduction
        self.n_reduced_confirmed = 0  # ... that proved the set infeasible
        self.n_exact_refuted = 0  # ... that found it feasible
        self.n_artifacts = 0
        # The exact lattice test (:mod:`discopt._iis_lattice`): constraint id ->
        # its integer equality rows, set by :meth:`arm_lattice` only when the
        # whole model is lattice-infeasible (None: disarmed, no cost per probe).
        self._lat: Optional[dict[int, list]] = None
        self.n_lattice = 0
        # Held-integer columns and their declared box in the current configuration.
        self._held = np.zeros(0, dtype=np.int64)
        self._cfg: tuple[np.ndarray, ...] = ()
        # highs-parity P3.2: the solver behind every probe (``discopt._iis_backend``):
        # the in-house MILP driver by default, HiGHS as the opt-in.
        self.backend = make_backend(engine, A, row_lo, row_hi, col_lb, col_ub, is_int)
        self.engine = engine
        self._inf = self.backend.inf

    def arm_lattice(self, stop_at: Optional[float] = None) -> bool:
        """Arm the lattice test when the whole model is lattice-infeasible.

        The test is monotone -- dropping a row, or relaxing an integrality, never
        turns a lattice-feasible system infeasible -- so when the full model passes
        it, every subsystem the filter will probe passes too, and the per-probe
        test could never fire: it is left disarmed. On MIPcc27's smallset it arms
        on enlight4, enlight11 and neos-807454 only."""
        from discopt._iis_lattice import integer_rows, lattice_infeasible

        rows = integer_rows(self.A, self.row_lo, self.row_hi)
        lat: dict[int, list] = {}
        for ci, rs in self.row_groups.items():
            got = [rows[int(r)] for r in np.asarray(rs).ravel() if int(r) in rows]
            if got:
                lat[ci] = got
        allrows = [t for ci in sorted(lat) for t in lat[ci]]
        held = set(int(j) for j in np.flatnonzero(self.is_int))
        if lattice_infeasible(allrows, held, stop_at) is True:
            self._lat = lat
        return self._lat is not None

    def _lattice_proves(self, active_cons, active_int) -> bool:
        """The set's integer equality rows are infeasible over the integers it
        holds, whatever its bounds: an exact proof (:mod:`discopt._iis_lattice`)."""
        if self._lat is None:
            return False
        from discopt._iis_lattice import lattice_infeasible

        lat = self._lat
        rows = [t for ci in sorted(active_cons) if ci in lat for t in lat[ci]]
        if not rows:
            return False
        return lattice_infeasible(rows, active_int, self.deadline) is True

    def _configure(
        self,
        active_cons: set[int],
        active_bounds: set[tuple[int, str]],
        active_int: set[int],
    ) -> None:
        """Load one probe's rows, bounds and integrality into the solver."""
        self._load(*self._arrays(active_cons, active_bounds, active_int))

    def _arrays(
        self,
        active_cons: set[int],
        active_bounds: set[tuple[int, str]],
        active_int: set[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(rlo, rhi, lb, ub, integ)`` of one probe, in the backend's infinity."""
        inf = self._inf
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
        return rlo, rhi, lb, ub, integ

    def _load(self, rlo, rhi, lb, ub, integ) -> None:
        self._held = np.flatnonzero(integ)
        self._cfg = (rlo, rhi, lb, ub)  # what this probe imposes, for witness checks
        self.backend.configure(rlo, rhi, lb, ub, integ)

    def _solve(
        self, presolve: str, stop_at: Optional[float] = None, max_nodes: Optional[int] = None
    ):
        """Solve the configured probe from scratch; returns the backend status
        (``discopt._iis_backend``: ``OPTIMAL`` / ``INFEASIBLE`` / ...).

        The run's budget is the per-probe limit, cut to ``self.deadline``; a
        caller-given ``stop_at`` is its own budget and replaces the per-probe
        limit (still cut to the deadline). Raises :class:`_OutOfTime` when that
        leaves nothing.

        Why ``stop_at`` replaces rather than cuts: only confirmations pass it
        (checkpoint 0, the ladder's set, and the exact-witness repairs inside
        them), each with a budget sized for that solve. Cut to the 30 s probe
        limit, checkpoint 0 gave up on MIPcc27's dives and neos-1461051, whose
        full models need 63 s and 90 s presolve-off: nothing was ever
        confirmed, and both runs ended in TimeoutError with no answer.
        """
        now = time.perf_counter()
        stop = now + float(self.probe_tl) if stop_at is None else math.inf
        hard = self.deadline
        if hard is not None and self.reserve_confirm:
            # Two confirmation solves, each about as long as recent ones, with a
            # 25% margin. (Each may still overrun its own budget by up to
            # ``_GRACE_S`` before the guard stops it.)
            hard -= max(2.5 * max(self._recent, default=0.0), 2.0 * self.confirm_ref)
        for lim in (hard, stop_at):
            if lim is not None:
                stop = min(stop, lim)
        if stop - now < _MIN_SOLVE_S:
            raise _OutOfTime
        self.n_solves += 1
        try:
            status = self.backend.run(presolve, stop, max_nodes)
        finally:
            self._recent.append(time.perf_counter() - now)
        if status == _bk.INTERRUPTED:
            self.n_interrupted += 1
        return status

    def _point(self) -> np.ndarray:
        return np.asarray(self.backend.point(), dtype=np.float64)

    def _witness(
        self, stop_at: Optional[float] = None, tol: Optional[float] = None
    ) -> Optional[np.ndarray]:
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

        ``tol`` replaces ``mip_feasibility_tolerance`` as the row standard (the
        certificate repair retries at 1e-9 when a point met its rows only to the
        MIP tolerance and so failed the certificate check).
        """
        x = self._point()
        held = self._held
        if held.size == 0:
            return x
        self.n_repairs += 1
        xr = np.round(x[held])
        mip_tol = self.backend.mip_tol() if tol is None else float(tol)
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
        be = self.backend
        be.set_cols(held, xr, xr, integer=False)
        try:
            with be.lp_tolerance(mip_tol):
                ok = self._solve("off", stop_at) == _bk.OPTIMAL
                y = self._point() if ok else None
        finally:
            lo = np.where(self.col_lb[held] <= -_INF, -self._inf, self.col_lb[held])
            hi = np.where(self.col_ub[held] >= _INF, self._inf, self.col_ub[held])
            be.set_cols(held, lo, hi, integer=True)
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
        if self._lattice_proves(active_cons, active_int):
            # Exact, so it needs no presolve-free cross-check either; and it
            # decides in milliseconds sets HiGHS spent the probe limit on (relaxed
            # columns left unbounded by the filter: enlight11).
            self.n_lattice += 1
            self._record_proven(active_cons, active_bounds, active_int)
            return True, None
        self._configure(active_cons, active_bounds, active_int)
        ms = self._solve("choose")
        if ms == _bk.OPTIMAL:
            x = self._witness()
            return (False, x) if x is not None else (None, None)
        if ms not in (_bk.INFEASIBLE, _bk.UNBOUNDED_OR_INFEASIBLE):
            return None, None
        if ms == _bk.INFEASIBLE and not self.confirm_each:
            self._record_proven(active_cons, active_bounds, active_int)
            return True, None
        # An infeasibility claim is what lets the filter DROP an element, so it
        # stands only when a presolve-free solve agrees -- the #1634 policy of
        # discopt's own HiGHS MILP route, where presolve-on and presolve-off each
        # produced false claims on a 1200-instance panel but never on the same
        # instance. (``kUnboundedOrInfeasible`` cannot be "unbounded" under a zero
        # objective; the presolve-free solve decides it, in either mode.)
        self.n_cross_checks += 1
        ms2 = self._solve("off")
        if ms2 == _bk.INFEASIBLE:
            if ms == _bk.INFEASIBLE:  # the presolve-on proof stands: on record
                self._record_proven(active_cons, active_bounds, active_int)
            return True, None
        if ms2 == _bk.OPTIMAL:
            x = self._witness()
            if x is not None:
                if ms == _bk.INFEASIBLE:
                    self.n_refuted += 1
                return False, x
        # Undecided: presolve-off neither agreed nor produced an exact-integer
        # witness against the presolve-on claim. Never acted on.
        self.n_cross_inconclusive += 1
        return None, None

    def presolve_proves(self, active_cons, active_bounds, active_int, stop_at=None) -> bool:
        """Does presolve ALONE prove the set infeasible? (The ladder's first rung:
        cheap, incomplete, and a presolve-on claim like any other.)"""
        self._configure(active_cons, active_bounds, active_int)
        now = time.perf_counter()
        stop = now + float(self.probe_tl)
        for lim in (self.deadline, stop_at):
            if lim is not None:
                stop = min(stop, lim)
        if stop - now < _MIN_SOLVE_S:
            raise _OutOfTime
        self.n_presolve += 1
        return bool(self.backend.presolve_only(stop))

    def capped_proves(self, active_cons, active_bounds, active_int, max_nodes: int) -> bool:
        """Does a presolve-on MIP solve limited to ``max_nodes`` branch-and-bound
        nodes prove the set infeasible? (A ladder rung. A node limit, not a time
        limit, so which elements the rung drops does not depend on machine speed
        (#912); the overall deadline still stops it.)"""
        self._configure(active_cons, active_bounds, active_int)
        return bool(self._solve("choose", max_nodes=int(max_nodes)) == _bk.INFEASIBLE)

    def _record_proven(self, active_cons, active_bounds, active_int) -> None:
        """Record this exact set's presolve-on ``Infeasible`` -- only when the
        probe returns it as proven, never a claim a cross-check then refuted."""
        self._proven_on = (
            frozenset(active_cons),
            frozenset(active_bounds),
            frozenset(active_int),
        )

    def proven_on(self, active_cons, active_bounds, active_int) -> bool:
        """Whether this exact set's presolve-on ``Infeasible`` is on record."""
        return self._proven_on == (
            frozenset(active_cons),
            frozenset(active_bounds),
            frozenset(active_int),
        )

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
        all_infeasible = True
        for p in presolve:
            if p == "off":
                if self._lattice_proves(active_cons, active_int):
                    # An integer Farkas certificate: an exact proof in
                    # milliseconds, where a presolve-free solve needed 47 s on
                    # enlight11's set.
                    self.n_exact += 1
                    self.n_exact_confirmed += 1
                    continue
                if self._reduced(stop_at) is True:
                    # Exact reduction + presolve-free HiGHS on the reduced model
                    # (:mod:`discopt._iis_reduce`): decides in 0.3-11 s the
                    # general-integer sets the raw presolve-free solve could not
                    # in 60-300 s (MIPcc27 dijle / dingo / dives).
                    continue
            st = self._solve(p, stop_at)
            if st == _bk.OPTIMAL and self._witness(stop_at) is not None:
                return False
            ok = st == _bk.INFEASIBLE
            if not ok and p == "off":
                # Presolve-off could not answer (on big-M sets it returns
                # "Optimal" points whose integers are 1e-11 off and do not
                # survive rounding). Decide the set exactly instead -- a strictly
                # stronger second route -- or None when too large. Not tried
                # first: its caps bound size, not time, and on flugplinf it ran
                # 13 s and 46 s to "undecided" where presolve-off took 0.04 s.
                exact = self._exact(stop_at)
                if exact is False:
                    self.n_exact_refuted += 1
                    return False
                if exact is True:
                    self.n_exact_confirmed += 1
                    ok = True
            all_infeasible &= ok
        return True if all_infeasible else None

    def _reduced(self, stop_at: Optional[float]) -> Optional[bool]:
        """The configured set through :func:`discopt._iis_reduce.reduced_infeasible`:
        True when proven infeasible, else None (no conclusion either way). Its
        budget is ``stop_at`` (or one probe's limit without one), cut to the
        deadline, with the engine's wall-clock guard on the HiGHS run."""
        from discopt._iis_reduce import reduced_infeasible

        stop = time.perf_counter() + float(self.probe_tl) if stop_at is None else stop_at
        if self.deadline is not None:
            stop = min(stop, self.deadline)
        held = np.zeros(self.col_lb.size, dtype=bool)
        held[self._held] = True
        rlo, rhi, lb, ub = self._cfg

        def guard(h, s) -> None:
            def _g(event) -> None:
                event.interrupt(time.perf_counter() > s + _GRACE_S)

            # #2221: ``_iis_reduce`` re-raises an exception from it after ``run``.
            g = wrap_callback(h, _g)
            h.cbMipInterrupt.subscribe(g)
            h.cbSimplexInterrupt.subscribe(g)
            h.cbIpmInterrupt.subscribe(g)

        self.n_reduced += 1
        verdict = reduced_infeasible(
            self.A, rlo, rhi, lb, ub, held, stop, guard, engine=self.engine
        )
        if verdict is True:
            self.n_reduced_confirmed += 1
        return verdict

    def _exact(self, stop_at: Optional[float]) -> Optional[bool]:
        """Exact decision of the configured set (:mod:`discopt._iis_exact`):
        True infeasible, False feasible, None undecided."""
        from discopt._iis_exact import exact_feasible

        stop = self.deadline if stop_at is None else stop_at
        if stop is not None and self.deadline is not None:
            stop = min(stop, self.deadline)
        rlo, rhi, lb, ub = self._cfg
        self.n_exact += 1
        return exact_feasible(self.A, rlo, rhi, lb, ub, self._held, stop_at=stop)

    def forced_point(
        self,
        cons: set[int],
        bounds: set[tuple[int, str]],
        ints: set[int],
        key: tuple,
        margin: float,
        stop_at: Optional[float],
        tol: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], bool]:
        """A candidate certificate for element ``key`` of the set: a point of the
        set without ``key`` that is forced to violate ``key`` by ``margin`` -- a
        row beyond one of its sides, a bound beyond its value, an integer column
        held inside ``[f + margin, f + 1 - margin]`` (its box still applies, the
        checker's rule for a relaxed integrality). Integers are rounded and the
        rest re-solved by :meth:`_witness` (row standard ``tol``).

        Returns ``(point, exhausted)``: ``point`` is None when no variant gave
        one; ``exhausted`` is True when every variant was proven infeasible, so
        no certificate with this margin exists. The point is only a candidate:
        the caller checks it by arithmetic (:func:`_check_certificates`). This
        decides nothing about the set; it is never an infeasibility claim."""
        kind = key[0]
        c2, b2, i2 = set(cons), set(bounds), set(ints)
        if kind == "constraint":
            c2.discard(key[1])
        elif kind == "bound":
            b2.discard((key[1], key[2]))
        else:
            i2.discard(key[1])
        rlo, rhi, lb, ub, integ = self._arrays(c2, b2, i2)
        inf = self._inf
        variants: list[tuple[str, int, float, float]] = []  # (row|col, index, lo, hi)
        if kind == "constraint":
            for r in np.asarray(self.row_groups[key[1]]).ravel():
                r = int(r)
                if self.row_hi[r] < _INF:
                    variants.append(("row", r, self.row_hi[r] + margin, inf))
                if self.row_lo[r] > -_INF:
                    variants.append(("row", r, -inf, self.row_lo[r] - margin))
        elif kind == "bound":
            j = key[1]
            if key[2] == "lower":
                variants.append(("col", j, -inf, self.col_lb[j] - margin))
            else:
                variants.append(("col", j, self.col_ub[j] + margin, inf))
        else:
            j = key[1]
            lo, hi = self.col_lb[j], self.col_ub[j]
            if lo > -_INF and hi < _INF and hi - lo <= _FORCED_INT_SPAN:
                fs = list(range(int(math.floor(lo)), int(math.ceil(hi))))
            else:
                # A wide or unbounded integer: take the fractional parts next to
                # where the set without the integrality puts the column.
                box_lb, box_ub = lb.copy(), ub.copy()
                box_lb[j] = -inf if lo <= -_INF else lo
                box_ub[j] = inf if hi >= _INF else hi
                self._load(rlo, rhi, box_lb, box_ub, integ)
                if self._solve("choose", stop_at) != _bk.OPTIMAL:
                    return None, False
                xj = float(self._point()[j])
                fs = [int(math.floor(xj)), int(math.floor(xj)) - 1]
            for f in fs:
                vlo = f + margin if lo <= -_INF else max(lo, f + margin)
                vhi = f + 1 - margin if hi >= _INF else min(hi, f + 1 - margin)
                if vlo <= vhi:
                    variants.append(("col", j, vlo, vhi))
        exhausted = bool(variants)
        for where, i, vlo, vhi in variants:
            r_lo, r_hi, c_lb, c_ub = rlo.copy(), rhi.copy(), lb.copy(), ub.copy()
            if where == "row":
                r_lo[i], r_hi[i] = vlo, vhi
            else:
                c_lb[i], c_ub[i] = vlo, vhi
            self._load(r_lo, r_hi, c_lb, c_ub, integ)
            st = self._solve("choose", stop_at)
            if st != _bk.INFEASIBLE:
                exhausted = False
            if st == _bk.OPTIMAL:
                y = self._witness(stop_at, tol)
                if y is not None:
                    return y, False
        return None, exhausted


def _resolve_iis_engine(iis_engine: Optional[str]) -> str:
    """The IIS matrix engine: an explicit argument, else ``DISCOPT_IIS_ENGINE``, else the
    global engine switch (``discopt.solver._lp_engine``: HiGHS where ``highspy`` is
    installed, else ``"rust"``, discopt's own MILP driver).

    History: ``"rust"`` was the default from 2026-11-07 (highs-parity P6.4) to the
    2026-11-17 engine switch. P3.2 measured 53/53 parity on the fixtures, at 1.08 vs
    0.77 s total wall, and #2225's MIPcc27 smallset found the Rust driver 1.9-2.6x
    slower and leaving probes undecided where HiGHS decides them. There is no Rust
    fallback here: a probe the engine cannot decide leaves the set marked incomplete,
    never wrong. ``DISCOPT_IIS_ENGINE=rust`` / ``DISCOPT_LP_ENGINE=rust`` select the
    Rust driver."""
    if iis_engine is None:
        from discopt.solver import _engine_default, _env_selector

        env = _env_selector("DISCOPT_IIS_ENGINE")
        return env if env is not None else _engine_default()
    return iis_engine


def _check_certificates(
    pending: dict, eng: _Engine, cons, bounds, ints, tol, own_min: Optional[float] = None
) -> dict:
    """The points in ``pending`` that certify their element: ``x`` satisfies the
    final subsystem minus ``key`` (every other row, applied bound and integrality
    within ``tol``) and violates ``key`` by more than ``tol`` (``own_min``
    instead, when given).

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
        if rest <= tol and e > (tol if own_min is None else own_min):
            out[key] = x
    return out


#: Forced-violation certificate repair (:func:`_polish`): margins tried in order.
#: Fable's measurement on kedron / otra / osun / dijle / dingo / enlight11 (every
#: missing certificate found): several elements failed at 1e-3 and passed at 1e-4.
#: The last is just above the strong-certificate floor ``_STRONG_OWN`` (2e-5),
#: the thinnest violation a certificate kept here can have: on otra, twelve
#: integralities in a big-M row (coefficient 1e5) admit fractional parts only up
#: to ~2.9e-5, so every margin down to 3e-5 is proven infeasible for them.
_FORCED_MARGINS = (1e-3, 1e-4, 3e-5, 2.1e-5)
#: A certificate counts as present for the repair only when it holds with margin:
#: every other element within ``_STRONG_OTHERS`` and its own violated by more than
#: ``_STRONG_OWN``. A point that passes the engine's ``cert_tol`` (1e-6) but
#: violates its element by, say, 5e-6 fails a checker at 1e-5; the repair then
#: looks for a stronger one and keeps the weak one if it finds none. Measured on
#: otra: 2 of 11 engine certificates were dropped by ``discopt.mipcc``'s check
#: (the same margins).
_STRONG_OTHERS = 5e-6
_STRONG_OWN = 2e-5
#: An integer column whose box spans at most this many units gets one variant per
#: unit interval; a wider one gets the two next to a relaxed solution's value.
_FORCED_INT_SPAN = 4


#: Add-one neighbourhood (:func:`_add_one`): the sweep's extent is a probe count
#: (deterministic: the same answer on any machine), and its share of the time
#: left is only a stop -- a sweep cut short by it is discarded, never half-used.
#: The same share caps the confirmation of a set it changed. The strategy's kill
#: criterion: a sweep must not cost more than 10% of the budget. Measured (600 s
#: budget, highspy 1.16): osun's move is found within its full 586-candidate
#: sweep; otra's full sweep, 2132 candidates and 12,796 probes, took 145 s and
#: found none, so the cap keeps such a sweep to ~45 s (11 ms a probe there).
#: The share saturates at ``_ADD_ONE_CAP_S`` (#1153 role 2): the sweep's work is
#: the probe count, so a bigger budget must not buy it a longer stop. The cap is
#: ``_ADD_ONE_FRACTION`` of a 600 s budget, so it never binds at <= 600 s (the time
#: left is at most the budget) and only stops the share growing beyond that.
_ADD_ONE_FRACTION = 0.10
_ADD_ONE_CAP_S = 60.0
_ADD_ONE_MAX_PROBES = 4000


def _element_keys(cons, bounds, ints) -> list[tuple]:
    return (
        [("constraint", c) for c in sorted(cons)]
        + [("integrality", j) for j in sorted(ints)]
        + [("bound", j, side) for j, side in sorted(bounds)]
    )


def _add_one(
    eng: _Engine,
    cons: set[int],
    bounds: set,
    ints: set[int],
    stop_at: float,
    max_probes: int = _ADD_ONE_MAX_PROBES,
) -> tuple[Optional[tuple[set[int], set, set[int]]], dict, bool, int]:
    """Add-one neighbourhood over constraints: for each constraint outside the
    answer that shares a column with it (most shared columns first), add it and
    re-run the constraint deletion filter over the answer's constraints, then
    test the added one. A move is adopted when the set shrinks, and the scan
    goes on from the new set. Integralities and bounds stay fixed during the
    scan (more of them only makes a deletion harder, so a constraint the scan
    keeps stays needed once some are dropped) and are re-filtered once at the
    end: each integrality jointly with its column's box, as the integrality
    phase does, then each bound.

    Returns ``(new_set or None, pending, decided, tried)``: ``new_set`` only
    when a move was adopted; ``pending`` the probe points of the final
    re-filter (candidate certificates); ``decided`` whether every probe behind
    the new set was decided; ``tried`` the candidates scanned. Probes run under
    the engine's policy (a presolve-on verdict is acted on), so the caller must
    confirm the new set (:meth:`_Engine.confirm_infeasible`).

    The scan stops after ``max_probes`` probes (the moves found so far stand:
    the count fixes them). Reaching ``stop_at`` first raises :class:`_OutOfTime`,
    so the caller discards the sweep: the clock decides whether there is a
    result, never which one.

    The smallest local search over IISs (strategy-for-smaller-iis, steps 2-3);
    measured by Fable's prototype: osun 20 -> 19, otra 23 -> 22."""
    A = eng.A
    cols_of: dict[int, set[int]] = {}

    def cols(c: int) -> set[int]:
        got = cols_of.get(c)
        if got is None:
            got = set()
            for r in np.asarray(eng.row_groups[c]).ravel():
                got.update(A.indices[A.indptr[r] : A.indptr[r + 1]].tolist())
            cols_of[c] = got
        return got

    best = set(cons)
    decided = True
    tried = 0
    moved = False
    scanned: set[int] = set()
    n_probes = [0]

    def probe(c2: set[int]) -> Optional[bool]:
        if time.perf_counter() >= stop_at:
            raise _OutOfTime
        n_probes[0] += 1
        return eng.probe(c2, bounds, ints)[0]

    while n_probes[0] < max_probes:
        supp: set[int] = set()
        for c in best:
            supp |= cols(c)
        cand = [
            (-len(cols(c) & supp), c)
            for c in eng.row_groups
            if c not in best and c not in scanned and cols(c) & supp
        ]
        improved = False
        for _, c in sorted(cand):
            if n_probes[0] >= max_probes:
                break
            scanned.add(c)
            tried += 1
            R = best | {c}
            und = False
            old = sorted(best)
            for k, q in enumerate(old):
                # A move needs two of the old constraints gone.
                if (len(best) + 1 - len(R)) + (len(old) - k) < 2:
                    break
                v = probe(R - {q})
                if v is True:
                    R.discard(q)
                elif v is None:
                    und = True
            if len(R) > len(best):
                continue
            v = probe(R - {c})
            if v is True:
                R.discard(c)
            elif v is None:
                und = True
            if len(R) < len(best):
                best, moved, improved = R, True, True
                decided = decided and not und
                break  # rescan around the new set
        if not improved:
            break
    if not moved:
        return None, {}, True, tried
    new_c, new_b, new_i = set(best), set(bounds), set(ints)
    pending: dict[tuple, np.ndarray] = {}
    for j in sorted(new_i):
        v, x = eng.probe(new_c, new_b, new_i - {j})
        if v is True:
            new_i.discard(j)
        elif v is None:
            decided = False
        elif x is not None:
            pending[("integrality", j)] = x
    for jb in sorted(new_b):
        v, x = eng.probe(new_c, new_b - {jb}, new_i)
        if v is True:
            new_b.discard(jb)
        elif v is None:
            decided = False
        elif x is not None:
            pending[("bound", jb[0], jb[1])] = x
    # The constraints once more, for their points (and in case dropping an
    # integrality or bound above changed nothing they rely on: then each stays).
    for q in sorted(new_c):
        v, x = eng.probe(new_c - {q}, new_b, new_i)
        if v is True:
            new_c.discard(q)
        elif v is None:
            decided = False
        elif x is not None:
            pending[("constraint", q)] = x
    return (new_c, new_b, new_i), pending, decided, tried


#: Integrality relaxation of an anytime answer (:func:`_relax_integrality`):
#: counterexample rounds per stage (each round restores at least one element, so
#: the cap only bounds a stage that would restore one at a time; it fixes the
#: result, the clock only stops it), and the share of the time left the pass may
#: use. Measured by Fable's prototype on dingo's 2197-element anytime answer:
#: 66 solves in 17 s, 1867 elements; on dijle's, 6 solves in 1.8 s.
#: The share saturates at ``_RELAX_CAP_S`` (#1153 role 2), ``_RELAX_FRACTION`` of
#: a 600 s budget: it never binds at <= 600 s and stops growing beyond.
_RELAX_MAX_ROUNDS = 64
_RELAX_FRACTION = 0.25
_RELAX_CAP_S = 150.0
#: Share of ``total_time_limit`` the filter leaves for the post-pass
#: (:func:`_polish`): an anytime answer arrives at the deadline otherwise, with no
#: time to relax or certify it. 10% of mipcc's 600 s run is ~57 s; the prototype's
#: post-pass (relaxation and every certificate) took ~36 s on dingo, ~12 s on dijle.
#: The reserve saturates at ``_POLISH_RESERVE_CAP_S`` (#1153 role 2): the post-pass
#: is bounded by counts (``_RELAX_MAX_ROUNDS``, ``_ADD_ONE_MAX_PROBES``, the forced
#: margins), so a reserve that kept growing with the budget would only take
#: filtering time it cannot use. The cap is the 10% of a 600 s budget, so it never
#: binds at <= 600 s; beyond that every extra second goes to the filter.
_POLISH_RESERVE_FRACTION = 0.10
_POLISH_RESERVE_CAP_S = 60.0
#: Below this a relaxed column's value counts as integral / inside its bound.
_RELAX_TOL = 1e-9


def _relax_integrality(
    eng: _Engine,
    cons: set[int],
    bounds: set,
    ints: set[int],
    keep: set[int],
    stop_at: float,
    max_rounds: int = _RELAX_MAX_ROUNDS,
) -> Optional[tuple[set[int], set, set[int]]]:
    """Trade held integralities of an answer for fewer or certifiable elements,
    by counterexample-guided refinement: relax a whole block at once; a feasible
    point names the columns it needs back (those it puts off-integer, or past a
    dropped bound), which are restored, and the block is re-solved until the set
    is infeasible again.

    1. General integers (box wider than one unit, or unbounded): relaxed with
       their bounds (one element less each).
    2. Binaries: relaxed keeping both bounds, then the upper bounds are dropped,
       then the lower bounds of the columns that kept their upper one. A column
       left with both bounds is held integer again (one element, not two), so
       the size never grows; a column with one bound costs what its integrality
       did, but a bound has a certificate where, under the checker's rule (a
       relaxed integrality keeps its box), the integrality may have none.

    ``keep``: integralities not to relax (those already holding a certificate).
    Every restore only tightens the set, so an infeasible set stays infeasible;
    probes run under the engine's policy, so the caller confirms the result
    (:meth:`_Engine.confirm_infeasible`). Returns the new set, or None when it
    is not smaller-or-equal and different. Each stage is bounded by
    ``max_rounds`` solves (deterministic); reaching ``stop_at`` raises
    :class:`_OutOfTime` and the caller discards the pass. Not run on a complete
    answer: there every integrality was already deletion-tested.
    """
    c0, b0, i0 = set(cons), set(bounds), set(ints)
    lb, ub = eng.col_lb, eng.col_ub

    def probe(b: set, i: set) -> tuple[Optional[bool], Optional[np.ndarray]]:
        if time.perf_counter() >= stop_at:
            raise _OutOfTime
        return eng.probe(c0, b, i)

    def frac(x: np.ndarray, j: int) -> bool:
        return abs(float(x[j]) - round(float(x[j]))) > _RELAX_TOL

    b, i = set(b0), set(i0)
    # Stage 1: general integers with their bounds.
    gen = {j for j in i - keep if not (lb[j] > -_INF and ub[j] < _INF and ub[j] - lb[j] <= 1)}
    relaxed = set(gen)
    for _ in range(max_rounds):
        if not relaxed:
            break
        v, x = probe(b, i - relaxed)
        if v is True:
            i -= relaxed
            break
        if v is None or x is None:
            relaxed = set()
            break
        back = {
            j
            for j in relaxed
            if frac(x, j) or x[j] < lb[j] - _RELAX_TOL or x[j] > ub[j] + _RELAX_TOL
        }
        if not back:
            relaxed = set()
            break
        relaxed -= back
    # Stage 2: binaries, keeping their box first.
    binr = {j for j in i - keep if lb[j] > -_INF and ub[j] < _INF and ub[j] - lb[j] <= 1}
    box = {(j, s) for j in binr for s in ("lower", "upper")}

    def refine(cand: set, test) -> set:
        """The subset of ``cand`` that ``test(subset)`` accepts, by restoring
        the counterexample's columns; empty when undecided or stuck."""
        cur = set(cand)
        for _ in range(max_rounds):
            if not cur:
                return cur
            v, x, back = test(cur)
            if v is True:
                return cur
            if v is None or not back:
                return set()
            cur -= back
        return set()

    def t_int(cur):
        v, x = probe(b | {(j, s) for j in cur for s in ("lower", "upper")}, i - cur)
        return v, x, (set() if x is None else {j for j in cur if frac(x, j)})

    rel = refine(binr, t_int)
    if rel:
        bb = (b - box) | {(j, s) for j in rel for s in ("lower", "upper")}
        ii = i - rel

        def t_up(cur):
            v, x = probe(bb - {(j, "upper") for j in cur}, ii)
            return v, x, (set() if x is None else {j for j in cur if x[j] > ub[j] + _RELAX_TOL})

        no_up = refine(rel, t_up)
        bb -= {(j, "upper") for j in no_up}

        def t_lo(cur):
            v, x = probe(bb - {(j, "lower") for j in cur}, ii)
            return v, x, (set() if x is None else {j for j in cur if x[j] < lb[j] - _RELAX_TOL})

        no_lo = refine(rel - no_up, t_lo)
        bb -= {(j, "lower") for j in no_lo}
        # Both bounds left: hold the integrality again (tighter, one element).
        both = {j for j in rel if (j, "lower") in bb and (j, "upper") in bb}
        bb -= {(j, s) for j in both for s in ("lower", "upper")}
        b, i = bb, ii | both
    size = len(b) + len(i)
    if (b, i) == (b0, i0) or size > len(b0) + len(i0):
        return None
    return c0, b, i


def _polish(
    arrays: tuple,
    res: MilpIIS,
    *,
    cert_tol: float,
    time_limit: float,
    engine: str,
    deadline: Optional[float],
) -> MilpIIS:
    """Post-pass on the answer: on an anytime (incomplete) answer the
    integrality relaxation (:func:`_relax_integrality`, at most
    ``_RELAX_FRACTION`` of the time left, at most ``_RELAX_CAP_S``; the filter
    stops ``_POLISH_RESERVE_FRACTION`` of the budget, at most
    ``_POLISH_RESERVE_CAP_S``, early to leave it room); on a complete one the
    add-one move (:func:`_add_one`; only with an overall budget, capped at
    ``_ADD_ONE_FRACTION`` of the time left, at most ``_ADD_ONE_CAP_S``); then a
    forced-violation certificate for every kept element that still has none
    (:meth:`_Engine.forced_point`, margins :data:`_FORCED_MARGINS`).

    Soundness: a set the add-one move changed replaces the answer only when
    :meth:`_Engine.confirm_infeasible` confirms it under both presolve
    configurations, as for every answer; otherwise the answer stands unchanged.
    The same holds for a set the relaxation changed.
    A certificate is kept only when :func:`_check_certificates` accepts it by
    arithmetic on the final set. Needs ``res.bounds`` to be every bound the set
    applies, i.e. ``include_bounds=True``. ``DISCOPT_IIS_ADD_ONE=0`` /
    ``DISCOPT_IIS_CERT_REPAIR=0`` / ``DISCOPT_IIS_RELAX_INT=0`` are opt-outs for
    A/B measurement."""
    add_one = (
        os.environ.get("DISCOPT_IIS_ADD_ONE", "1") != "0" and res.complete and deadline is not None
    )
    relax = (
        os.environ.get("DISCOPT_IIS_RELAX_INT", "1") != "0"
        and not res.complete
        and deadline is not None
        and bool(res.integrality)
    )
    repair = os.environ.get("DISCOPT_IIS_CERT_REPAIR", "1") != "0"
    if not add_one and not relax and not repair:
        return res
    t0 = time.perf_counter()
    if deadline is not None and deadline - t0 < 4 * _MIN_SOLVE_S:
        return res
    A, row_lo, row_hi, row_groups, col_lb, col_ub, is_int = arrays
    eng = _Engine(
        A,
        row_lo,
        row_hi,
        col_lb,
        col_ub,
        is_int,
        row_groups,
        time_limit,
        False,
        deadline=deadline,
        reserve_confirm=False,
        engine=engine,
    )
    eng.arm_lattice(deadline)
    cons, bounds, ints = set(res.constraints), set(res.bounds), set(res.integrality)
    pending: dict[tuple, np.ndarray] = dict(res.certificates)
    proven = res.proven_irreducible
    tried = moves = n_relaxed = 0
    s_tol = min(cert_tol, _STRONG_OTHERS)
    if relax:
        assert deadline is not None
        share = min(_RELAX_FRACTION * max(deadline - t0, 0.0), _RELAX_CAP_S)
        keep = {
            k[1]
            for k in _check_certificates(pending, eng, cons, bounds, ints, s_tol, _STRONG_OWN)
            if k[0] == "integrality"
        }
        try:
            got = _relax_integrality(eng, cons, bounds, ints, keep, t0 + share)
            if got is not None:
                t_c = time.perf_counter()
                verdict = eng.confirm_infeasible(
                    *got,
                    stop_at=min(deadline, t_c + share),
                    presolve=("off",) if eng.proven_on(*got) else ("choose", "off"),
                )
                if verdict is True:
                    n_relaxed = len(ints - got[2])
                    cons, bounds, ints = got
        except _OutOfTime:
            pass  # the relaxation or its confirmation ran out of time: the answer stands
    if add_one:
        assert deadline is not None
        share = min(_ADD_ONE_FRACTION * max(deadline - t0, 0.0), _ADD_ONE_CAP_S)
        try:
            got, pend, decided, tried = _add_one(eng, cons, bounds, ints, t0 + share)
            if got is not None:
                t_c = time.perf_counter()
                verdict = eng.confirm_infeasible(
                    *got,
                    stop_at=min(deadline, t_c + share),
                    presolve=("off",) if eng.proven_on(*got) else ("choose", "off"),
                )
                if verdict is True:
                    cons, bounds, ints = got
                    pending.update(pend)
                    proven = proven and decided
                    moves = 1
        except _OutOfTime:
            pass  # the sweep or its confirmation ran out of time: the answer stands
    certs = _check_certificates(pending, eng, cons, bounds, ints, cert_tol)
    n_forced = n_forced_ok = 0
    if repair:
        strong = _check_certificates(certs, eng, cons, bounds, ints, s_tol, _STRONG_OWN)
        try:
            for key in _element_keys(cons, bounds, ints):
                if key in strong:
                    continue
                n_forced += 1
                for margin in _FORCED_MARGINS:
                    for tol in (None, 1e-9):
                        y, _ = eng.forced_point(cons, bounds, ints, key, margin, deadline, tol)
                        if y is None:
                            break
                        ok = _check_certificates(
                            {key: y}, eng, cons, bounds, ints, s_tol, _STRONG_OWN
                        )
                        if ok:
                            strong.update(ok)
                            certs.update(ok)
                            break
                    if key in strong:
                        n_forced_ok += 1
                        break
        except _OutOfTime:
            pass  # the certificates found so far stand
    return replace(
        res,
        constraints=sorted(cons),
        bounds=sorted(bounds),
        integrality=sorted(ints),
        certificates=certs,
        n_solves=res.n_solves + eng.n_solves,
        proven_irreducible=proven,
        n_forced=n_forced,
        n_forced_certified=n_forced_ok,
        n_add_one_tried=tried,
        n_add_one_moves=moves,
        n_relaxed=n_relaxed,
    )


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
#: Staged-oracle ladder (order flag ``l``): its capped-MIP rung's node limit,
#: ``max(_LADDER_MIN_NODES, _LADDER_NODE_FACTOR * base proof's nodes)``. It scales
#: with the base proof because a fixed small rung could not even prove the full
#: neos859080 or neos-1461051 (3.8 s / 7.3 s base proofs).
_LADDER_MIN_NODES = 1000
_LADDER_NODE_FACTOR = 10
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


class _EncodeAbort(Exception):
    """A row's encoding would pass the clause cap ("size") or the stage's stop
    time ("time"); raised before any variable or clause of that row exists."""


def _encode_geq(
    w: list[int],
    lits: list[int],
    K: int,
    guard: int,
    nv: int,
    clauses: list,
    stop_at: Optional[float] = None,
    max_clauses: Optional[int] = None,
):
    """Clauses for ``guard -> sum w_i l_i >= K`` as a reduced BDD (Een & Sorensson
    2006): node ``(i, k)`` means "the suffix from ``i`` reaches ``k``", with
    ``node -> ite(l_i, (i+1, k - w_i), (i+1, k))``; the constraint is monotone, so
    also ``node -> (i+1, k - w_i)``. ``k <= 0`` is true, ``k`` above the suffix's
    total is false. Built layer by layer, without recursion. Returns the new
    variable count.

    The BDD can have up to n*K nodes: one 32-column knapsack row with weights up
    to 1e6 ran a 2 s stage budget to 46 s. So the layers are sized first, the
    clock and ``max_clauses`` (at most two clauses per node, plus one) are
    checked per layer, and :class:`_EncodeAbort` is raised before anything of
    the row is created.
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
    nodes = 1
    for i in range(m):
        nxt = set()
        for k in layers[i]:
            for k2 in (k - w[i], k):
                if 0 < k2 <= suf[i + 1]:
                    nxt.add(k2)
        layers.append(nxt)
        nodes += len(nxt)
        if max_clauses is not None and len(clauses) + 2 * nodes + 1 > max_clauses:
            raise _EncodeAbort("size")
        if stop_at is not None and time.perf_counter() >= stop_at:
            raise _EncodeAbort("time")
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


def _lattice_candidate(A, row_lo, row_hi, is_int, row_groups, stop_at):
    """A small constraint set that is infeasible by the lattice test alone
    (:func:`discopt._iis_lattice.lattice_core`), or None.

    Returns ``(constraints, held integer columns, columns the rows touch)``: a
    proposal -- the caller holds those integralities, keeps the bounds of the
    touched columns, drops every other element, and has the base probe prove it
    infeasible before using it. Only constraints with a single row enter.

    On MIPcc27's smallset it exists for enlight4 (8 rows + 10 integralities),
    enlight11 (48 + 48) and neos-807454 (9 + 32; Gurobi's IIS there is those 9
    rows and 56 integralities), at 0.01-3 s; the mod-2 core it replaced gave 8,
    48 and 9 rows with every touched column held.
    """
    from discopt._iis_lattice import integer_rows, lattice_core

    rows = integer_rows(A, row_lo, row_hi)
    by_con = {
        ci: rows[int(rs[0])] for ci, rs in row_groups.items() if len(rs) == 1 and int(rs[0]) in rows
    }
    if not by_con:
        return None
    held = set(int(j) for j in np.flatnonzero(is_int))
    core = lattice_core(by_con, held, orders=_LATTICE_ORDERS, stop_at=stop_at)
    if core is None:
        return None
    cons, ints = core
    touched: set[int] = set()
    for ci in cons:
        touched.update(by_con[ci][0])
    return cons, ints, touched


#: Deletion orders :func:`_lattice_candidate` tries (deterministic seeds; the
#: smallest core is kept). Each is a few hundred lattice tests. On the smallset
#: the best cores (18 / 96 / 41) first appear at orders 3 / 4 / 1 of 0-15.
_LATTICE_ORDERS = 16


def _integral(x) -> bool:
    return bool(np.all(np.asarray(x) == np.round(x)))


def _lift_implied_binaries(A, row_lo, row_hi, col_lb, col_ub, is_int):
    """Continuous columns that may be taken binary without changing feasibility.

    Returns the set of continuous columns ``c`` for which the model stays
    feasible-equivalent when ``c`` is restricted to ``{0, 1}``, judged per row so
    that the equivalence holds for *every subset* of the rows (the property the
    CDCL core stage needs: a core of the lifted model must be infeasible as a
    subsystem of the original one). A column qualifies when ``col_lb == 0``,
    ``col_ub >= 1`` and every row it appears in is single-sided and, written as
    ``sum a_k x_k >= L``, is one of:

    - *upper row*: ``a_c == -1``, ``L`` integral, and every other column of the
      row integer with an integral coefficient. It reads ``c <= (integer
      expression)``, so rounding ``c`` up keeps it.
    - *covering row*: ``a_c > 0`` with ``a_c >= L``, and every other term
      nonnegative over its box (``a_k > 0`` with ``lb_k >= 0``, or ``a_k < 0``
      with ``ub_k <= 0``). Raising ``c`` only helps it, and ``c = 1`` alone
      satisfies it, so capping ``c`` at 1 keeps it too.

    Proof: given a feasible point, hold the integer columns and map each lifted
    ``c`` to ``min(ceil(c), 1)``. An upper row bounds ``c`` by an integer at least
    ``c``, so by at least ``ceil(c)`` and (being decreased by the cap) by the new
    value. A covering row loses activity only where some ``c > 1`` is capped, and
    then ``a_c * 1 + (nonnegative rest) >= L``. Only lifted columns move, each
    within ``[0, 1]`` so their own covering-row terms stay nonnegative; an upper
    row holds one lifted column at most (its other columns are integer). Measured
    on MIPcc27 neos-1461051 (80 continuous columns, all qualifying): the CDCL
    stage then gives a 308-row core where it declined before.
    """
    n = col_lb.size
    cont = np.flatnonzero(~is_int)
    if cont.size == 0:
        return set()
    At = A.tocsc()
    out: set[int] = set()
    for c in cont:
        c = int(c)
        if col_lb[c] != 0.0 or col_ub[c] < 1.0:
            continue
        ok = At.indptr[c + 1] > At.indptr[c]
        for p in range(At.indptr[c], At.indptr[c + 1]):
            r = int(At.indices[p])
            lo, hi = row_lo[r], row_hi[r]
            if (lo > -_INF) == (hi < _INF):  # two-sided (or free) row
                ok = False
                break
            sgn = 1.0 if lo > -_INF else -1.0
            L = lo if lo > -_INF else -hi
            s0, e0 = A.indptr[r], A.indptr[r + 1]
            cols = A.indices[s0:e0]
            vals = sgn * A.data[s0:e0]
            others = cols != c
            a_c = float(vals[~others][0])
            oc, ov = cols[others], vals[others]
            if a_c == -1.0:
                ok = _integral(L) and bool(np.all(is_int[oc])) and _integral(ov)
            elif a_c > 0.0:
                ok = a_c >= L and bool(
                    np.all(((ov > 0) & (col_lb[oc] >= 0)) | ((ov < 0) & (col_ub[oc] <= 0)))
                )
            else:
                ok = False
            if not ok:
                break
        if ok:
            out.add(c)
    assert all(0 <= c < n for c in out)
    return out


#: Relative margin of the structural sweep's activity tests on rows with
#: non-integral data (integral data is compared exactly, in Python ``int``).
_SWEEP_REL = 1e-9

#: The sweep checks the clock every this many rows or columns.
_SWEEP_CLOCK_EVERY = 256


def _structural_sweep(
    A_csr,
    A_csc,
    row_lo,
    row_hi,
    row_groups,
    col_lb,
    col_ub,
    cons,
    bounds,
    ints_active,
    include_bounds=True,
    stop_at=None,
) -> tuple[int, int, int]:
    """Drop elements of an infeasible subsystem with no solve; returns
    ``(constraints, bounds, integralities)`` removed. Mutates the three sets.

    ``stop_at`` (a ``time.perf_counter()`` value) ends the sweep early. Every
    removal made up to then is valid on its own, so an early stop is sound.

    Each step below removes constraints from the CURRENT subsystem ``S`` only
    when ``S`` minus them is still infeasible, by an exact argument about ``S``
    itself, and steps are applied one after another, each judged on the system
    the previous one left (judging several rows redundant *jointly* is unsound:
    each may be implied only by the others). The rules, applied to a fixpoint:

    * **Redundant row.** Every row of a group holds for every point of the
      current column boxes (a column's box: its declared bounds when it is held
      integer, else the bounds still in ``bounds``). Dropping it leaves the
      feasible set unchanged. Only boxes are used, never other rows.
    * **Monotone column** (pure-literal elimination). A column ``j`` and a
      direction ``d`` such that every current row containing ``j`` is one-sided
      and loosens as ``x_j`` moves along ``d``; let ``v`` be ``j``'s current
      bound in that direction. If ``v`` is infinite, or every such row holds at
      ``x_j = v`` for every point of the other columns' boxes (``v`` integral
      when ``j`` is held), then a point satisfying ``S`` minus those rows
      extends to ``S`` by moving ``x_j`` to ``v`` (or far enough), so ``S``
      minus them is infeasible. Groups are dropped only when all their rows are
      among those rows.
    * **Orphan column.** No current row contains ``j``. Its integrality and
      bounds go together, provided its box is nonempty and, if held, contains
      an integer: then any point of the rest extends by a value of ``x_j`` from
      the box. (Relaxing the integrality alone would turn its bounds into
      counted elements.)

    Removing constraints never invalidates a certificate of a kept element (a
    point feasible for ``S`` minus that element stays feasible for a subset
    of it). In exact arithmetic a sweep never drops an element whose
    certificate shows it is needed, since that would contradict the argument
    above; the engine checks this and reverts a sweep that does
    (``MilpIIS.n_sweep_reverted``). Measured on MIPcc27's dijle, whose
    3961-element answer kept 1176 orphan elements.

    Integral data is compared in Python ``int`` arithmetic: a float sum of
    integral terms is exact only below 2**53, and above it rounding can push
    an activity across the row side (``2**53 + 3`` rounds to ``2**53 + 4``),
    which would drop a needed row. Other data uses a relative margin.

    Cost: the redundant-row test depends only on column boxes, and a sweep
    changes a box only for a column no kept row touches, so one pass over the
    rows suffices. Columns go through a worklist: a column is re-examined only
    when a row containing it was dropped.
    """
    n = col_lb.size
    held = np.zeros(n, dtype=bool)
    held[list(ints_active)] = True
    has_lo = np.zeros(n, dtype=bool)
    has_hi = np.zeros(n, dtype=bool)
    for j, side in bounds:
        (has_lo if side == "lower" else has_hi)[j] = True
    finite_lo = col_lb > -_INF
    finite_hi = col_ub < _INF
    blo = np.where((held | has_lo) & finite_lo, col_lb, -math.inf)
    bhi = np.where((held | has_hi) & finite_hi, col_ub, math.inf)

    group_of: dict[int, int] = {}
    for ci in cons:
        for r in np.asarray(row_groups[ci]).ravel():
            group_of[int(r)] = ci

    def integral(x) -> bool:
        return math.isfinite(x) and float(x).is_integer()

    def holds(r: int, j: int = -1, v: float = 0.0) -> bool:
        """Row ``r`` holds on every point of the boxes (``x_j = v`` if given)."""
        s, e = A_csr.indptr[r], A_csr.indptr[r + 1]
        # Python floats: ``int >= np.float64`` converts the int to float64 and
        # rounds; ``int >= float`` compares exactly.
        L, H = float(row_lo[r]), float(row_hi[r])
        check_lo, check_hi = L > -_INF, H < _INF
        lo_act = hi_act = mag = 0.0
        ilo = ihi = 0  # exact activities while every term is integral
        exact = True
        for k, a in zip(A_csr.indices[s:e], A_csr.data[s:e]):
            if a == 0.0:
                continue
            lb, ub = (v, v) if k == j else (blo[k], bhi[k])
            # Bound giving the least / greatest term a * x_k.
            bl, bh = (lb, ub) if a > 0 else (ub, lb)
            if check_lo and not math.isfinite(bl):
                return False  # the minimum activity is -inf
            if check_hi and not math.isfinite(bh):
                return False  # the maximum activity is +inf
            if exact and integral(a):
                ia = int(a)
                if check_lo:
                    exact = integral(bl)
                    if exact:
                        ilo += ia * int(bl)
                if exact and check_hi:
                    exact = integral(bh)
                    if exact:
                        ihi += ia * int(bh)
            else:
                exact = False
            if check_lo:
                lo_act += a * bl
                mag += abs(a * bl)
            if check_hi:
                hi_act += a * bh
                mag += abs(a * bh)
        if exact:
            # int vs float comparisons are exact in Python.
            return (not check_lo or ilo >= L) and (not check_hi or ihi <= H)
        if check_lo and not lo_act >= L + _SWEEP_REL * (1.0 + mag + abs(L)):
            return False
        if check_hi and not hi_act <= H - _SWEEP_REL * (1.0 + mag + abs(H)):
            return False
        return True

    def drop_group(ci: int) -> None:
        cons.discard(ci)
        for r in np.asarray(row_groups[ci]).ravel():
            group_of.pop(int(r), None)

    n_c = n_b = n_i = 0
    ticks = 0

    def out_of_time() -> bool:
        nonlocal ticks
        ticks += 1
        return (
            stop_at is not None
            and ticks % _SWEEP_CLOCK_EVERY == 0
            and time.perf_counter() >= stop_at
        )

    if stop_at is not None and time.perf_counter() >= stop_at:
        return n_c, n_b, n_i
    # Redundant rows, judged on the boxes alone. One pass: no justification
    # uses another row, and no box of a column in a kept row changes later.
    for ci in sorted(cons):
        if out_of_time():
            return n_c, n_b, n_i
        grp = np.asarray(row_groups[ci]).ravel()
        if grp.size and all(holds(int(r)) for r in grp):
            drop_group(ci)
            n_c += 1
    # Monotone and orphan columns, one column at a time, through a worklist:
    # a column's verdict changes only when a kept row containing it is dropped.
    queue = deque(range(n))
    queued = np.ones(n, dtype=bool)

    def requeue(ci: int) -> None:
        for r in np.asarray(row_groups[ci]).ravel():
            s, e = A_csr.indptr[r], A_csr.indptr[r + 1]
            for k in A_csr.indices[s:e]:
                if not queued[k]:
                    queued[k] = True
                    queue.append(int(k))

    while queue:
        if out_of_time():
            return n_c, n_b, n_i
        j = queue.popleft()
        queued[j] = False
        s, e = A_csc.indptr[j], A_csc.indptr[j + 1]
        rows = [
            (int(r), a)
            for r, a in zip(A_csc.indices[s:e], A_csc.data[s:e])
            if a != 0.0 and int(r) in group_of
        ]
        lo_j, hi_j = blo[j], bhi[j]
        if lo_j > hi_j:
            continue  # an empty box is itself the contradiction
        if not rows:
            if not (held[j] or (include_bounds and (has_lo[j] or has_hi[j]))):
                continue  # nothing of this column is an element
            if held[j] and math.isfinite(lo_j) and math.ceil(lo_j) > hi_j:
                continue  # no integer in the box
            was_held = bool(held[j])
            if was_held:
                held[j] = False
                ints_active.discard(j)
                n_i += 1
            if include_bounds:
                for side, flag in (("lower", has_lo), ("upper", has_hi)):
                    if (j, side) in bounds:
                        bounds.discard((j, side))
                        n_b += not was_held  # a held column's box was no element
                    flag[j] = False
            blo[j] = col_lb[j] if has_lo[j] and finite_lo[j] else -math.inf
            bhi[j] = col_ub[j] if has_hi[j] and finite_hi[j] else math.inf
            continue
        for d in (-1, 1):
            ok = True
            for r, a in rows:
                lo_f, hi_f = row_lo[r] > -_INF, row_hi[r] < _INF
                if lo_f == hi_f:
                    ok = False  # two-sided (or free) row
                    break
                # A >= row loosens as a*x_j grows; a <= row as it shrinks.
                if (a * d > 0) != lo_f:
                    ok = False
                    break
            if not ok:
                continue
            v = lo_j if d < 0 else hi_j
            if math.isfinite(v):
                if held[j] and not integral(v):
                    continue
                if not all(holds(r, j, v) for r, _ in rows):
                    continue
            groups = {group_of[r] for r, _ in rows}
            mine = {r for r, _ in rows}
            if not all(
                set(int(x) for x in np.asarray(row_groups[ci]).ravel()) <= mine for ci in groups
            ):
                continue
            for ci in groups:
                drop_group(ci)
                n_c += 1
                requeue(ci)
            break
    return n_c, n_b, n_i


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
        # Encoding is part of the stage's budget too: it grows with the model.
        if stop_at is not None and time.perf_counter() >= stop_at:
            return None, 0
        for r in row_groups[ci]:
            s0, e0 = A.indptr[r], A.indptr[r + 1]
            rows = _pb_rows(A.indices[s0:e0], A.data[s0:e0], row_lo[r], row_hi[r])
            if rows is None:
                return None, 0
            for w, lits, K in rows:
                try:
                    nv = _encode_geq(w, lits, K, n + 1 + gi, nv, clauses, stop_at, _SAT_MAX_CLAUSES)
                except _EncodeAbort as why:
                    if why.args[0] == "time":
                        return None, 0
                    # Too large to encode: leave this row out. The encoded rows
                    # are exact, so a core over them is still an infeasible
                    # subset (a group with a row left out is relaxed, not lost).
                    continue
    from discopt._rust import PySatSolver

    sat = PySatSolver(nv)
    sel = [n + 1 + gi for gi in range(len(groups))]
    for c in clauses:
        sat.add_clause(c)

    def out_of_time() -> bool:
        return stop_at is not None and time.perf_counter() >= stop_at

    class _Abandon(Exception):
        """The stage's share ran out: drop the work in hand, keep nothing partial."""

    def solve_sliced(assumptions: list[int], conflicts: int) -> Optional[bool]:
        """``sat.solve`` in slices of ``_SAT_SLICE_CONFLICTS``: the verdict, or
        None when ``conflicts`` are spent undecided. Raises :class:`_Abandon` at
        ``stop_at`` -- time never turns into a verdict (#2318)."""
        spent0 = sat.stats()[0]
        while True:
            verdict: Optional[bool] = sat.solve(assumptions, _SAT_SLICE_CONFLICTS)
            if verdict is not None:
                return verdict
            if sat.stats()[0] - spent0 >= conflicts:
                return None
            if out_of_time():
                raise _Abandon

    # The clock may only choose between results this stage produces COMPLETE --
    # no candidate, the first core, or the fully refined core -- each fixed by
    # conflict counts alone (the CDCL solver is deterministic). It used to keep a
    # partly refined core when the share ran out mid-refinement, and a refinement
    # trial the clock cut short kept its element as if it were needed, so the
    # candidate's size -- and the IIS the filter starts from -- was a function of
    # machine speed (#912 role 2; #2318). That dependence is quantized, not gone:
    # which of the three results comes back still depends on whether refinement
    # finishes inside the share, so the gate stays ``residual`` in the #912
    # inventory (``test_912_wall_budget_inventory``). Measured on the in-repo IIS tests before
    # the change: 41 candidates, none changed when re-run with no time limit, so
    # this is bound-neutral there; it matters only where refinement overruns.
    try:
        if solve_sliced(sel, _SAT_FIRST_CONFLICTS) is not False:
            return None, sat.stats()[0]  # SAT, or no proof in budget: no candidate
    except _Abandon:
        return None, sat.stats()[0]  # no proof in time: no candidate
    first_core = {abs(lit) - (n + 1) for lit in sat.core()}
    rem = set(first_core)
    kept: set[int] = set()
    try:
        for gi in sorted(rem):
            if gi not in rem:
                continue
            # Checked only before a trial runs: an element already dropped
            # costs nothing, so a refinement whose last trial is done is done.
            if out_of_time():
                raise _Abandon
            trial = rem - {gi}
            verdict = solve_sliced([sel[j] for j in trial | kept], _SAT_REFINE_CONFLICTS)
            if verdict is False:
                core = {abs(lit) - (n + 1) for lit in sat.core()}
                kept &= core
                rem = core - kept
            else:  # needed (SAT) or undecided within its conflict budget: keep it
                kept.add(gi)
                rem.discard(gi)
    except _Abandon:
        # Refinement did not finish in the share: the first core, complete and
        # unsatisfiable, not the half-refined set in hand.
        return {groups[gi] for gi in first_core}, sat.stats()[0]
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
    portfolio: Optional[Sequence[str] | bool] = None,
    workers: Optional[int] = None,
    portfolio_wait: str = "fast",
    iis_engine: Optional[str] = None,
) -> MilpIIS:
    """IIS of a linear (LP/MILP) ``model`` with integrality as an element.

    ``iis_engine`` (highs-parity P3.2) names the MILP solver behind every probe:
    ``"highs"`` or ``"rust"``, discopt's own MILP driver through a
    persistent :class:`~discopt.solvers.milp_handle.MilpHandle`. ``None`` follows
    :func:`_resolve_iis_engine` (HiGHS where installed). On
    P0.6's fixtures the two agree (``DISCOPT_FARKAS_EXACT_OPEN``, default ON); the Rust
    driver still leaves some probes undecided where HiGHS decides them -- integrality
    slack on big-M rows (``INT_TOL`` 1e-5) and hard combinatorial MILPs (pigeonhole)
    -- recorded in ``docs/dev/highs-parity-plan.md`` P3.2. It was the default from
    2026-11-07 (P6.4) until the 2026-11-17 engine switch; ``iis_engine="rust"`` /
    ``DISCOPT_IIS_ENGINE=rust`` selects it. The deletion
    filter, the #1634 presolve-on / presolve-off cross-checks, the witness repair and the
    confirmations are the same code either way (``discopt._iis_backend``).

    ``portfolio_wait`` (portfolio runs only): ``"fast"`` stops the other members
    shortly after one completes, ``"all"`` waits for every member (the smallest
    set, at up to the whole budget); see :data:`PORTFOLIO_WAITS`.

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

    ``sat_candidate``: start from a small candidate subsystem instead of the
    full model -- a minimal lattice (integer Farkas) certificate over integer
    equality rows (:func:`_lattice_candidate`) when one exists, else, on a pure-binary model with
    integral coefficients, an unsatisfiable constraint set found by discopt's
    CDCL solver (:func:`_binary_core`). Either is only a proposal: it is used if
    HiGHS proves it infeasible (the base probe), otherwise the full model is
    used, and the returned subsystem is confirmed and certified exactly as
    without it.

    Raises ``NotImplementedError`` (from the linear extractor) for a model with a
    nonlinear or non-algebraic constraint, ``ValueError`` when the full model is
    not proven infeasible, and ``TimeoutError`` as above.
    """
    iis_engine = _resolve_iis_engine(iis_engine)
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
    portfolio = _normalize_portfolio(portfolio)
    if portfolio_wait not in PORTFOLIO_WAITS:
        raise ValueError(
            f"portfolio_wait must be one of {sorted(PORTFOLIO_WAITS)}, got {portfolio_wait!r}"
        )
    grace, grace_window = PORTFOLIO_WAITS[portfolio_wait] if portfolio else (math.inf, None)
    return milp_iis_arrays(
        arrays,
        include_bounds=include_bounds,
        time_limit=time_limit,
        cert_tol=cert_tol,
        max_batch=max_batch,
        confirm=confirm,
        total_time_limit=total_time_limit,
        sat_candidate=sat_candidate,
        order=order,
        portfolio=portfolio,
        workers=workers,
        t_start=t_start,
        grace=grace,
        grace_window=grace_window,
        iis_engine=iis_engine,
    )


def milp_iis_arrays(
    arrays: tuple,
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
    t_start: Optional[float] = None,
    grace: float = math.inf,
    grace_window: Optional[tuple[float, float]] = None,
    iis_engine: Optional[str] = None,
    threads: Optional[int] = None,
) -> MilpIIS:
    """:func:`milp_iis` on a model already in matrix form, for callers that have
    no :class:`Model` (``discopt.mipcc`` reads an MPS file straight into arrays).

    ``threads`` (portfolio only): the portfolio's thread budget, shared out among the
    members; see :func:`_portfolio_iis`.

    ``arrays`` is ``(A, row_lo, row_hi, row_groups, col_lb, col_ub, is_int)``:
    a CSR matrix; row bounds with ``±_INF`` (1e20) for "none"; ``row_groups``
    mapping each constraint element's id to the rows it owns (one IIS element
    per key); column bounds with the same sentinel; and the integrality mask.
    Element ids in the result are the ``row_groups`` keys and column indices.
    ``t_start`` (``time.perf_counter()``) is when the budget began, so time the
    caller spent before the call counts against ``total_time_limit``.
    ``grace`` / ``grace_window`` (portfolio only): see :func:`_portfolio_iis`;
    the default waits for every member.
    """
    iis_engine = _resolve_iis_engine(iis_engine)
    if t_start is None:
        t_start = time.perf_counter()
    portfolio = _normalize_portfolio(portfolio)
    kwargs: dict[str, Any] = dict(
        include_bounds=include_bounds,
        time_limit=time_limit,
        cert_tol=cert_tol,
        max_batch=max_batch,
        confirm=confirm,
        sat_candidate=sat_candidate,
        engine=iis_engine,
    )
    from discopt._iis_backend import ENGINES

    if iis_engine not in ENGINES:
        raise ValueError(f"iis_engine={iis_engine!r}: expected one of {ENGINES}")
    if workers is not None and portfolio is None:
        raise ValueError("workers= needs portfolio=; a single order runs in-process")
    # The filter stops early enough to leave the post-pass its share
    # (_POLISH_RESERVE_FRACTION, saturating at _POLISH_RESERVE_CAP_S); the
    # post-pass runs against the whole budget.
    filter_limit = total_time_limit
    if (
        total_time_limit is not None
        and include_bounds
        and os.environ.get("DISCOPT_IIS_RELAX_INT", "1") != "0"
    ):
        reserve = min(_POLISH_RESERVE_FRACTION * float(total_time_limit), _POLISH_RESERVE_CAP_S)
        filter_limit = float(total_time_limit) - reserve
    if portfolio is not None:
        res = _portfolio_iis(
            arrays,
            kwargs,
            portfolio,
            workers,
            t_start,
            filter_limit,
            grace=grace,
            grace_window=grace_window,
            threads=threads,
        )
    else:
        if grace != math.inf or grace_window is not None:
            raise ValueError("grace= / grace_window= need portfolio=")
        if threads is not None:
            raise ValueError("threads= needs portfolio=; a single order runs in-process")
        res = _milp_iis_arrays(
            arrays, **kwargs, order=order, total_time_limit=filter_limit, t_start=t_start
        )
    if not include_bounds:
        return res  # the post-pass needs every applied bound listed (see _polish)
    return _polish(
        arrays,
        res,
        cert_tol=cert_tol,
        time_limit=time_limit,
        engine=iis_engine,
        deadline=None if total_time_limit is None else t_start + float(total_time_limit),
    )


def _parse_order(order: str) -> tuple[str, bool, bool, bool]:
    """``"ICB"`` -> (``"ICB"``, reverse=False, sat=True, ladder=False).

    A permutation of ``"ICB"`` (phase order), optionally followed by ``"-"`` and
    flags: ``r`` visits elements in reverse, ``n`` skips the candidate stages, ``l``
    runs the staged-oracle ladder (:func:`_milp_iis_arrays`) before filtering.
    E.g. ``"CIB-r"``, ``"ICB-n"``, ``"CIB-l"``.
    """
    phases, _, flags = order.partition("-")
    if (
        sorted(phases) != ["B", "C", "I"]
        or not set(flags) <= {"r", "n", "l"}
        or len(set(flags)) != len(flags)
        or ("-" in order and not flags)
    ):
        raise ValueError(
            "order must be a permutation of 'ICB' with optional '-' flags "
            f"('r' reverse, 'n' no candidate stage, 'l' oracle ladder), got {order!r}"
        )
    return phases, "r" in flags, "n" not in flags, "l" in flags


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
    engine: str = "rust",
) -> MilpIIS:
    """:func:`milp_iis` on the extracted matrix, with one deletion order."""
    if t_start is None:
        t_start = time.perf_counter()
    deadline = None if total_time_limit is None else t_start + float(total_time_limit)
    phase_order, reverse, order_sat, use_ladder = _parse_order(order)
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
        engine=engine,
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
    # The structural sweep (``_structural_sweep``): solve-free deletion between
    # phases and on the answer. ``DISCOPT_IIS_SWEEP=0`` is the opt-out for A/B.
    use_sweep = os.environ.get("DISCOPT_IIS_SWEEP", "1") != "0"
    # Two-pass integrality phase (joint relaxation first, integrality-only
    # second). ``DISCOPT_IIS_INT_TWO_PASS=0`` restores the single pass.
    two_pass = os.environ.get("DISCOPT_IIS_INT_TWO_PASS", "1") != "0"
    n_swept = 0
    n_sweep_reverted = 0
    A_csr = A.tocsr() if use_sweep else None
    A_csc = A.tocsc() if use_sweep else None

    def sweep(stop_at: Optional[float] = None) -> None:
        """Sweep the current sets in place. A sweep that would drop an element
        holding a certificate is undone whole: the certificate and the sweep's
        exact argument contradict each other, which only tolerances can cause,
        and the pre-sweep set is the one the filter established."""
        nonlocal n_swept, n_sweep_reverted
        if not use_sweep:
            return
        before = (set(cons), set(bounds), set(ints_active))
        removed = _structural_sweep(
            A_csr,
            A_csc,
            row_lo,
            row_hi,
            row_groups,
            col_lb,
            col_ub,
            cons,
            bounds,
            ints_active,
            include_bounds=include_bounds,
            stop_at=deadline if stop_at is None else stop_at,
        )
        if not sum(removed):
            return
        dropped: set[tuple] = {("constraint", c) for c in before[0] - cons}
        dropped |= {("bound", j, side) for j, side in before[1] - bounds}
        dropped |= {("integrality", j) for j in before[2] - ints_active}
        if any(k in pending for k in dropped):
            cons.clear()
            cons.update(before[0])
            bounds.clear()
            bounds.update(before[1])
            ints_active.clear()
            ints_active.update(before[2])
            n_sweep_reverted += 1
            return
        n_swept += sum(removed)

    sat_rows: Optional[int] = None
    sat_conflicts = 0
    parity_rows: Optional[int] = None
    full_state = (set(cons), set(bounds), set(ints_active))
    if eng.arm_lattice(deadline) and sat_candidate:
        pc = _lattice_candidate(A, row_lo, row_hi, is_int, row_groups, deadline)
        if pc is not None:
            # The certificate needs only its rows and the integralities the
            # core holds. The bounds of the columns the rows touch are kept as
            # well (elements, filtered like any other): dropping them up front
            # left relaxed columns unbounded, and their probe points then failed
            # the 1e-6 certificate check (neos-807454: 9 of 65 members certified
            # instead of 65).
            cons = set(pc[0])
            ints_active &= pc[1]
            bounds = {jb for jb in bounds if jb[0] in pc[2]}
            parity_rows = len(cons)
    if sat_candidate and parity_rows is None:
        # The clause-core stage gets a share of an overall budget that SATURATES
        # at a constant ceiling (#1153): where it pays off it needs seconds, so a
        # bigger budget must buy filtering, not a longer give-up on pigeonhole.
        sat_budget = (
            _SAT_STAGE_CAP_S
            if deadline is None
            else min(_SAT_STAGE_FRACTION * float(total_time_limit or 0.0), _SAT_STAGE_CAP_S)
        )
        # The share starts when the stage does, as every later stage's does
        # (``min(deadline, now + budget)``), and the overall deadline still caps
        # it. It used to start at ``t_start``, which charged model extraction,
        # first-call imports and the lattice candidate to it: where those outlasted
        # the share, the stage began past its own deadline and did no work at
        # all, so whether the candidate ran depended on machine speed (#2318,
        # CI on PR #2311; #912 role 2).
        sat_stop = time.perf_counter() + sat_budget
        if deadline is not None:
            sat_stop = min(sat_stop, deadline)
        # Continuous columns that are binary in effect (see
        # _lift_implied_binaries) are handed to the CDCL stage as binaries. Only
        # the proposal sees them so: the base probe below proves the candidate
        # on the ORIGINAL types, and the filter runs on them.
        lifted = _lift_implied_binaries(A, row_lo, row_hi, col_lb, col_ub, is_int)
        sat_is_int, sat_ub = is_int, col_ub
        if lifted:
            idx = np.fromiter(lifted, dtype=np.int64)
            sat_is_int = is_int.copy()
            sat_is_int[idx] = True
            sat_ub = col_ub.copy()
            sat_ub[idx] = 1.0
        cand, sat_conflicts = _binary_core(
            A, row_lo, row_hi, col_lb, sat_ub, sat_is_int, row_groups, sat_stop
        )
        if cand is not None and len(cand) < len(cons):
            cons = set(cand)
            sat_rows = len(cand)

    # The base proof is mandatory, so under an overall budget it may use all of
    # it rather than one probe's ``time_limit`` (no-ip-65059 needs > 30 s).
    eng.probe_tl = math.inf if deadline is not None else time_limit
    try:
        base, _ = eng.probe(cons, bounds, ints_active)
        if base is not True and (sat_rows is not None or parity_rows is not None):
            # HiGHS did not prove the candidate infeasible: discard it and take
            # the base proof on the full model, exactly as without it.
            cons, bounds, ints_active = (set(x) for x in full_state)
            sat_rows = parity_rows = None
            base, _ = eng.probe(cons, bounds, ints_active)
    except _OutOfTime:
        raise TimeoutError(
            "compute_iis: total_time_limit ran out before the model was proven infeasible"
        ) from None
    eng.probe_tl = time_limit
    base_nodes = eng.backend.last_nodes()
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
                roll()
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

    def batch_filter_columns(cols: list, joint_only: bool = False, first: int = 1) -> list:
        """The integrality phase with each column's finite bounds removed
        together with its integrality ("relax the column").

        Relaxing an integrality alone turns the column's bounds into counted
        elements, so mid-phase the subsystem grows and no anytime answer can
        follow the progress: on MIPcc27's dijle at 3600 s the filter had
        certified Gurobi's 825 rows and 375 of its 376 integralities, yet the
        answer stayed at 3961 because every set on the way counted more. Here a
        column that loses its integrality usually loses its bounds in the same
        probe (up to three elements fewer). When the joint removal is refuted
        for a single column, the integrality alone is tried (the previous
        step), so nothing the old phase dropped is kept; its bounds then stay
        for the bound phase. Same adaptive batching as :func:`batch_filter`.

        With ``joint_only`` that single-column integrality-only probe is not
        made: the column is returned (in order) for a later pass, so every
        probe of this pass can only shrink the counted size. ``first`` is the
        size of the opening batch.
        """
        nonlocal proven
        deferred: list = []

        def relax(j) -> list:
            ints_active.discard(j)
            gone = [jb for jb in ((j, "lower"), (j, "upper")) if jb in bounds]
            for jb in gone:
                bounds.discard(jb)
            return gone

        i, size = 0, max(1, min(first, cap))
        while i < len(cols):
            batch = cols[i : i + size]
            removed = {j: relax(j) for j in batch}
            try:
                res, x = eng.probe(cons, bounds, ints_active)
            except _OutOfTime:
                for j, gone in removed.items():  # undecided: back into the subsystem
                    ints_active.add(j)
                    bounds.update(gone)
                raise
            if res is True:
                i += len(batch)
                size = min(2 * size, cap)
                roll()
                continue
            for j, gone in removed.items():
                ints_active.add(j)
                bounds.update(gone)
            if len(batch) > 1:
                size = max(1, len(batch) // 2)
                continue
            # One column: its bounds may be what is needed -- try the
            # integrality alone, as the previous phase did (or, in the joint
            # pass, leave that to the next pass).
            j = batch[0]
            if joint_only and res is not None:
                deferred.append(j)
                i += 1
                continue
            if removed[j]:
                ints_active.discard(j)
                try:
                    res, x = eng.probe(cons, bounds, ints_active)
                except _OutOfTime:
                    ints_active.add(j)
                    raise
                if res is True:
                    i += 1
                    roll()
                    continue
                ints_active.add(j)
            if res is None:
                proven = False
            else:
                assert x is not None
                pending[("integrality", j)] = x
            i += 1
        return deferred

    def phase_integrality() -> None:
        # Guieu & Chinneck: if the LP relaxation is infeasible, all integrality
        # goes in one probe.
        if not ints_active:
            return
        res, _ = eng.probe(cons, bounds, set())
        if res is True:
            ints_active.clear()
        elif include_bounds and two_pass:
            # Pass 1 relaxes columns jointly (integrality and both bounds), so
            # each success shrinks the counted size and the rolling checkpoint
            # follows it. It opens with every general-integer column (domain
            # wider than {lb, lb + 1}) in one probe: on MIPcc27's dijle one
            # joint relaxation of the 384 remaining ones took the subsystem
            # from 2785 to 2401 (0.9 s), while one-at-a-time from binaries
            # never reached them. Pass 2 tries integrality alone on what pass
            # 1 refused, batched; its bounds then go to the bound phase.
            cols = sorted(ints_active)
            gen = [j for j in cols if not (col_ub[j] - col_lb[j] <= 1)]
            rest = [j for j in cols if col_ub[j] - col_lb[j] <= 1]
            deferred = batch_filter_columns(
                _ord(gen) + _ord(rest), joint_only=True, first=max(1, len(gen))
            )
            batch_filter(deferred, ints_active, lambda j: ("integrality", j))
        elif include_bounds:
            batch_filter_columns(_ord(sorted(ints_active)))
        else:
            # Bounds are not elements: they always apply, only integrality goes.
            batch_filter(_ord(sorted(ints_active)), ints_active, lambda j: ("integrality", j))

    def phase_constraints() -> None:
        batch_filter(_ord(sorted(cons)), cons, lambda ci: ("constraint", ci))

    # Bounds already put through the deletion filter. A bound of a column held
    # integer is not an element while it is held; if a later I phase releases the
    # column (orders with B before I), its bounds become elements of the result and
    # must be filtered too, or a removable bound is reported as proven irreducible.
    tested_bounds: set[tuple[int, str]] = set()

    def untested_bounds() -> list[tuple[int, str]]:
        return [
            jb
            for jb in finite_bounds
            if jb in bounds and jb[0] not in ints_active and jb not in tested_bounds
        ]

    def phase_bounds() -> None:
        # Bounds of columns not held integer (a held column's box is not an element).
        if include_bounds:
            todo = untested_bounds()
            batch_filter(_ord(todo), bounds, lambda jb: ("bound", jb[0], jb[1]))
            tested_bounds.update(todo)

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
            # The redo runs on the engine the caller chose; it used to fall back to the
            # function default (then "highs") -- a silent engine switch, and a
            # MissingHighsError without highspy (PR #2196 review).
            engine=engine,
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
    # Budget of a candidate set's confirmation (checkpoint 0, the ladder's set):
    # a share of the caller's budget, saturating at a constant (#1153).
    base_budget = min(_BASE_CONFIRM_FRACTION * float(total_time_limit or 0.0), _BASE_CONFIRM_CAP_S)
    if confirm == "final" and deadline is not None:
        # Checkpoint 0: the base set, proven just above (presolve-on; or, when
        # presolve-on answered kUnboundedOrInfeasible, by the presolve-free
        # re-solve the probe ran), confirmed presolve-off too. Without it a
        # budget that runs out before the first phase ends has nothing confirmed
        # to return (TimeoutError), although the base set is a valid -- if large
        # -- infeasible subsystem. Capped so a slow presolve-free solve cannot
        # eat the filter's budget.
        base_stop = min(deadline, time.perf_counter() + base_budget)
        n_recent = len(eng._recent)
        t_ck0 = time.perf_counter()
        try:
            verdict0 = eng.confirm_infeasible(
                cons, bounds, ints_active, stop_at=base_stop, presolve=("off",)
            )
        except _OutOfTime:
            verdict0 = None
        # Not a filtering probe: keep its duration out of the confirmation
        # reserve (2.5x the slowest recent solve), or a checkpoint solve that
        # used its whole cap would hold back ~62% of the budget from the filter.
        while len(eng._recent) > n_recent:
            eng._recent.pop()
        if verdict0 is True:
            checkpoint = (set(cons), set(bounds), set(ints_active))
            eng.confirm_ref = max(eng.confirm_ref, time.perf_counter() - t_ck0)
        elif verdict0 is False:
            return redo_each()

    def reported_size(c, b, i) -> int:
        """Size as the result reports it (and the competition counts it): rows,
        held integralities, and the bounds of columns NOT held integer -- a held
        column's box is not an element. Relaxing an integrality therefore turns
        its finite bounds into elements: mid-way through the integrality phase
        the subsystem is usually LARGER than the base set (dijle: 8001 vs 6282)."""
        return len(c) + len(i) + sum(1 for jb in b if jb[0] not in i)

    roll_ref = reported_size(cons, bounds, ints_active)  # at the last attempt / checkpoint
    roll_spent = 0.0  # seconds all rolling attempts used
    roll_last_ok = 0.0  # duration of the last successful one
    n_rolling = 0  # successful rolling checkpoints

    def roll() -> None:
        """A rolling checkpoint: confirm the current subsystem presolve-free and
        keep it as the anytime answer. The filter holds the last set a
        presolve-on probe proved, so only the second route runs. Without these
        a run stopped mid-phase fell back to the last phase-end checkpoint --
        inside the first phase, checkpoint 0 (the base set). Only a set smaller
        than the current anytime answer, as :func:`reported_size` counts it, is
        attempted, so the integrality phase (where the count grows) rarely
        triggers one; the row and bound phases do."""
        nonlocal checkpoint, roll_ref, roll_spent, roll_last_ok, n_rolling
        if confirm != "final" or deadline is None:
            return
        size = reported_size(cons, bounds, ints_active)
        if size > _ROLL_SHRINK * roll_ref:
            return  # not 10% smaller than the last attempt: not worth a solve
        if checkpoint is not None and size >= reported_size(*checkpoint):
            return
        now = time.perf_counter()
        if roll_spent > _ROLL_SHARE * (now - t_start):
            return
        if not eng.proven_on(cons, bounds, ints_active):
            return
        roll_ref = size
        unspent = _ROLL_SHARE * (now - t_start) - roll_spent
        budget = min(base_budget, max(_ROLL_MIN_S, 4.0 * eng.confirm_ref, unspent))
        n_recent = len(eng._recent)
        try:
            verdict = eng.confirm_infeasible(
                cons,
                bounds,
                ints_active,
                stop_at=min(deadline, now + budget),
                presolve=("off",),
            )
        except _OutOfTime:
            verdict = None
        finally:
            while len(eng._recent) > n_recent:  # not a filtering probe (see checkpoint 0)
                eng._recent.pop()
            roll_spent += time.perf_counter() - now
        if verdict is True:
            checkpoint = (set(cons), set(bounds), set(ints_active))
            roll_last_ok = time.perf_counter() - now
            eng.confirm_ref = max(eng.confirm_ref, roll_last_ok)
            n_rolling += 1
        elif verdict is False:
            raise _Refuted

    ladder_calls = 0
    ladder_size: Optional[int] = None
    ladder_rejected = False
    if use_ladder:
        # Staged-oracle ladder: the deletion filter run first with oracles that
        # are cheap but incomplete -- HiGHS presolve alone, then a MIP capped at
        # a multiple of the measured base proof -- dropping an element only on a
        # proven Infeasible and keeping it otherwise. The filter below then runs
        # on what is left, so the result is decided, confirmed and certified
        # exactly as without the ladder. On big-M models the engine's ordinary
        # probes stall after integralities are relaxed, while presolve proves
        # the subsystems that matter in milliseconds (MIPcc27's kedron: the
        # presolve rung alone gives Gurobi's 12-element IIS in 3.4 s).
        # Only the overall deadline stops it (when to stop, #912): its work is
        # bounded by the element count and the rung's node limit.
        n1 = max(_LADDER_MIN_NODES, _LADDER_NODE_FACTOR * base_nodes)

        def oracle(rung: Optional[int]) -> bool:
            nonlocal ladder_calls
            ladder_calls += 1
            if rung is None:
                return eng.presolve_proves(cons, bounds, ints_active)
            return eng.capped_proves(cons, bounds, ints_active, rung)

        def ladder_filter(cands: list, active: set, rung: Optional[int]) -> None:
            i, size = 0, 1
            while i < len(cands):
                batch = cands[i : i + size]
                for e in batch:
                    active.discard(e)
                try:
                    ok = oracle(rung)
                except _OutOfTime:
                    for e in batch:
                        active.add(e)
                    raise
                if ok:
                    i += len(batch)
                    size *= 2
                    continue
                for e in batch:
                    active.add(e)
                if len(batch) > 1:
                    size = max(1, len(batch) // 2)
                else:
                    i += 1

        def ladder_phase(ph: str, rung: Optional[int]) -> None:
            if ph == "C":
                ladder_filter(_ord(sorted(cons)), cons, rung)
            elif ph == "I":
                ladder_filter(_ord(sorted(ints_active)), ints_active, rung)
            elif include_bounds:
                ladder_filter(
                    _ord(sorted(jb for jb in finite_bounds if jb[0] not in ints_active)),
                    bounds,
                    rung,
                )

        n_before = len(cons) + len(bounds) + len(ints_active)
        pre_ladder = (set(cons), set(bounds), set(ints_active))
        try:
            for rung in (None, n1):
                # A rung that cannot prove the current set would fail on every
                # element (dijle: presolve alone cannot, ~3000 failed calls).
                if not oracle(rung):
                    continue
                for ph in phase_order:
                    ladder_phase(ph, rung)
        except _OutOfTime:
            pass  # every drop so far was proven: the current set is a candidate
        ladder_size = len(cons) + len(bounds) + len(ints_active)
        if ladder_size < n_before:
            # Every ladder drop rests on ONE presolve-on claim (presolve alone or a
            # node-capped presolve-on MIP), never cross-checked -- the #1634 failure
            # mode. Like the SAT candidate, the ladder's set is only a candidate:
            # it is used only once confirmed infeasible under both presolve
            # configurations, in every confirm mode. Otherwise (refuted, or not
            # decided in time) its drops are discarded and the filter starts from
            # the pre-ladder set, which is already confirmed. Not redo_each(): that
            # would rerun the same ladder (PR #2114 review).
            stop_l = None
            if deadline is not None:
                stop_l = min(deadline, time.perf_counter() + base_budget)
            try:
                verdict_l = eng.confirm_infeasible(cons, bounds, ints_active, stop_at=stop_l)
            except _OutOfTime:
                verdict_l = None
            if verdict_l is True:
                if confirm == "final":
                    # A smaller confirmed fallback than checkpoint 0.
                    checkpoint = (set(cons), set(bounds), set(ints_active))
            else:
                cons, bounds, ints_active = (set(c) for c in pre_ladder)
                ladder_rejected = True

    complete = True
    try:
        by_name = {"I": phase_integrality, "C": phase_constraints, "B": phase_bounds}
        phases = [by_name[c] for c in phase_order]
        if include_bounds and phase_order.index("B") < phase_order.index("I"):
            # B runs before I: bounds of columns I releases are filtered at the end.
            phases.append(phase_bounds)
        sweep()
        for k, phase in enumerate(phases):
            phase()
            sweep()
            if confirm == "final" and k < len(phases) - 1:
                if checkpoint is not None and reported_size(
                    cons, bounds, ints_active
                ) >= reported_size(*checkpoint):
                    # Not smaller than the anytime answer (after the integrality
                    # phase it is usually larger): nothing to gain from it.
                    continue
                # Its own budget, like checkpoint 0's: cut to the 30 s probe
                # limit it failed on dijle's 825-row set (needs 15-40 s), and
                # only the presolve-free route when presolve-on is on record.
                t_pe = time.perf_counter()
                n_recent_pe = len(eng._recent)
                try:
                    verdict = eng.confirm_infeasible(
                        cons,
                        bounds,
                        ints_active,
                        stop_at=None if deadline is None else min(deadline, t_pe + base_budget),
                        presolve=("off",)
                        if eng.proven_on(cons, bounds, ints_active)
                        else ("choose", "off"),
                    )
                except _OutOfTime:
                    # Out of THIS confirmation's budget: undecided, like roll()'s
                    # and checkpoint 0's. Only the overall deadline ends the
                    # filter -- before, a phase-end confirmation that used up its
                    # own budget skipped every remaining phase (#2145 review).
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise
                    verdict = None
                finally:
                    while len(eng._recent) > n_recent_pe:  # not a filtering probe
                        eng._recent.pop()
                if verdict is False:
                    return redo_each()
                if verdict is True:
                    checkpoint = (set(cons), set(bounds), set(ints_active))
                    eng.confirm_ref = max(eng.confirm_ref, time.perf_counter() - t_pe)
    except _OutOfTime:
        complete = False
        proven = False
    except _Refuted:
        return redo_each()

    if confirm == "final":
        eng.reserve_confirm = False  # the reserve is for this confirmation
        # The set's presolve-on proof is usually on record already (the last
        # batch the filter could not drop was put back), so only the
        # presolve-free route runs; and it may use the reserve up to the
        # deadline instead of one probe's limit. Measured (MIPcc27, 600 s): this
        # did NOT rescue dijle / dingo / dives -- with 80-92 s instead of 30 s
        # their final sets still did not confirm presolve-free (the hypothesis
        # that the 30 s cap was the cause was falsified). It removes a solve
        # that repeated a proof on record and gives the reserve its purpose;
        # anytime progress on that class comes from rolling checkpoints (``roll``).
        #
        # An unfinished run answers with the SMALLEST confirmed set, not the
        # latest: mid-way through the integrality phase the current set counts
        # more elements than the base set (relaxed integralities turn bounds
        # into elements). Confirming it anyway made dijle answer 8005 instead of
        # checkpoint 0's 6282 at 300 s, once this confirmation began to succeed.
        if (
            not complete
            and checkpoint is not None
            and reported_size(cons, bounds, ints_active) >= reported_size(*checkpoint)
        ):
            verdict = None  # not worth confirming: the checkpoint is the answer
        else:
            try:
                verdict = eng.confirm_infeasible(
                    cons,
                    bounds,
                    ints_active,
                    stop_at=deadline,
                    presolve=("off",)
                    if eng.proven_on(cons, bounds, ints_active)
                    else ("choose", "off"),
                )
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

    # The answer itself, confirmed or the fallback checkpoint: a run stopped in
    # the integrality phase leaves columns no kept row touches (dijle: 1176
    # elements). Dropping them needs no solve and keeps every certificate.
    # The deadline has usually fired by now, so this sweep runs with no clock:
    # its work is finite and fixed by the data (one pass over the rows; a column
    # is re-queued only when one of its rows is dropped, so at most n + nnz column
    # visits), and it decides nothing by machine speed (#912). Measured on the
    # 50 MIPcc27 smallset models: a whole-model sweep takes at most 0.09 s
    # (neos-3135947-otra, 139k nonzeros).
    sweep(stop_at=math.inf)
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
        parity_rows=parity_rows,
        order=order,
        n_exact=eng.n_exact,
        ladder_size=ladder_size,
        ladder_calls=ladder_calls,
        ladder_rejected=ladder_rejected,
        n_rolling=n_rolling,
        n_reduced_confirmed=eng.n_reduced_confirmed,
        n_exact_confirmed=eng.n_exact_confirmed,
        n_lattice=eng.n_lattice,
        n_exact_refuted=eng.n_exact_refuted,
        n_swept=n_swept,
        n_sweep_reverted=n_sweep_reverted,
    )


#: The recommended portfolio (``compute_iis(portfolio=DEFAULT_PORTFOLIO)``).
#: Chosen on MIPcc27's smallset (50 instances x 4 orders, 60 s, every output
#: verified): as a portfolio waiting for all members, ICB+CIB+ICB-r completes 31
#: instances, 29 of them at or below Gurobi's reference size, against 29 / 23 for
#: ICB alone; adding CIB-r changed nothing, and stopping members 2x after the
#: first completion lost size on 2 instances.
#: With the SAT core extended to PB rows, ``ICB-n`` (no core) joins: the core
#: steers some pure-binary models to a larger many-integralities IIS while an
#: almost-LP one is smaller (domset-hexgrid 446* -> 176). Re-measured on the
#: final stack (50 x 6 orders incl. ICB-n and the oracle ladder CIB-l, 60 s):
#: ICB+CIB+ICB-r+ICB-n completes 35, 33 of them at or below Gurobi (ICB alone:
#: 31 / 26); adding CIB-r or CIB-l completes nothing more.
#: ``CIB-l`` (the oracle ladder) joins for the big-M class, where the plain
#: filter's probes stall and presolve proves the small subsystem: at a 120 s
#: budget neos-3135947-otra goes from 884 (incomplete, CIB) to 23 = Gurobi's size,
#: complete; kedron 12 in 10 s vs 14.5 s. In the 60 s sweep it changed no
#: completion (its exact confirmations need ~20 s each on otra), so it costs a
#: core there and nothing else.
DEFAULT_PORTFOLIO: tuple[str, ...] = ("ICB", "CIB", "ICB-r", "ICB-n", "CIB-l")


def _normalize_portfolio(portfolio: Any) -> Optional[tuple[str, ...]]:
    """Validate the public ``portfolio=`` argument and return it as a tuple (#2206).

    ``True`` means :data:`DEFAULT_PORTFOLIO` and ``False`` means no portfolio
    (same as ``None``): both are the natural first guess for a switch-like
    argument. A bare ``str`` is refused -- it would iterate as one-letter
    "orders" (``"ICB"`` -> ``"I", "C", "B"``) -- as is anything that is not an
    iterable of strings. Order *syntax* is checked later by :func:`_parse_order`.
    """
    if portfolio is None or portfolio is False:
        return None
    if portfolio is True:
        return DEFAULT_PORTFOLIO
    hint = (
        "portfolio must be a sequence of order strings such as ['ICB', 'CIB-r'] "
        "(or True for DEFAULT_PORTFOLIO)"
    )
    if isinstance(portfolio, (str, bytes)):
        raise TypeError(f"{hint}, got the bare string {portfolio!r}; wrap it: [{portfolio!r}]")
    try:
        orders = tuple(portfolio)
    except TypeError:
        raise TypeError(f"{hint}, got {type(portfolio).__name__} {portfolio!r}") from None
    bad = [o for o in orders if not isinstance(o, str)]
    if bad:
        raise TypeError(f"{hint}, got non-string member(s) {bad!r}")
    return orders


#: Wall time held back from the members' budget for process start-up and for
#: returning their results, capped at this fraction of the budget.
_PORTFOLIO_MARGIN_S = 2.0
_PORTFOLIO_MARGIN_FRACTION = 0.05


def _iis_size(res: MilpIIS) -> int:
    return len(res.constraints) + len(res.bounds) + len(res.integrality)


#: The parent's discopt package directory, which every portfolio member must load.
_DISCOPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    # The member must run the parent's discopt, not one that happens to come
    # first on a rebuilt sys.path (CLAUDE.md s8).
    import discopt

    here = os.path.dirname(os.path.abspath(discopt.__file__))
    if here != job["discopt_dir"]:
        raise RuntimeError(
            f"portfolio member imported discopt from {here}, parent uses {job['discopt_dir']}"
        )

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
    # Before any solve: this member's share of the caller's thread budget, not
    # HiGHS's automatic scheduler (see pin_highs_threads).
    from discopt._iis_backend import pin_highs_threads

    pin_highs_threads(int(job["highs_threads"]))
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


def _grace_extra(
    t_done: float, grace: float, window: Optional[tuple[float, float]] = None
) -> float:
    """Seconds the other portfolio members may keep running after the first one
    completes at elapsed time ``t_done``: ``(grace - 1) * t_done``, clamped to
    ``window`` when given (see :func:`_portfolio_iis`)."""
    extra = (grace - 1.0) * t_done
    if window is not None:
        extra = min(max(extra, float(window[0])), float(window[1]))
    return extra


def _portfolio_iis(
    arrays: tuple,
    kwargs: dict,
    portfolio: Sequence[str],
    workers: Optional[int],
    t_start: float,
    total_time_limit: Optional[float],
    grace: float = math.inf,
    grace_window: Optional[tuple[float, float]] = None,
    threads: Optional[int] = None,
) -> MilpIIS:
    """Run one deletion filter per order in ``portfolio``, each in its own process,
    against one shared budget, and return the smallest complete result.

    ``threads``: the thread budget for the whole portfolio (default: every core).
    Each member's HiGHS scheduler gets ``max(1, threads // workers)`` threads, set
    before its first solve (:func:`discopt._iis_backend.pin_highs_threads`).

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
    ``grace_window=(lo, hi)`` clamps the extra wait ``(grace - 1) * t`` to
    ``[lo, hi]`` seconds: on MIPcc27's 6-order sweep every later member that beat
    the first finisher did so within 31 s of it, whatever ``t`` was, so a
    proportional wait cut off improvements on fast instances (bell3ainf: first
    at 4 s, Gurobi's size at 17 s) and idled for minutes on slow ones
    (neos859080: complete at ~285 s, waited to ~570 s).
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
    budget_threads = (os.cpu_count() or 1) if threads is None else int(threads)
    if budget_threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    # Each member's HiGHS gets an equal share, so all the members together stay
    # within the thread budget.
    member_threads = max(1, budget_threads // n_workers)
    if not grace >= 1.0:
        raise ValueError(f"grace must be >= 1, got {grace}")
    if grace_window is not None:
        lo_w, hi_w = (float(v) for v in grace_window)
        if not (0.0 <= lo_w <= hi_w) or not math.isfinite(grace):
            raise ValueError(f"grace_window needs 0 <= lo <= hi and a finite grace: {grace_window}")

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
                        discopt_dir=_DISCOPT_DIR,
                        highs_threads=member_threads,
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
                        t_done = time.time() - wall0
                        grace_end = wall0 + t_done + _grace_extra(t_done, grace, grace_window)
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
