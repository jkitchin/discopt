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
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from discopt.modeling.core import Model

_INF = 1e20  # discopt's unbounded-bound sentinel

#: A solve is not started with less wall time than this left in its budget.
_MIN_SOLVE_S = 0.05
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
        off, at HiGHS's standard feasibility tolerance. Its point has exact
        integers and is what the caller gets (and what becomes a certificate).
        Only integrality is made exact: tightening the continuous tolerance too
        (it was 1e-9) rejected ordinary points -- every one of ``dell``'s 203
        repairs, whose rounded points met their rows to 9.4e-8 -- while osun's
        artifacts fail the repair at 1e-9, 1e-7 and 1e-6 alike.
        """
        x = self._point()
        held = self._held
        if held.size == 0:
            return x
        self.n_repairs += 1
        h, vt = self.h, self.hs.HighsVarType
        xr = np.round(x[held])
        idx = held.astype(np.int32)
        h.changeColsBounds(idx.size, idx, xr, xr)
        h.changeColsIntegrality(idx.size, idx, np.array([vt.kContinuous] * idx.size))
        try:
            ok = self._solve("off", stop_at) == self.hs.HighsModelStatus.kOptimal
            y = self._point() if ok else None
        finally:
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
    ) -> Optional[bool]:
        """``True`` iff presolve-on AND presolve-off both return ``Infeasible``;
        ``False`` if either finds an exact-integer feasible point (an earlier
        verdict was wrong); ``None`` if undecided (time limit, interrupt, or a
        "feasible" status whose point is a tolerance artifact)."""
        self._configure(active_cons, active_bounds, active_int)
        MS = self.hs.HighsModelStatus
        all_infeasible = True
        for p in ("choose", "off"):
            st = self._solve(p, stop_at)
            if st == MS.kOptimal and self._witness(stop_at) is not None:
                return False
            all_infeasible &= st == MS.kInfeasible
        return True if all_infeasible else None


def _violation(x, eng: _Engine, rows=None, cols_lb=None, cols_ub=None, cols_int=None):
    """Max violation of the given rows / bounds / integralities at ``x``."""
    v = 0.0
    if rows is not None and len(rows):
        act = eng.A[rows] @ x
        lo, hi = eng.row_lo[rows], eng.row_hi[rows]
        v = max(v, float(np.max(np.where(lo > -_INF, lo - act, 0.0), initial=0.0)))
        v = max(v, float(np.max(np.where(hi < _INF, act - hi, 0.0), initial=0.0)))
    if cols_lb is not None and len(cols_lb):
        v = max(v, float(np.max(eng.col_lb[cols_lb] - x[cols_lb], initial=0.0)))
    if cols_ub is not None and len(cols_ub):
        v = max(v, float(np.max(x[cols_ub] - eng.col_ub[cols_ub], initial=0.0)))
    if cols_int is not None and len(cols_int):
        xi = x[cols_int]
        v = max(v, float(np.max(np.abs(xi - np.round(xi)), initial=0.0)))
    return v


def _check_certificate(x, key, eng, cons, bounds, ints, tol) -> bool:
    """``x`` satisfies the final subsystem minus ``key`` and violates ``key``."""
    kind = key[0]
    cons_o = [c for c in cons if not (kind == "constraint" and c == key[1])]
    ints_o = [j for j in ints if not (kind == "integrality" and j == key[1])]
    held = set(ints_o)
    # Bounds of a column held integer apply; for the integrality element itself
    # its own bounds still apply too (relaxing integrality keeps the box).
    lb_cols = sorted(held | {j for j, s in bounds if s == "lower"})
    ub_cols = sorted(held | {j for j, s in bounds if s == "upper"})
    if kind == "integrality":
        lb_cols = sorted(set(lb_cols) | {key[1]})
        ub_cols = sorted(set(ub_cols) | {key[1]})
    if kind == "bound":
        j, s = key[1], key[2]
        if s == "lower":
            lb_cols = [c for c in lb_cols if c != j]
        else:
            ub_cols = [c for c in ub_cols if c != j]
    rows = np.concatenate([eng.row_groups[c] for c in cons_o]) if cons_o else np.zeros(0, int)
    lb_cols = [c for c in lb_cols if eng.col_lb[c] > -_INF]
    ub_cols = [c for c in ub_cols if eng.col_ub[c] < _INF]
    if _violation(x, eng, rows, lb_cols, ub_cols, ints_o) > tol:
        return False
    # ``key`` itself must be violated by more than tol.
    if kind == "constraint":
        e = _violation(x, eng, rows=eng.row_groups[key[1]])
    elif kind == "integrality":
        e = _violation(x, eng, cols_int=[key[1]])
    else:
        j, s = key[1], key[2]
        e = _violation(
            x, eng, cols_lb=[j] if s == "lower" else None, cols_ub=[j] if s == "upper" else None
        )
    return bool(e > tol)


def milp_iis(
    model: "Model",
    *,
    include_bounds: bool = True,
    time_limit: float = 30.0,
    cert_tol: float = 1e-6,
    max_batch: Optional[int] = None,
    confirm: str = "final",
    total_time_limit: Optional[float] = None,
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
    (two recent solve durations) to confirm it, and the subsystem confirmed at
    the end of each completed phase is kept as a fallback. If no subsystem can be confirmed in time,
    :class:`TimeoutError` is raised rather than an unconfirmed claim returned.

    Raises ``NotImplementedError`` (from the linear extractor) for a model with a
    nonlinear or non-algebraic constraint, ``ValueError`` when the full model is
    not proven infeasible, and ``TimeoutError`` as above.
    """
    t_start = time.perf_counter()
    deadline = None if total_time_limit is None else t_start + float(total_time_limit)
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

    # The base proof is mandatory, so under an overall budget it may use all of
    # it rather than one probe's ``time_limit`` (no-ip-65059 needs > 30 s).
    eng.probe_tl = math.inf if deadline is not None else time_limit
    try:
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
            batch_filter(sorted(ints_active), ints_active, lambda j: ("integrality", j))

    def phase_constraints() -> None:
        batch_filter(sorted(cons), cons, lambda ci: ("constraint", ci))

    def phase_bounds() -> None:
        # Bounds of columns not held integer (a held column's box is not an element).
        if include_bounds:
            batch_filter(
                [jb for jb in finite_bounds if jb[0] not in ints_active],
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
        redo = milp_iis(
            model,
            include_bounds=include_bounds,
            time_limit=time_limit,
            cert_tol=cert_tol,
            max_batch=max_batch,
            confirm="each",
            total_time_limit=left,
        )
        redo.n_solves += eng.n_solves
        redo.n_interrupted += eng.n_interrupted
        redo.confirm = "final"
        redo.fallback = True
        return redo

    complete = True
    # "final": the latest subsystem confirmed infeasible under both
    # configurations, kept as the anytime fallback. ("each" confirms every
    # verdict, so its current subsystem is always confirmed.)
    checkpoint: Optional[tuple[set, set, set]] = None
    try:
        for phase in (phase_integrality, phase_constraints, phase_bounds):
            phase()
            if confirm == "final" and phase is not phase_bounds:
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
    certs = {}
    for key, x in pending.items():
        if _check_certificate(
            x, key, eng, sorted(cons), applied_bounds, sorted(ints_active), cert_tol
        ):
            certs[key] = x
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
    )
