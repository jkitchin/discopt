"""The MILP solver behind the IIS engine (highs-parity P3.2).

:class:`discopt._iis_milp._Engine` builds one model and toggles elements, solving after
each toggle. Everything solver-specific it needs is here, behind one small interface,
so the deletion filter, the #1634 cross-checks, the witness repair and the ladder are
the same code whichever solver answers:

* ``configure(rlo, rhi, lb, ub, integ)`` -- load one probe's rows, bounds, integrality;
* ``set_cols(idx, lo, hi, integer)`` -- edit a few columns (the witness repair);
* ``run(presolve, stop, max_nodes=None)`` -- solve from scratch, ``presolve`` being
  ``"choose"`` (on) or ``"off"``, ``stop`` an absolute ``time.perf_counter()`` budget;
  returns one of :data:`OPTIMAL` (a feasible point exists: the objective is zero),
  :data:`INFEASIBLE`, :data:`UNBOUNDED_OR_INFEASIBLE`, :data:`INTERRUPTED` (stopped by
  the budget) or :data:`OTHER` (any other non-proof status);
* ``presolve_only(stop)`` -- does presolve alone prove the set infeasible;
* ``point()``, ``last_nodes()`` -- the last run's point and node count;
* ``mip_tol()`` and ``lp_tolerance(tol)`` -- the witness repair's tolerances.

:class:`HighsBackend` is the code ``_Engine`` ran before P3.2, moved here unchanged.
:class:`RustBackend` is the in-house MILP driver through
:class:`discopt.solvers.milp_handle.MilpHandle`, selected by ``iis_engine="rust"``;
``"highs"`` is the default (P3.2).
"""

from __future__ import annotations

import contextlib
import math
import time
from typing import Any, Iterator, Optional

import numpy as np

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED_OR_INFEASIBLE = "unbounded_or_infeasible"
INTERRUPTED = "interrupted"
OTHER = "other"

#: The sentinel the IIS arrays use for "no bound" (``discopt._iis_milp._INF``).
_INF = 1e20
#: The engine's wall-clock guard lets a run overrun its budget by this much before
#: interrupting it (``discopt._iis_milp._GRACE_S``; kept in step by the import there).
GRACE_S = 0.5
#: HiGHS's default ``mip_feasibility_tolerance``. The witness repair's arithmetic
#: check (``_Engine._witness``) judges a rounded point by it; the Rust backend uses
#: the same number so a point is accepted or refused the same way on either engine.
MIP_FEASIBILITY_TOLERANCE = 1e-6

ENGINES = ("rust", "highs")


def make_backend(engine: str, A: Any, row_lo, row_hi, col_lb, col_ub, is_int) -> Any:
    """The backend ``engine`` names, holding the model ``row_lo <= A x <= row_hi``,
    ``col_lb <= x <= col_ub``, ``x_j`` integer where ``is_int``."""
    if engine == "highs":
        return HighsBackend(A, row_lo, row_hi, col_lb, col_ub, is_int)
    if engine == "rust":
        return RustBackend(A, row_lo, row_hi, col_lb, col_ub, is_int)
    raise ValueError(f"iis_engine={engine!r}: expected one of {ENGINES}")


class HighsBackend:
    """One ``Highs`` object (the engine's pre-P3.2 code, unchanged in behaviour)."""

    name = "highs"

    def __init__(self, A, row_lo, row_hi, col_lb, col_ub, is_int) -> None:
        from discopt._optional_highs import require_highspy

        highspy = require_highspy("the HiGHS IIS backend (iis_engine='highs')")
        self.hs = highspy
        n, m = col_lb.size, row_lo.size
        inf = highspy.kHighsInf
        self.inf = inf
        h = highspy.Highs()
        h.silent()
        mn = h.getOptionValue("mip_max_nodes")
        self._default_max_nodes = int(mn[1] if isinstance(mn, tuple) else mn)
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
        self.n, self.m = n, m
        self._stop = math.inf

        # Wall-clock guard on every run: HiGHS's own ``time_limit`` was measured
        # overrunning 7-10x on presolve-free re-solves, so a budget is enforced
        # here too. An interrupted run is a non-proof status (inconclusive).
        def _guard(event) -> None:
            # Set the flag both ways: HiGHS keeps a raised interrupt across runs.
            event.interrupt(time.perf_counter() > self._stop)

        h.cbMipInterrupt.subscribe(_guard)
        h.cbSimplexInterrupt.subscribe(_guard)
        h.cbIpmInterrupt.subscribe(_guard)

    def configure(self, rlo, rhi, lb, ub, integ) -> None:
        h, n, m = self.h, self.n, self.m
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

    def set_cols(self, idx: np.ndarray, lo, hi, integer: bool) -> None:
        h, vt = self.h, self.hs.HighsVarType
        idx = idx.astype(np.int32)
        h.changeColsBounds(idx.size, idx, lo, hi)
        kind = vt.kInteger if integer else vt.kContinuous
        h.changeColsIntegrality(idx.size, idx, np.array([kind] * idx.size))

    def run(self, presolve: str, stop: float, max_nodes: Optional[int] = None) -> str:
        h = self.h
        now = time.perf_counter()
        h.clearSolver()
        # HiGHS's run clock is cumulative over this object's lifetime and LP
        # solves compare it with ``time_limit``, so a relative budget made every
        # LP solve return "Time limit reached" (0 iterations) once the engine had
        # run longer than one budget. Offset by the clock; the wall guard
        # enforces the real budget whichever clock HiGHS consults.
        h.setOptionValue("time_limit", h.getRunTime() + (stop - now))
        h.setOptionValue("presolve", presolve)
        if max_nodes is not None:
            h.setOptionValue("mip_max_nodes", int(max_nodes))
        self._stop = stop + GRACE_S
        try:
            h.run()
        finally:
            self._stop = math.inf
            if max_nodes is not None:
                h.setOptionValue("mip_max_nodes", self._default_max_nodes)
        MS = self.hs.HighsModelStatus
        st = h.getModelStatus()
        if st == MS.kOptimal:
            return OPTIMAL
        if st == MS.kInfeasible:
            return INFEASIBLE
        if st == MS.kUnboundedOrInfeasible:
            return UNBOUNDED_OR_INFEASIBLE
        if st == MS.kInterrupt:
            return INTERRUPTED
        return OTHER

    def presolve_only(self, stop: float) -> bool:
        h = self.h
        now = time.perf_counter()
        h.clearSolver()
        h.setOptionValue("time_limit", h.getRunTime() + (stop - now))
        h.setOptionValue("presolve", "choose")
        h.presolve()
        return bool(h.getModelPresolveStatus() == self.hs.HighsPresolveStatus.kInfeasible)

    def point(self) -> np.ndarray:
        return np.array(self.h.getSolution().col_value, dtype=np.float64)

    def last_nodes(self) -> int:
        return int(getattr(self.h.getInfo(), "mip_node_count", 0) or 0)

    def mip_tol(self) -> float:
        v = self.h.getOptionValue("mip_feasibility_tolerance")
        return float(v[1] if isinstance(v, tuple) else v)

    @contextlib.contextmanager
    def lp_tolerance(self, tol: float) -> Iterator[None]:
        h = self.h
        old = h.getOptionValue("primal_feasibility_tolerance")
        # (status, value) tuples in some highspy versions
        old = old[1] if isinstance(old, tuple) else old
        h.setOptionValue("primal_feasibility_tolerance", tol)
        try:
            yield
        finally:
            h.setOptionValue("primal_feasibility_tolerance", old)


class RustBackend:
    """The in-house MILP driver through a persistent ``MilpHandle``.

    Status mapping. The probe's objective is zero, so a driver ``optimal`` or
    ``feasible`` (an incumbent exists) is a feasible point: :data:`OPTIMAL`. A driver
    ``infeasible`` is a proof: :data:`INFEASIBLE`. A ``node_limit`` stop is
    :data:`INTERRUPTED` when the budget was spent, else :data:`OTHER` (the ladder's
    node cap); ``unbounded`` cannot hold with a zero objective and is :data:`OTHER`.
    The driver honours ``time_limit`` itself (P0.7 bounded its per-node overrun), so
    there is no interrupt callback.
    """

    name = "rust"

    def __init__(self, A, row_lo, row_hi, col_lb, col_ub, is_int) -> None:
        from discopt.solvers.milp_handle import MilpHandle

        self.inf = math.inf
        self.n, self.m = col_lb.size, row_lo.size
        self.handle = MilpHandle(
            np.zeros(self.n), A, row_lo, row_hi, col_lb, col_ub, np.asarray(is_int, dtype=bool)
        )
        self._cfg = [
            np.asarray(row_lo, float).copy(),
            np.asarray(row_hi, float).copy(),
            np.asarray(col_lb, float).copy(),
            np.asarray(col_ub, float).copy(),
            np.asarray(is_int, dtype=bool).copy(),
        ]
        self._last: Any = None
        self._A = __import__("scipy.sparse").sparse.csc_matrix(A, dtype=np.float64)
        self._ladder_x: Optional[np.ndarray] = None

    def configure(self, rlo, rhi, lb, ub, integ) -> None:
        self._cfg = [
            np.asarray(rlo, float).copy(),
            np.asarray(rhi, float).copy(),
            np.asarray(lb, float).copy(),
            np.asarray(ub, float).copy(),
            np.asarray(integ, dtype=bool).copy(),
        ]
        self.handle.configure(*self._cfg)

    def set_cols(self, idx: np.ndarray, lo, hi, integer: bool) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        self._cfg[2][idx] = lo
        self._cfg[3][idx] = hi
        self._cfg[4][idx] = integer
        self.handle.configure(*self._cfg)

    def run(self, presolve: str, stop: float, max_nodes: Optional[int] = None) -> str:
        left = max(stop - time.perf_counter(), 1e-3)
        kw: dict[str, Any] = {"time_limit": left, "presolve": presolve != "off"}
        if max_nodes is not None:
            kw["max_nodes"] = int(max_nodes)
        r = self.handle.solve(**kw)
        self._last = r
        self._ladder_x = None
        if r.status in ("optimal", "feasible"):
            return OPTIMAL
        if r.status == "infeasible":
            return INFEASIBLE
        if r.status == "node_limit" and time.perf_counter() >= stop:
            return INTERRUPTED
        if not self._cfg[4].any():
            return self._lp_ladder(stop)
        return OTHER

    def _lp_ladder(self, stop: float) -> str:
        """An undecided pure-LP probe through the verified LP ladder (P3.2).

        The driver cannot certify an LP's infeasibility when its Farkas ray carries
        roundoff on a free column (``FarkasRejectOpen``, ``lp/simplex/primal.rs``);
        ``solve_lp_std(engine="rust")`` decides it through the phase-1 proof and the
        exact dual correction. Its verdicts are certificates (``gap_certified``) or
        a verified point, never a guess."""
        import scipy.sparse as sp

        from discopt.solvers import lp_milp_highs as H

        rlo, rhi, lb, ub, _ = self._cfg
        A = self._A
        m, n = A.shape
        full = sp.hstack([A, -sp.identity(m, format="csc")], format="csc")
        xl = np.concatenate([np.where(lb > -_INF, lb, -H.INF), np.where(rlo > -_INF, rlo, -H.INF)])
        xu = np.concatenate([np.where(ub < _INF, ub, H.INF), np.where(rhi < _INF, rhi, H.INF)])
        sf = H.StdForm.from_arrays(np.zeros(n + m), full, np.zeros(m), xl, xu, 0.0)
        left = max(stop - time.perf_counter(), 1e-3)
        out = H.solve_lp_std(sf, time_limit=left, engine="rust")
        if out.status == "infeasible" and out.gap_certified:
            return INFEASIBLE
        if out.status == "optimal" and out.x is not None:
            self._ladder_x = np.asarray(out.x, float)[:n]
            return OPTIMAL
        return OTHER

    def presolve_only(self, stop: float) -> bool:
        return self.handle.presolve_only()

    def point(self) -> np.ndarray:
        if self._ladder_x is not None:
            return self._ladder_x.copy()
        return np.asarray(self._last.x, dtype=np.float64).copy()

    def last_nodes(self) -> int:
        return 0 if self._last is None else int(self._last.nodes)

    def mip_tol(self) -> float:
        return MIP_FEASIBILITY_TOLERANCE

    @contextlib.contextmanager
    def lp_tolerance(self, tol: float) -> Iterator[None]:
        # The Rust simplex's audit tolerance is fixed; the repair's verdict is still
        # judged by ``mip_tol`` arithmetic in ``_Engine._witness``.
        yield
