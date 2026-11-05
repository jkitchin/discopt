"""#2143: ``DISCOPT_HEUR_FIXING_SCREEN`` -- root heuristics stop re-solving fixings
already shown to give nothing.

Measured on GDPlib ``syngas`` under hull: the root ``integer_local_search`` posed 7
sub-NLPs on ONE integer fixing (3.5-4.2 s each), and inside the LOA route's
native-kernel seed 25 of 25 ILS sub-NLPs posed the same start; every round of the
root ``feasibility_pump`` was refuted by node FBBT before its NLP, and the pump still
spent 28.7 s on capped NLPs and a repair.

* ILS (:class:`_FixingMemo`): a fixing ``subnlp`` refused on a proof over the fixed
  box is not re-posed (any start); an identical start whose NLP failed on a status
  other than ``TIME_LIMIT`` is not re-posed. The work budget is charged as before.
* pump: each round's fixing goes through ``_refuted_by_integer_rows`` and
  ``_implied_box`` first; a refuted round runs neither the NLP nor the repair.

Default OFF; the legacy behaviour is pinned under ``=0``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as PH
from discopt.solvers import NLPResult, SolveStatus

pytestmark = pytest.mark.smoke

_FLAG = "DISCOPT_HEUR_FIXING_SCREEN"


class _Spy:
    """An NLP backend that counts calls; delegates to the real one unless ``status`` is set."""

    def __init__(self, status: SolveStatus | None = None):
        self.status = status
        self.calls = 0
        self.real = None
        if status is None:
            from discopt.solvers.nlp_backend import get_nlp_solver

            self.real = get_nlp_solver("auto")

    def __call__(self, evaluator, x0, options=None, **kw):
        self.calls += 1
        if self.real is not None:
            return self.real(evaluator, x0, options=options, **kw)
        # A failed solve that returns no point: subnlp returns None.
        return NLPResult(status=self.status, x=None)


def _refuted_model() -> dm.Model:
    """Every fixing of ``y`` is refuted by node FBBT: ``x - y >= 2`` with ``x <= 1``.

    The row holds the continuous ``x``, so the integer-row test cannot see it; only
    FBBT on the fixed box can.
    """
    m = dm.Model("refuted")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.binary("y")
    m.subject_to(x - y >= 2.0)
    m.minimize((x - 0.5) ** 2 + y)
    return m


def _feasible_model() -> dm.Model:
    """Both fixings of ``y`` have continuous completions."""
    m = dm.Model("feasible")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.binary("y")
    m.subject_to(x >= 1.0 + y)
    m.minimize((x - 2.5) ** 2 + y)
    return m


def _count_implied(monkeypatch) -> list[int]:
    calls = [0]
    real = PH._implied_box

    def spy(evaluator):
        calls[0] += 1
        return real(evaluator)

    monkeypatch.setattr(PH, "_implied_box", spy)
    return calls


# ── the flag ──────────────────────────────────────────────────────────────


def test_flag_is_default_on(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert PH._heur_fixing_screen_enabled() is True  # graduated 2026-11-05 (#2143)
    monkeypatch.setenv(_FLAG, "0")
    assert PH._heur_fixing_screen_enabled() is False
    monkeypatch.setenv(_FLAG, "1")
    assert PH._heur_fixing_screen_enabled() is True
    monkeypatch.setenv(_FLAG, "0")
    assert PH._heur_fixing_screen_enabled() is False


# ── subnlp reports why it returned nothing ─────────────────────────────────


def test_subnlp_reports_an_fbbt_refutation_before_any_solve():
    m = _refuted_model()
    spy = _Spy()
    out: dict = {}
    assert PH.subnlp(m, np.array([0.5, 0.0]), backend=spy, outcome=out) is None
    assert out == {"refuted": "fbbt"} and spy.calls == 0


def test_subnlp_reports_the_nlp_status():
    m = _feasible_model()
    out: dict = {}
    spy = _Spy(SolveStatus.ERROR)
    assert PH.subnlp(m, np.array([2.0, 0.0]), backend=spy, outcome=out) is None
    assert out == {"nlp_status": SolveStatus.ERROR} and spy.calls == 1


# ── A1: integer_local_search memo ──────────────────────────────────────────


def _ils(m, backend, x0):
    stats: dict = {}
    r = PH.integer_local_search(
        m, x0, backend=backend, eval_budget=10_000, solve_budget=50, stats=stats
    )
    return r, stats


def test_ils_does_not_repose_a_refuted_fixing(monkeypatch):
    # Every restart descends to y = 0, which FBBT refutes: ON poses it once.
    m = _refuted_model()
    x0 = np.array([0.5, 0.0])
    monkeypatch.setenv(_FLAG, "0")
    off_calls = _count_implied(monkeypatch)
    r_off, st_off = _ils(m, _Spy(), x0)
    n_off = off_calls[0]
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.undo()
    monkeypatch.setenv(_FLAG, "1")
    on_calls = _count_implied(monkeypatch)
    r_on, st_on = _ils(m, _Spy(), x0)
    assert r_off is None and r_on is None
    # Legacy: every repair re-runs FBBT on the same fixing.
    assert st_off["subnlp_calls"] >= 2 and n_off == st_off["subnlp_calls"]
    assert st_off["memo_skipped_refuted"] == 0
    # ON: the same number of repairs is reached, all but the distinct fixings skipped.
    assert st_on["memo_skipped_refuted"] >= 1
    assert st_on["subnlp_calls"] + st_on["memo_skipped_refuted"] == st_off["subnlp_calls"]
    assert on_calls[0] == st_on["subnlp_calls"] < n_off


def test_ils_does_not_repose_an_identical_failed_start(monkeypatch):
    m = _feasible_model()
    x0 = np.array([2.0, 0.0])
    monkeypatch.setenv(_FLAG, "0")
    off = _Spy(SolveStatus.ERROR)
    r_off, st_off = _ils(m, off, x0)
    monkeypatch.setenv(_FLAG, "1")
    on = _Spy(SolveStatus.ERROR)
    r_on, st_on = _ils(m, on, x0)
    assert r_off is None and r_on is None
    assert st_on["memo_skipped_failed"] >= 1
    assert st_on["subnlp_calls"] + st_on["memo_skipped_failed"] == st_off["subnlp_calls"]
    assert on.calls == off.calls - st_on["memo_skipped_failed"]


def test_ils_reposes_a_start_that_stopped_on_its_wall_clock(monkeypatch):
    # TIME_LIMIT depends on machine speed: a repeat might get further, so it is solved.
    m = _feasible_model()
    monkeypatch.setenv(_FLAG, "1")
    on = _Spy(SolveStatus.TIME_LIMIT)
    _, st = _ils(m, on, np.array([2.0, 0.0]))
    assert st["memo_skipped_failed"] == 0 and st["memo_skipped_refuted"] == 0
    assert st["subnlp_calls"] >= 2


def test_ils_still_finds_the_point_with_the_memo_on(monkeypatch):
    m = _feasible_model()
    monkeypatch.setenv(_FLAG, "1")
    r, _ = _ils(m, None, np.array([2.0, 1.0]))
    assert r is not None
    assert r[1] == pytest.approx(0.0, abs=1e-5)  # y=0, x=2.5


def test_memo_keys_a_fixing_as_subnlp_poses_it():
    mask = np.array([False, True])
    lb, ub = np.array([0.0, 0.0]), np.array([4.0, 1.0])
    memo = PH._FixingMemo(mask, lb, ub)
    memo.record(np.array([2.0, 0.4]), {"refuted": "fbbt"})  # rounds to y = 0
    assert memo.known_to_fail(np.array([3.7, -0.0]))  # any start, same integers
    assert not memo.known_to_fail(np.array([3.7, 0.6]))
    memo.record(np.array([1.5, 1.0]), {"nlp_status": SolveStatus.ERROR})
    assert memo.known_to_fail(np.array([1.5, 1.0]))
    assert not memo.known_to_fail(np.array([1.25, 1.0]))  # another start: solved
    memo.record(np.array([3.0, 1.0]), {"nlp_status": SolveStatus.TIME_LIMIT})
    assert not memo.known_to_fail(np.array([3.0, 1.0]))
    assert (memo.skipped_refuted, memo.skipped_failed) == (1, 1)


# ── A2: feasibility_pump screen ────────────────────────────────────────────


def _count_repairs(monkeypatch) -> list[int]:
    calls = [0]
    real = PH._repair_to_feasible

    def spy(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    monkeypatch.setattr(PH, "_repair_to_feasible", spy)
    return calls


def test_pump_solves_nothing_on_refuted_rounds(monkeypatch):
    m = _refuted_model()
    x = np.array([0.5, 0.3])
    monkeypatch.setenv(_FLAG, "0")
    rep_off = _count_repairs(monkeypatch)
    off = _Spy()
    assert PH.feasibility_pump(m, x, max_rounds=5, backend=off) is None
    # Legacy: the NLP runs every round, and the repair once.
    assert off.calls == 5 and rep_off[0] == 1

    monkeypatch.undo()
    monkeypatch.setenv(_FLAG, "1")
    rep_on = _count_repairs(monkeypatch)
    on = _Spy()
    stats: dict = {}
    assert PH.feasibility_pump(m, x, max_rounds=5, backend=on, stats=stats) is None
    assert on.calls == 0 and rep_on[0] == 0
    assert stats == {"screened": 5}


def test_pump_screen_refuses_an_integer_row_before_the_nlp(monkeypatch):
    # ``y0 + y1 == 1`` holds integers only: rounding (1, 1) breaks it.
    m = dm.Model("introw")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y0, y1 = m.binary("y0"), m.binary("y1")
    m.subject_to(y0 + y1 == 1)
    m.subject_to(x >= 1.0 + y0)
    m.minimize((x - 2.5) ** 2)
    monkeypatch.setenv(_FLAG, "1")
    implied = _count_implied(monkeypatch)
    spy = _Spy()
    stats: dict = {}
    PH.feasibility_pump(m, np.array([2.0, 0.9, 0.8]), max_rounds=1, backend=spy, stats=stats)
    assert spy.calls == 0 and implied[0] == 0 and stats == {"screened": 1}


def test_pump_still_returns_a_point_on_a_consistent_fixing(monkeypatch):
    m = _feasible_model()
    monkeypatch.setenv(_FLAG, "1")
    stats: dict = {}
    x = PH.feasibility_pump(m, np.array([2.0, 0.2]), max_rounds=5, stats=stats)
    assert x is not None and x[1] == 0.0
    assert stats == {"posed": 1}
