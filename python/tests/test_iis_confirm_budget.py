"""Confirmation solves in the MILP IIS engine get the budget their caller gave.

Measured on MIPcc27 at 600 s, single order:

* checkpoint 0 (confirming the base set presolve-off) was cut to the 30 s
  per-probe limit inside ``_Engine._solve``. dives and neos-1461051 need 63 s and
  90 s presolve-off, so nothing was ever confirmed and both runs ended in
  ``TimeoutError`` with no answer;
* the end-of-budget confirmation re-proved the current set presolve-on and then
  presolve-off, each cut to 30 s. On dijle and dingo it failed, and the runs fell
  back to the untouched base set after filtering away half the integralities.

The set the filter holds when it stops is the last set a presolve-on probe
proved infeasible (a batch it could not drop is put back), so the final
confirmation needs only the presolve-free route, with the reserve up to the
deadline. Both routes are still required; only the redundant re-solve goes.
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_milp
from discopt._iis_milp import _Engine, milp_iis

highspy = pytest.importorskip("highspy")


def _engine(probe_tl=0.5, deadline=None):
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    return _Engine(
        A,
        np.array([2.5]),
        np.array([1e20]),
        np.zeros(2),
        np.ones(2),
        np.array([True, True]),
        {0: np.array([0])},
        probe_tl,
        confirm_each=False,
        deadline=deadline,
    )


def _captured_time_limits(eng, monkeypatch):
    """HiGHS time_limit values set by ``_solve``, relative to the run clock."""
    limits: list[float] = []
    real = eng.h.setOptionValue

    def spy(name, value):
        if name == "time_limit":
            limits.append(value - eng.h.getRunTime())
        return real(name, value)

    monkeypatch.setattr(eng.h, "setOptionValue", spy)
    return limits


def test_a_caller_given_stop_replaces_the_per_probe_limit(monkeypatch):
    import time

    eng = _engine(probe_tl=0.5)
    eng._configure({0}, set(), {0, 1})
    limits = _captured_time_limits(eng, monkeypatch)
    eng._solve("off")  # an ordinary probe: the per-probe limit
    eng._solve("off", stop_at=time.perf_counter() + 100.0)  # a confirmation's own budget
    assert len(limits) == 2
    assert limits[0] <= 0.5 + 1e-6
    assert limits[1] > 99.0  # was cut to 0.5


def test_a_caller_given_stop_is_still_cut_to_the_deadline(monkeypatch):
    import time

    eng = _engine(probe_tl=0.5, deadline=time.perf_counter() + 5.0)
    eng._configure({0}, set(), {0, 1})
    limits = _captured_time_limits(eng, monkeypatch)
    eng._solve("off", stop_at=time.perf_counter() + 100.0)
    assert len(limits) == 1 and limits[0] <= 5.0 + 1e-6


def test_proven_on_records_exactly_the_last_presolve_on_proof():
    eng = _engine()
    assert not eng.proven_on({0}, set(), {0, 1})
    res, _ = eng.probe({0}, set(), {0, 1})  # x0 + x1 >= 2.5 over binaries: infeasible
    assert res is True
    assert eng.proven_on({0}, set(), {0, 1})
    assert not eng.proven_on({0}, set(), {0})  # a different set is not covered
    # x1 relaxed, its box kept as bound elements: still infeasible (1 + 1 < 2.5).
    box = {(1, "lower"), (1, "upper")}
    res, _ = eng.probe({0}, box, {0})
    assert res is True and eng.proven_on({0}, box, {0})
    assert not eng.proven_on({0}, set(), {0, 1})


def _parity_model(n=12):
    m = dm.Model("parity")
    x = m.binary("x", shape=(n,))
    m.subject_to(sum(x[k] for k in range(n)) == n / 2 + 0.5, name="half")
    m.minimize(0 * x[0])
    return m


def test_final_confirmation_skips_the_redundant_presolve_on_solve(monkeypatch):
    """With a budget, the finished set's presolve-on proof is on record, so its
    confirmation is one presolve-free solve with the deadline as its stop."""
    calls: list[tuple] = []
    real = _iis_milp._Engine.confirm_infeasible

    def spy(self, cons, bounds, ints, stop_at=None, presolve=("choose", "off")):
        calls.append((tuple(presolve), stop_at is not None))
        return real(self, cons, bounds, ints, stop_at=stop_at, presolve=presolve)

    monkeypatch.setattr(_iis_milp._Engine, "confirm_infeasible", spy)
    res = milp_iis(_parity_model(), total_time_limit=60.0, sat_candidate=False)
    assert res.complete and res.proven_irreducible
    final = calls[-1]
    assert final == (("off",), True)
    # Without a budget the confirmation is unchanged apart from the skipped re-solve.
    calls.clear()
    res = milp_iis(_parity_model(), sat_candidate=False)
    assert res.complete and calls[-1] == (("off",), False)


def test_final_confirmation_still_refutes_a_wrong_presolve_on_claim(monkeypatch):
    """The skipped solve is only the redundant one: a presolve-free route that
    finds an exact-integer point still sends the run to the per-probe rerun."""
    real_solve = _iis_milp._Engine._solve
    real_witness = _iis_milp._Engine._witness
    state = {"armed": False, "used": False}

    def solve(self, presolve, stop_at=None):
        if presolve == "off" and state["armed"] and not state["used"]:
            return highspy.HighsModelStatus.kOptimal
        return real_solve(self, presolve, stop_at)

    def witness(self, stop_at=None):
        if state["armed"] and not state["used"]:
            state["used"] = True
            return np.zeros(self.col_lb.size)
        return real_witness(self, stop_at)

    real_confirm = _iis_milp._Engine.confirm_infeasible
    n = {"confirms": 0}

    def confirm(self, *a, **k):
        n["confirms"] += 1
        if k.get("presolve") == ("off",) and k.get("stop_at") is None and n["confirms"] > 0:
            state["armed"] = True  # the final confirmation (no budget: stop_at None)
        return real_confirm(self, *a, **k)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    monkeypatch.setattr(_iis_milp._Engine, "_witness", witness)
    monkeypatch.setattr(_iis_milp._Engine, "confirm_infeasible", confirm)
    res = milp_iis(_parity_model(8), sat_candidate=False)
    assert state["used"]
    assert res.fallback and res.complete  # redone with per-probe confirmation
