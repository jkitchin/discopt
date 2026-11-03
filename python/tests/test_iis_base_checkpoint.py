"""Checkpoint 0: under ``total_time_limit`` the base set is confirmed up front.

With ``confirm="final"`` the filter acts on presolve-on verdicts and confirms
presolve-off only at phase ends. A budget that ran out before the first phase
ended therefore had nothing confirmed to return and raised ``TimeoutError``
(MIPcc27's ``enlight11`` and ``bf1355-075`` at 3500 s), although the base set --
proven infeasible presolve-on before filtering began -- is a valid infeasible
subsystem. The engine now confirms it presolve-off right after the base proof
and falls back to it.

The budget is simulated through ``_Engine._solve`` (raising the engine's own
out-of-time signal) so the tests are deterministic, not timing-dependent.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _iis_milp
from discopt._iis_milp import milp_iis

highspy = pytest.importorskip("highspy")
MS = highspy.HighsModelStatus


def _parity_model(n=30):
    """LP-feasible, integer-infeasible: sum of n binaries equal to n/2 + 0.5.
    Every integrality is necessary, so the integrality phase needs ~n probes."""
    m = dm.Model("parity")
    x = m.binary("x", shape=(n,))
    m.subject_to(sum(x[k] for k in range(n)) == n / 2 + 0.5, name="half")
    m.minimize(0 * x[0])
    return m


def _choose_budget(monkeypatch, n_real):
    """Presolve-on solves after the first ``n_real`` raise out-of-time; record calls."""
    real = _iis_milp._Engine._solve
    calls: list[str] = []

    def solve(self, presolve, stop_at=None):
        calls.append(presolve)
        if presolve == "choose" and calls.count("choose") > n_real:
            raise _iis_milp._OutOfTime
        return real(self, presolve, stop_at)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    return calls


def test_budget_out_in_first_phase_returns_confirmed_base_set(monkeypatch):
    # Real solves: the base proof and the LP-relaxation probe. The first
    # deletion probe and the final confirmation then find the budget gone.
    calls = _choose_budget(monkeypatch, n_real=2)
    m = _parity_model()
    res = milp_iis(m, total_time_limit=60.0)
    assert calls[:2] == ["choose", "off"]  # base proof, then checkpoint 0
    assert not res.complete and not res.proven_irreducible
    assert res.constraints == [0]
    assert res.integrality == list(range(30))  # the whole base set


def test_without_a_budget_there_is_no_base_confirmation(monkeypatch):
    calls = _choose_budget(monkeypatch, n_real=10**9)
    res = milp_iis(_parity_model(8))
    assert calls[:2] == ["choose", "choose"]
    assert res.complete


def test_base_confirmation_costs_one_solve_and_changes_nothing_else():
    # 46 columns: past the exact route's integer cap, so the second route is a
    # presolve-free solve (the exact route would confirm a smaller set with none).
    m = _parity_model(46)
    free = milp_iis(m)
    timed = milp_iis(m, total_time_limit=600.0)
    assert timed.n_solves == free.n_solves + 1
    assert (timed.constraints, timed.bounds, timed.integrality) == (
        free.constraints,
        free.bounds,
        free.integrality,
    )
    assert timed.complete and timed.proven_irreducible


def test_base_set_refuted_presolve_off_reruns_with_per_probe_confirmation(monkeypatch):
    # Presolve-off "finds" an exact-integer point for the base set: the
    # presolve-on base proof is not agreed, so nothing built on it may stand.
    real_solve = _iis_milp._Engine._solve
    real_witness = _iis_milp._Engine._witness
    state = {"faked": False}
    calls: list[str] = []

    def solve(self, presolve, stop_at=None):
        calls.append(presolve)
        if presolve == "off" and not state["faked"]:
            state["faked"] = True
            state["at_call"] = len(calls)
            return MS.kOptimal
        return real_solve(self, presolve, stop_at)

    def witness(self, stop_at=None):
        if state["faked"] and not state.get("used"):
            state["used"] = True
            return np.zeros(self.col_lb.size)
        return real_witness(self, stop_at)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    monkeypatch.setattr(_iis_milp._Engine, "_witness", witness)
    res = milp_iis(_parity_model(46), total_time_limit=60.0)  # past the exact cap
    assert state["used"]
    assert state["at_call"] == 2  # refuted at checkpoint 0, right after the base proof
    assert res.fallback and res.complete
    assert res.integrality == list(range(46))
