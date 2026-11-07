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
from discopt import _iis_backend as _bk
from discopt import _iis_milp
from discopt._iis_milp import milp_iis


@pytest.fixture(autouse=True)
def _pin_highs_engine(monkeypatch):
    """These tests' budgets and probe counts were tuned on the HiGHS IIS engine.
    Since highs-parity P6.4 (2026-11-07) the default engine is the in-house
    one, so these HiGHS-path tests pin ``DISCOPT_IIS_ENGINE=highs`` (a test that sets it still
    overrides)."""
    monkeypatch.setenv("DISCOPT_IIS_ENGINE", "highs")


highspy = pytest.importorskip("highspy")
MS = _bk  # backend statuses (highs-parity P3.2)


def _parity_model(n=30):
    """LP-feasible, integer-infeasible: sum of n binaries equal to n/2 + 0.5.
    Every integrality is necessary, so the integrality phase needs ~n probes."""
    m = dm.Model("parity")
    x = m.binary("x", shape=(n,))
    m.subject_to(sum(x[k] for k in range(n)) == n / 2 + 0.5, name="half")
    m.minimize(0 * x[0])
    return m


@pytest.fixture(autouse=True)
def _raw_presolve_free_route(monkeypatch):
    """These tests pin checkpoint 0 against the raw presolve-free solve, faked
    through ``_Engine._solve``. The exact-reduction route (tried first, see
    test_iis_reduce.py) would decide these sets before the faked solve runs."""
    monkeypatch.setattr(_iis_milp._Engine, "_reduced", lambda self, stop_at: None)


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
    m = _parity_model(12)
    free = milp_iis(m)
    timed = milp_iis(m, total_time_limit=600.0)
    # One solve for checkpoint 0, and one fewer at the end of the integrality
    # phase: every integrality is needed, so that set is no smaller than the
    # base set checkpoint 0 already confirmed, and its confirmation is skipped.
    assert timed.n_solves == free.n_solves
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
            return MS.OPTIMAL
        return real_solve(self, presolve, stop_at)

    def witness(self, stop_at=None):
        if state["faked"] and not state.get("used"):
            state["used"] = True
            return np.zeros(self.col_lb.size)
        return real_witness(self, stop_at)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    monkeypatch.setattr(_iis_milp._Engine, "_witness", witness)
    res = milp_iis(_parity_model(8), total_time_limit=60.0)
    assert state["used"]
    assert state["at_call"] == 2  # refuted at checkpoint 0, right after the base proof
    assert res.fallback and res.complete
    assert res.integrality == list(range(8))


def test_checkpoint_solve_does_not_inflate_the_confirmation_reserve(monkeypatch):
    """The reserve held back for the final confirmation is 2.5x the slowest
    recent solve. The checkpoint-0 solve is not a filtering probe: if its
    duration stayed in ``_recent``, a checkpoint that used its whole cap
    (25% of the budget) would hold back ~62% of it from the filter."""
    real_solve = _iis_milp._Engine._solve
    state = {"offs": 0}
    seen: list[float] = []

    def solve(self, presolve, stop_at=None):
        if presolve == "choose" and state["offs"]:
            seen.append(max(self._recent, default=0.0))
        out = real_solve(self, presolve, stop_at)
        if presolve == "off" and state["offs"] == 0:
            state["offs"] += 1
            self._recent[-1] = 5.0  # as if the checkpoint-0 solve took 5 s
        return out

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    milp_iis(_parity_model(46), total_time_limit=60.0)
    assert state["offs"] == 1 and seen
    assert seen[0] < 1.0  # the filter's first probe does not see the 5 s checkpoint
