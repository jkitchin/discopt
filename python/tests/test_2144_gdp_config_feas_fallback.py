"""#2144: the GDP configuration search's feasibility-first fallback.

On GDPlib batch_processing the relaxation's per-group argmax is infeasible and the
feasible configurations sit 15+ demotions away from it, so the ranked wave and the
dive (both steered by the relaxation) return nothing, while ~90 % of *uniformly*
sampled valid configurations are feasible. ``DISCOPT_GDP_CONFIG_FEAS_FALLBACK``
samples, once every other root constructor has missed, on a budget of its own: a
fixed number of sub-NLP **attempts**, not a share of the wave's deadline (the #2150 lesson -- a
shared deadline either starved the samples or let them displace a ranked plan).

The model here has the same property in miniature: 12 two-way disjunctions whose
cheap disjunct each claims 5 units of a 30-unit resource, so at most 6 cheap picks
fit, while the relaxation point (and the objective) favour the cheap disjunct in
every group. The argmax configuration is infeasible, and about 61 % of uniform
samples are feasible.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt._relax.primal_heuristics import (
    one_hot_config_feasibility_fallback,
    one_hot_config_subnlp,
    subnlp,
)
from discopt.solver import _gdp_feas_fallback_due
from discopt.validation.feasibility import max_constraint_violation

N_GROUPS = 12


def _argmax_infeasible_gdp():
    m = dm.Model("argmax_infeasible_gdp")
    y = m.binary("y", 2 * N_GROUPS)  # y[2i] cheap disjunct, y[2i+1] expensive
    x = m.continuous("x", shape=(N_GROUPS,), lb=0.0, ub=10.0)
    for i in range(N_GROUPS):
        m.subject_to(y[2 * i] + y[2 * i + 1] == 1, name=f"disj{i}")
        m.subject_to(x[i] >= 5.0 * y[2 * i], name=f"claim{i}")
    m.subject_to(sum(x[i] for i in range(N_GROUPS)) <= 30.0, name="resource")
    m.minimize(
        sum(1.0 * y[2 * i] + 3.0 * y[2 * i + 1] for i in range(N_GROUPS))
        + 0.01 * sum(x[i] * x[i] for i in range(N_GROUPS))
    )
    return m


def _relaxation_point():
    """What the relaxation says: the cheap disjunct, confidently, in every group."""
    x = np.zeros(3 * N_GROUPS)
    x[0 : 2 * N_GROUPS : 2] = 0.9
    x[1 : 2 * N_GROUPS : 2] = 0.1
    x[2 * N_GROUPS :] = 2.5
    return x


def _argmax_config_point():
    x = np.zeros(3 * N_GROUPS)
    x[0 : 2 * N_GROUPS : 2] = 1.0
    return x


@pytest.mark.smoke
def test_the_argmax_configuration_is_infeasible():
    """The property the fallback exists for (else the test below proves nothing)."""
    assert subnlp(_argmax_infeasible_gdp(), _argmax_config_point()) is None


@pytest.mark.smoke
def test_ranked_searches_miss_and_the_fallback_finds_a_point_within_its_attempt_budget():
    """The regression the issue asks for: a fixed ATTEMPT budget, not a clock.

    The ranked constructor, out of time, finds nothing (the fallback no longer
    lives inside it, so it cannot fire there); the fallback, given no deadline at
    all, returns a verified point within ``_FEAS_SAMPLE_ATTEMPTS`` sub-NLP solves.
    """
    m = _argmax_infeasible_gdp()
    expired = time.perf_counter() - 1.0
    assert one_hot_config_subnlp(m, _relaxation_point(), deadline=expired) == []

    stats: dict = {}
    found = one_hot_config_feasibility_fallback(m, _relaxation_point(), stats=stats)
    assert len(found) == 1
    assert 1 <= stats["attempted"] <= ph._FEAS_SAMPLE_ATTEMPTS
    assert stats["first_feasible_at"] == stats["attempted"]
    x, obj = found[0]
    assert max_constraint_violation(m, x) <= 1e-6
    y = np.round(x[: 2 * N_GROUPS])
    assert np.all(y[0::2] + y[1::2] == 1.0)
    assert y[0::2].sum() <= 6.0  # the resource row: at most 6 cheap picks
    assert np.isfinite(obj)


@pytest.mark.smoke
def test_the_fallback_is_reproducible():
    a = one_hot_config_feasibility_fallback(_argmax_infeasible_gdp(), _relaxation_point())
    b = one_hot_config_feasibility_fallback(_argmax_infeasible_gdp(), _relaxation_point())
    assert len(a) == len(b) == 1 and np.array_equal(a[0][0], b[0][0])


@pytest.mark.smoke
def test_fallback_stops_at_its_attempt_cap_when_nothing_is_feasible():
    """Every configuration infeasible: the fallback spends exactly its count."""
    m = _argmax_infeasible_gdp()
    m.subject_to(m._variables[1][0] >= 1.0, name="contradiction")  # x[0] >= 1 ...
    m.subject_to(m._variables[1][0] <= 0.5, name="contradiction2")  # ... and <= 0.5
    stats: dict = {}
    assert one_hot_config_feasibility_fallback(m, _relaxation_point(), stats=stats) == []
    assert stats["attempted"] == ph._FEAS_SAMPLE_ATTEMPTS


class _Tree:
    def __init__(self, inc):
        self._inc = inc

    def incumbent(self):
        return self._inc


@pytest.mark.smoke
def test_the_solver_runs_it_last_and_only_when_switched_on(monkeypatch):
    """It must not displace the constructors gated on "no incumbent yet": measured
    on mod_hens big-M, firing at the root gave 110441.39 where the pump finds
    106766.53. So: past the root, after the deferred call, with no incumbent."""
    seed = np.zeros(3)
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_FALLBACK", "1")
    assert _gdp_feas_fallback_due(seed, None, 1, _Tree(None))
    assert not _gdp_feas_fallback_due(seed, None, 0, _Tree(None))  # root batch
    assert not _gdp_feas_fallback_due(seed, {"ready_at": 0.0}, 5, _Tree(None))  # deferred
    assert not _gdp_feas_fallback_due(seed, None, 5, _Tree((seed, 1.0)))  # incumbent
    assert not _gdp_feas_fallback_due(None, None, 5, _Tree(None))  # ran / never armed
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_FALLBACK", "0")
    assert not _gdp_feas_fallback_due(seed, None, 1, _Tree(None))
