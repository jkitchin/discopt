"""#2144: the GDP configuration search's feasibility-first fallback.

On GDPlib batch_processing the relaxation's per-group argmax is infeasible and the
feasible configurations sit 15+ demotions away from it, so the ranked wave and the
dive (both steered by the relaxation) return nothing, while ~90 % of *uniformly*
sampled valid configurations are feasible. ``DISCOPT_GDP_CONFIG_FEAS_FALLBACK``
samples when both ranked searches miss, on a budget of its own: a fixed number of
sub-NLP **attempts**, not a share of the wave's deadline (the #2150 lesson -- a
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
from discopt._relax.primal_heuristics import one_hot_config_subnlp, subnlp
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
def test_fallback_finds_a_point_on_its_own_attempt_budget(monkeypatch):
    """Fails without the fallback: the wave's deadline has already passed, so the
    ranked searches issue nothing. The fallback's budget is a count, independent
    of that deadline, and it returns a verified feasible point within it."""
    m = _argmax_infeasible_gdp()
    expired = time.perf_counter() - 1.0

    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_FALLBACK", "0")
    off_stats: dict = {}
    off = one_hot_config_subnlp(m, _relaxation_point(), deadline=expired, stats=off_stats)
    assert off == [] and off_stats["attempted"] == 0

    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_FALLBACK", "1")
    stats: dict = {}
    found = one_hot_config_subnlp(m, _relaxation_point(), deadline=expired, stats=stats)
    assert len(found) == 1
    assert 1 <= stats["fallback_attempted"] <= ph._FEAS_SAMPLE_ATTEMPTS
    assert stats["attempted"] == stats["fallback_attempted"]
    x, obj = found[0]
    assert max_constraint_violation(m, x) <= 1e-6
    y = np.round(x[: 2 * N_GROUPS])
    assert np.all(y[0::2] + y[1::2] == 1.0)
    assert y[0::2].sum() <= 6.0  # the resource row: at most 6 cheap picks
    assert np.isfinite(obj)


@pytest.mark.smoke
def test_fallback_is_skipped_for_a_caller_holding_an_incumbent(monkeypatch):
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_FALLBACK", "1")
    stats: dict = {}
    found = one_hot_config_subnlp(
        _argmax_infeasible_gdp(),
        _relaxation_point(),
        deadline=time.perf_counter() - 1.0,
        stats=stats,
        feasibility_fallback=False,
    )
    assert found == [] and stats["attempted"] == 0 and "fallback_attempted" not in stats


@pytest.mark.smoke
def test_fallback_stops_at_its_attempt_cap_when_nothing_is_feasible(monkeypatch):
    """Every configuration infeasible: the fallback spends exactly its count."""
    m = _argmax_infeasible_gdp()
    m.subject_to(m._variables[1][0] >= 1.0, name="contradiction")  # x[0] >= 1 ...
    m.subject_to(m._variables[1][0] <= 0.5, name="contradiction2")  # ... and <= 0.5
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_FALLBACK", "1")
    stats: dict = {}
    found = one_hot_config_subnlp(
        m, _relaxation_point(), deadline=time.perf_counter() - 1.0, stats=stats
    )
    assert found == []
    assert stats["fallback_attempted"] == ph._FEAS_SAMPLE_ATTEMPTS
