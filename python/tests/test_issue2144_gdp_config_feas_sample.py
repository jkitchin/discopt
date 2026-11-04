"""Regression tests for issue #2144 — the GDP configuration search's
feasibility-first fallback.

#2144 measured on GDPlib ``batch_processing`` that the root relaxation is
confident in the *wrong* disjunct of every group: its per-group argmax and every
least-confident flip the ranked wave reaches are infeasible, while a uniformly
random valid configuration solves feasibly 54 times in 60. The solve ended with
no incumbent under both big-M and hull.

The synthetic model here has the same property, gated on structure and not on a
name (CLAUDE.md §2): ``N_GROUPS`` two-way disjunctions, a cost that pulls every
group to disjunct 0, and a row that needs at least ``NEED`` groups on disjunct 1.
The ranked wave enumerates by Hamming distance from the argmax, and its
``_WAVE_SOLVE_CAP`` solves never reach distance ``NEED``; a uniform sample is
feasible with probability ``1 - sum_{k<NEED} C(N,k) / 2**N`` (0.945 here).

The budget is stated in sub-NLP **attempts**, not wall-clock, as the issue asks.
"""

from __future__ import annotations

from math import comb

import discopt._relax.primal_heuristics as ph
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.primal_heuristics import _FEAS_SAMPLE_CAP, _WAVE_SOLVE_CAP

N_GROUPS = 10
_REAL_SUBNLP = ph.subnlp
NEED = 3


def _argmax_infeasible_gdp():
    m = dm.Model("argmax_infeasible")
    y = m.binary("y", 2 * N_GROUPS)
    x = m.continuous("x", lb=0.0, ub=0.5)
    for i in range(N_GROUPS):
        m.subject_to(y[2 * i] + y[2 * i + 1] == 1, name=f"pick_{i}")
    m.subject_to(sum(y[2 * i + 1] for i in range(N_GROUPS)) + x >= NEED, name="need")
    m.minimize(sum((1.0 + 0.1 * i) * y[2 * i + 1] for i in range(N_GROUPS)) + x**2)
    return m


def _relaxation_point() -> np.ndarray:
    """Confident in disjunct 0 of every group -- the batch_processing shape."""
    x = np.empty(2 * N_GROUPS + 1)
    for i in range(N_GROUPS):
        conf = 0.95 - 0.01 * i  # distinct margins, so the wave's order is defined
        x[2 * i], x[2 * i + 1] = conf, 1.0 - conf
    x[-1] = 0.25
    return x


def _run(monkeypatch, flag: str) -> tuple[list[bool], list]:
    """Run the wave with the real ``subnlp``; record each attempt's outcome."""
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_SAMPLE", flag)
    outcomes: list[bool] = []
    real = _REAL_SUBNLP

    def _counted(*a, **kw):
        found = real(*a, **kw)
        outcomes.append(found is not None)
        return found

    monkeypatch.setattr(ph, "subnlp", _counted)
    # The dive is a separate mechanism; this pins the wave and its fallback.
    monkeypatch.setattr(ph, "one_hot_config_dive", lambda *a, **kw: [])
    stats: dict = {}
    found = ph.one_hot_config_subnlp(_argmax_infeasible_gdp(), _relaxation_point(), stats=stats)
    assert stats["attempted"] == len(outcomes), "probe vacuous: attempts not all counted"
    return outcomes, found


def test_premise_the_ranked_wave_cannot_reach_a_feasible_configuration(monkeypatch):
    """Sampling OFF reproduces #2144: a full wave, nothing feasible."""
    distance_below_need = sum(comb(N_GROUPS, k) for k in range(NEED))
    assert distance_below_need >= _WAVE_SOLVE_CAP, "premise stale: wave reaches NEED"
    outcomes, found = _run(monkeypatch, "0")
    assert len(outcomes) == _WAVE_SOLVE_CAP, f"wave ran {len(outcomes)} solves"
    assert not any(outcomes) and found == []


@pytest.mark.smoke
def test_feasibility_first_sampling_finds_a_point_within_an_attempt_budget(monkeypatch):
    outcomes, found = _run(monkeypatch, "1")
    assert outcomes, "probe vacuous: no sub-NLP ran"
    assert found, "no feasible configuration found"
    first = outcomes.index(True) + 1
    # Anchor (ranked plan 0), then one sample per ranked plan. P(sample feasible) is
    # 0.945, so a correct fallback hits in the first couple of samples; 6 attempts
    # is three samples, failing with probability ~1.7e-4 for a fair RNG.
    assert first <= 6, f"first feasible point at attempt {first}"
    for x, obj in found:
        ones = int(round(sum(x[2 * i + 1] for i in range(N_GROUPS))))
        assert ones + x[-1] >= NEED - 1e-6
        assert np.isfinite(obj)


@pytest.mark.smoke
def test_sampling_stops_at_the_first_feasible_point_and_is_capped(monkeypatch):
    outcomes, _ = _run(monkeypatch, "1")
    first = outcomes.index(True)
    # After the first hit the wave alone continues, so the total is the wave cap
    # plus the samples spent before the hit -- never more than _FEAS_SAMPLE_CAP.
    samples = len(outcomes) - _WAVE_SOLVE_CAP
    assert 1 <= samples <= _FEAS_SAMPLE_CAP
    assert samples == (first + 1) // 2, (outcomes[: first + 1], samples)


@pytest.mark.smoke
def test_sampling_is_reproducible(monkeypatch):
    a, fa = _run(monkeypatch, "1")
    b, fb = _run(monkeypatch, "1")
    assert a == b
    assert [round(o, 9) for _, o in fa] == [round(o, 9) for _, o in fb]


@pytest.mark.smoke
def test_no_sampling_when_the_anchor_is_feasible(monkeypatch):
    """Where the relaxation is right, the ranked wave is exactly as before."""
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_SAMPLE", "1")
    monkeypatch.setattr(ph, "one_hot_config_dive", lambda *a, **kw: [])
    x = _relaxation_point()
    for i in range(NEED):  # the argmax now puts NEED groups on disjunct 1
        x[2 * i], x[2 * i + 1] = x[2 * i + 1], x[2 * i]
    stats: dict = {}
    found = ph.one_hot_config_subnlp(_argmax_infeasible_gdp(), x, stats=stats)
    assert found
    assert stats["sampled"] == 0
