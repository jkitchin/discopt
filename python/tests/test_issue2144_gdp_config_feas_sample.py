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


N_RESID = 4


def _argmax_infeasible_gdp(resid: bool = False):
    m = dm.Model("argmax_infeasible")
    y = m.binary("y", 2 * N_GROUPS)
    x = m.continuous("x", lb=0.0, ub=0.5)
    for i in range(N_GROUPS):
        m.subject_to(y[2 * i] + y[2 * i + 1] == 1, name=f"pick_{i}")
    m.subject_to(sum(y[2 * i + 1] for i in range(N_GROUPS)) + x >= NEED, name="need")
    if resid:
        # Binaries outside every one-hot group, coupled by an integer-only row (the
        # shape of a nested-indicator or logic row): 5 of 16 0/1 draws satisfy it.
        z = m.binary("z", N_RESID)
        m.subject_to(sum(z[k] for k in range(N_RESID)) <= 1, name="logic")
    m.minimize(sum((1.0 + 0.1 * i) * y[2 * i + 1] for i in range(N_GROUPS)) + x**2)
    return m


def _relaxation_point(resid: bool = False) -> np.ndarray:
    """Confident in disjunct 0 of every group -- the batch_processing shape."""
    x = np.zeros(2 * N_GROUPS + 1 + (N_RESID if resid else 0))
    for i in range(N_GROUPS):
        conf = 0.95 - 0.01 * i  # distinct margins, so the wave's order is defined
        x[2 * i], x[2 * i + 1] = conf, 1.0 - conf
    x[2 * N_GROUPS] = 0.25
    return x


def _run(
    monkeypatch, flag: str, *, resid: bool = False, sample_stream: int = 0, seeds=None
) -> tuple[list[bool], list, dict]:
    """Run the wave with the real ``subnlp``; record each attempt's outcome."""
    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FEAS_SAMPLE", flag)
    outcomes: list[bool] = []
    real = _REAL_SUBNLP

    def _counted(model, x0, *a, **kw):
        if seeds is not None:
            seeds.append(
                tuple(
                    np.round(
                        np.asarray(x0)[:-1]
                        if not resid
                        else np.delete(np.asarray(x0), 2 * N_GROUPS)
                    )
                    .astype(int)
                    .tolist()
                )
            )
        found = real(model, x0, *a, **kw)
        outcomes.append(found is not None)
        return found

    monkeypatch.setattr(ph, "subnlp", _counted)
    # The dive is a separate mechanism; this pins the wave and its fallback.
    monkeypatch.setattr(ph, "one_hot_config_dive", lambda *a, **kw: [])
    stats: dict = {}
    found = ph.one_hot_config_subnlp(
        _argmax_infeasible_gdp(resid),
        _relaxation_point(resid),
        stats=stats,
        sample_stream=sample_stream,
    )
    assert stats["attempted"] == len(outcomes), "probe vacuous: attempts not all counted"
    return outcomes, found, stats


def _sample_slots(n_attempts: int) -> list[int]:
    """0-based attempt indices at which the doubling schedule draws a sample, for a
    run with nothing found: after ranked plans 1, 2, 4, 8, ... (then one per step
    once the wave is exhausted)."""
    slots, k, i = [], 0, 0
    while i < n_attempts:
        if k < _WAVE_SOLVE_CAP:
            k += 1
            i += 1
            if k & (k - 1) == 0:
                slots.append(i)
                i += 1
        else:
            slots.append(i)
            i += 1
    return [s for s in slots if s < n_attempts]


def test_premise_the_ranked_wave_cannot_reach_a_feasible_configuration(monkeypatch):
    """Sampling OFF reproduces #2144: a full wave, nothing feasible."""
    distance_below_need = sum(comb(N_GROUPS, k) for k in range(NEED))
    assert distance_below_need >= _WAVE_SOLVE_CAP, "premise stale: wave reaches NEED"
    outcomes, found, _ = _run(monkeypatch, "0")
    assert len(outcomes) == _WAVE_SOLVE_CAP, f"wave ran {len(outcomes)} solves"
    assert not any(outcomes) and found == []


@pytest.mark.smoke
def test_feasibility_first_sampling_finds_a_point_within_an_attempt_budget(monkeypatch):
    outcomes, found, _ = _run(monkeypatch, "1")
    assert outcomes, "probe vacuous: no sub-NLP ran"
    assert found, "no feasible configuration found"
    first = outcomes.index(True) + 1
    # Samples come after ranked plans 1, 2 and 4, i.e. at attempts 2, 4 and 7. A
    # uniform sample is feasible with probability 0.945, so any stream hits within
    # those three except with probability ~1.7e-4; the stream is fixed, so this run
    # is deterministic.
    assert first <= 7, f"first feasible point at attempt {first}"
    for x, obj in found:
        ones = int(round(sum(x[2 * i + 1] for i in range(N_GROUPS))))
        assert ones + x[-1] >= NEED - 1e-6
        assert np.isfinite(obj)


@pytest.mark.smoke
def test_sampling_stops_at_the_first_feasible_point_and_is_capped(monkeypatch):
    outcomes, _, stats = _run(monkeypatch, "1")
    first = outcomes.index(True)
    # After the first hit the wave alone continues, so the total is the wave cap
    # plus the samples spent up to the hit -- never more than _FEAS_SAMPLE_CAP.
    samples = len(outcomes) - _WAVE_SOLVE_CAP
    assert 1 <= samples <= _FEAS_SAMPLE_CAP
    assert samples == stats["sampled"]
    assert samples == len(_sample_slots(first + 1)), (outcomes[: first + 1], samples)


def test_schedule_costs_the_ranked_wave_at_most_a_log(monkeypatch):
    """Samples share the wave's deadline, so a 1:1 interleave would halve how deep
    the ranked wave gets in a binding slice. With the doubling schedule, ranked plan
    k is reached after at most k + floor(log2 k) + 1 solves."""
    for k in (1, 2, 3, 8, 17, _WAVE_SOLVE_CAP):
        n = k
        while n - len(_sample_slots(n)) < k:
            n += 1
        assert n <= k + k.bit_length(), (k, n)


@pytest.mark.smoke
def test_sampling_is_reproducible(monkeypatch):
    a, fa, sa = _run(monkeypatch, "1")
    b, fb, _ = _run(monkeypatch, "1")
    assert sa["sampled"] > 0, "vacuous: nothing was sampled"
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


def test_a_deferred_call_draws_a_different_sample_stream(monkeypatch):
    """The deferred call starts from the root call's relaxation point; with one
    shared stream it redrew, and re-solved, exactly the root's samples."""
    monkeypatch.setattr(ph, "_FEAS_SAMPLE_CAP", 4)
    root: list = []
    deferred: list = []
    # Every sample infeasible, so all four are drawn in both calls.
    monkeypatch.setattr(ph, "_refuted_by_integer_rows", lambda *a, **kw: False)
    _run(monkeypatch, "1", seeds=root)
    _run(monkeypatch, "1", seeds=deferred, sample_stream=1)
    off: list = []
    _run(monkeypatch, "0", seeds=off)
    ranked = set(off)
    s_root = [x for x in root if x not in ranked]
    s_def = [x for x in deferred if x not in ranked]
    assert s_root and s_def, "vacuous: no sample identified"
    assert s_root != s_def


def test_draws_breaking_an_integer_only_row_are_rejected_unsolved(monkeypatch):
    """Residual binaries are drawn 0/1; a draw that breaks an integer-only row is
    refused by ``subnlp`` without a solve, so it must not spend the sample cap."""
    seeds: list = []
    off: list = []
    _, found, stats = _run(monkeypatch, "1", resid=True, seeds=seeds)
    _run(monkeypatch, "0", resid=True, seeds=off)
    ranked = set(off)
    samples = [x for x in seeds if x not in ranked]
    assert stats["sampled"] >= 1 and samples, "vacuous: nothing was sampled"
    assert stats["sample_rejected"] >= 1, "vacuous: no draw broke the logic row"
    for x in samples:
        assert sum(x[-N_RESID:]) <= 1, x
    assert found
