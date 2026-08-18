"""Regression test for #1062: the GDP config DIVE must not be gated on wave failure.

``one_hot_config_subnlp`` reached ``one_hot_config_dive`` through::

    if results or not groups:
        return results

so the dive ran *only when the wave returned nothing*. On the syn/rsyn family the
wave always returns a feasible-but-poor plan, so that branch never fired and the
search written for exactly those models was unreachable.

Measured at the production envelope (9 s shared by both searches, i.e. what
``_gdp_config_deadline`` grants at a 60 s limit), best objective in the solver's
internal minimise convention, wave alone vs. wave + dive:

    rsyn0805m   -321.96  ->  -1267.38      syn40m      +0.95  ->   -23.16
    rsyn0840m    -11.06  ->    -92.11      syn20m02m  -636.72 ->  -636.72 (tie)

For reference the published optima are 1296.12 / 67.71 / 325.55 / 1752.13 in the
models' own maximise sense, and a full 60 s solve before this change returned
1116.46 / 33.20 / 37.59 / 636.72.

The tests below assert the *mechanism* — the dive is reached even when the wave
succeeds, and its points are merged rather than discarded — on a small one-hot
model, so nothing here depends on a named instance or on any model being hard
enough on the day (CLAUDE.md §2).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph


def _one_hot_model():
    """A model whose disjunction the one-hot scanner actually recognises.

    ``_scan_one_hot_rows`` requires an ``==`` row of unit-coefficient binaries
    summing to 1, so this is a genuine structural match rather than a stand-in.
    """
    m = dm.Model("config_dive")
    y = m.binary("y", 3)
    x = m.continuous("x", 3, lb=0.0, ub=4.0)
    m.subject_to(y[0] + y[1] + y[2] == 1, name="pick_one")
    for i in range(3):
        m.subject_to(x[i] <= 4.0 * y[i], name=f"link{i}")
    m.subject_to(x[0] + x[1] + x[2] >= 1.0, name="demand")
    m.minimize(sum((i + 1) * x[i] * x[i] for i in range(3)) + y[0] + 2.0 * y[1] + 3.0 * y[2])
    return m


def _seed_for(model):
    from discopt._relax.nlp_evaluator import NLPEvaluator

    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    return evaluator, 0.5 * (lb + ub)


@pytest.mark.smoke
def test_one_hot_groups_are_detected(monkeypatch):
    """§6 guard: without a recognised disjunction the tests below are vacuous."""
    model = _one_hot_model()
    int_mask = ph._get_integer_mask(model)
    groups = ph._scan_one_hot_rows(model, int_mask, int(int_mask.size))
    assert groups, "no one-hot group detected — the dive tests would prove nothing"


@pytest.mark.smoke
def test_dive_runs_even_when_the_wave_found_points(monkeypatch):
    """The defect: a successful wave used to make the dive unreachable."""
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _one_hot_model()
    evaluator, seed = _seed_for(model)

    calls: list[int] = []
    sentinel_obj = -12345.0
    sentinel_x = np.zeros(int(np.asarray(seed).size), dtype=np.float64)

    def spy_dive(*args, **kwargs):
        calls.append(1)
        return [(sentinel_x.copy(), sentinel_obj)]

    monkeypatch.setattr(ph, "one_hot_config_dive", spy_dive)

    results = ph.one_hot_config_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=None,
    )

    # The wave has to have succeeded, or this test is measuring the old
    # empty-wave branch and proves nothing about the change (§6).
    wave_points = [r for r in results if float(r[1]) != sentinel_obj]
    assert wave_points, "the wave found nothing — this test did not exercise the fix"

    assert calls, (
        "one_hot_config_dive was never called even though the wave succeeded — "
        "the dive is still gated behind wave failure (#1062)"
    )
    # Merged, not substituted: the wave's own points must survive alongside it.
    assert any(float(o) == sentinel_obj for _, o in results), (
        "the dive ran but its points were discarded rather than merged"
    )


@pytest.mark.smoke
def test_dive_still_runs_when_the_wave_finds_nothing(monkeypatch):
    """The pre-existing #993 path must keep working — this is not a swap."""
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _one_hot_model()
    evaluator, seed = _seed_for(model)

    calls: list[int] = []

    def spy_dive(*args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(ph, "one_hot_config_dive", spy_dive)
    # A deadline already in the past stops the wave before its first sub-NLP, which
    # is the "wave returned nothing" state #993 wired the dive up for.
    ph.one_hot_config_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=-1.0,
    )
    assert calls, "the dive is no longer reached on the empty-wave path (#993 regression)"
