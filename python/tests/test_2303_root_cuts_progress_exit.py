"""#2303: a relative-progress exit for NLP-BB's root cut loop.

The stall exit (``STALL_ROUNDS``) fires only after rounds that gain nothing, so a loop
whose every round gains a little runs on: 102-162 rounds on #2303's GDP depending on
the BLAS kernel's round-off. ``DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_EXIT`` stops it once the
last ``K`` rounds closed less than ``frac`` of what the stage has closed so far.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.solvers import _root_cuts as RC


def _synthesis_gdp() -> dm.Model:
    reactors = [(22.0, 120.0, 2.0), (29.0, 260.0, 2.5), (32.0, 380.0, 2.4)]
    seps = [(0.95, 150.0, 1.2), (0.90, 60.0, 1.6)]
    m = dm.Model("synthesis_nl")
    f, p = m.continuous("F", lb=0, ub=100), m.continuous("P", lb=0, ub=100)
    s = m.continuous("S", lb=0, ub=60)
    cr, cs = m.continuous("CR", lb=0, ub=600), m.continuous("CS", lb=0, ub=300)
    yr_b, ys_b = m.boolean("YR", shape=3), m.boolean("YS", shape=2)
    yr, ys = yr_b.variable, ys_b.variable
    m.either_or(
        [
            [p <= a * dm.log(1 + f / 10), cr >= fx + v * f, yr[k] == 1]
            for k, (a, fx, v) in enumerate(reactors)
        ],
        name="reactor",
    )
    m.either_or(
        [[s == rec * p, cs >= fx + v * p, ys[k] == 1] for k, (rec, fx, v) in enumerate(seps)],
        name="separator",
    )
    m.exactly(1, [yr[k] for k in range(3)])
    m.exactly(1, [ys[k] for k in range(2)])
    m.logical(yr_b[1].implies(~ys_b[1]))
    m.maximize(30.0 * s - 9.0 * f - cr - cs)
    return m


def _solve_with_spy(monkeypatch):
    got = {}
    real = RC.generate_root_cuts

    def spy(*a, **k):
        got["r"] = real(*a, **k)
        return got["r"]

    monkeypatch.setattr(RC, "generate_root_cuts", spy)
    r = _synthesis_gdp().solve(gdp_method="big-m")
    assert "r" in got, "the root-cut stage never ran (CLAUDE.md §6)"
    return r, got["r"]


@pytest.fixture
def _nlpbb(monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    for k in (
        "DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_EXIT",
        "DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_K",
        "DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_FRAC",
    ):
        monkeypatch.delenv(k, raising=False)


def test_exit_fires_and_keeps_a_valid_bound(monkeypatch, _nlpbb):
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_EXIT", "1")
    r, rc = _solve_with_spy(monkeypatch)
    assert rc.stop_reason == "progress", (rc.rounds_run, rc.stop_reason)
    # A maximization: the stage's bound must stay at or above the optimum.
    assert rc.lp_bound is None or rc.lp_bound >= 487.7362 - 1e-6
    # Monotone trace: the exit only ever stops a loop whose bound only improves.
    assert all(b >= a - 1e-6 for a, b in zip(rc.bound_trace[1:], rc.bound_trace[:-1]))
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(487.7362, abs=1e-3)


def test_off_never_stops_on_progress(monkeypatch, _nlpbb):
    _, rc = _solve_with_spy(monkeypatch)
    assert rc.stop_reason != "progress"


@pytest.mark.parametrize("k,frac", [("0", "0.01"), ("5", "0"), ("5", "-1")])
def test_bad_knobs_are_refused(monkeypatch, k, frac):
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_EXIT", "1")
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_K", k)
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS_PROGRESS_FRAC", frac)
    with pytest.raises(ValueError, match="progress exit"):
        RC.root_cuts_progress_exit()
