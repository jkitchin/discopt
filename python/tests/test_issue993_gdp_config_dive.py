"""Regression tests for issue #993 — the disjunct-configuration *reachability* gap.

``one_hot_config_subnlp`` (#823) fixed *validity*: it selects one disjunct per
``sum_k y_k == 1`` row, so the fixing it hands the sub-NLP never contradicts a
disjunction outright. It did not fix *reachability*. Both of its searches rank
disjuncts from a single static point — the relaxation's per-group argmax — and
enumerate outward in Hamming waves without ever re-solving. When the answer sits
many demotions away, no wave budget reaches it:

==================  ======  ========================  ========================
model               groups  demotions from the argmax  plan index in the wave
==================  ======  ========================  ========================
syngas                  26  3                          743 (``max_configs`` 256)
batch_processing        29  15                         C(29,15) ~ 7.7e7
==================  ======  ========================  ========================

Nothing else was the cause. With the integers pinned to BARON's proven
configuration the fixed-integer sub-NLP solves in 0.07 s (batch_processing, at the
reference optimum 679365.33) and 0.36 s (syngas, 4669.0235) *from the
constructor's own zero start*, so the start point, the presolved bounds, the seed
ranking and the environment were each eliminated as explanations.

``one_hot_config_dive`` re-solves the relaxation between choices, which is what
makes distant configurations reachable: once a prefix of cheap disjuncts is pinned,
the remaining fractional indicators move, and a prefix that cannot be completed
makes the relaxation infeasible — information a wave around a fixed point never
receives.

These tests pin the mechanism on a synthetic model whose *structure* creates the
gap, not on any named instance (CLAUDE.md §2): a capacity GDP where the objective
pulls every group's argmax to the cheap disjunct at once, so the integral answer
requires many simultaneous demotions.
"""

from __future__ import annotations

import time
from math import comb

import discopt._relax.primal_heuristics as ph
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.primal_heuristics import (
    _WAVE_SOLVE_CAP,
    _check_constraint_feasibility,
    _get_variable_bounds,
    _scan_one_hot_rows,
    cached_evaluator,
    one_hot_config_dive,
    one_hot_config_subnlp,
)

N_GROUPS = 20
CAP = 5.0
REQUIRED = 45.0
#: Cheapest integral solution: ceil(REQUIRED/CAP) groups must take the expensive
#: disjunct, the rest the cheap one.
N_EXPENSIVE = 9
OPTIMUM = (N_GROUPS - N_EXPENSIVE) * 1.0 + N_EXPENSIVE * 10.0


def _capacity_gdp():
    """A capacity GDP whose relaxation prefers a configuration that is infeasible.

    Each of ``N_GROUPS`` disjunctions chooses between a cheap disjunct (cost 1, no
    capacity) and an expensive one (cost 10, capacity ``CAP``), and the total
    capacity must reach ``REQUIRED``. The big-M row ``p_i <= CAP * y_expensive``
    is *slack at fractional y*, so the relaxation buys the required capacity with
    twenty half-open expensive disjuncts and every group's argmax still reads
    cheap — while any integral solution needs ``N_EXPENSIVE`` of them fully open.

    That is the shape of the real failure: the argmax configuration is infeasible
    and the answer is ``N_EXPENSIVE`` demotions away, not one or two.
    """
    m = dm.Model("capacity_gdp")
    y = m.binary("y", 2 * N_GROUPS)  # y[2i] cheap, y[2i+1] expensive
    p = m.continuous("p", N_GROUPS, lb=0.0, ub=CAP)

    total = p[0]
    cost = y[0] + 10.0 * y[1]
    for i in range(N_GROUPS):
        m.subject_to(y[2 * i] + y[2 * i + 1] == 1, name=f"disj{i}")
        m.subject_to(p[i] - CAP * y[2 * i + 1] <= 0.0, name=f"bigm{i}")
        if i:
            total = total + p[i]
            cost = cost + y[2 * i] + 10.0 * y[2 * i + 1]
    m.subject_to(total >= REQUIRED, name="demand")
    m.minimize(cost)
    return m


def _relaxation_point():
    """The analytic LP relaxation optimum, so the tests need no solve to start.

    Every group splits identically: expensive share ``REQUIRED/(CAP*N_GROUPS)``,
    each ``p_i`` equal. Written out rather than solved so the arithmetic behind
    "the argmax reads cheap" is visible and the test is deterministic.
    """
    frac = REQUIRED / (CAP * N_GROUPS)
    x = np.zeros(3 * N_GROUPS)
    for i in range(N_GROUPS):
        x[2 * i] = 1.0 - frac
        x[2 * i + 1] = frac
        x[2 * N_GROUPS + i] = REQUIRED / N_GROUPS
    return x


@pytest.mark.smoke
def test_the_answer_is_out_of_reach_of_any_wave_budget():
    """Pin *why* the wave fails here: it is arithmetic, not tuning.

    The argmax configuration (all cheap) supplies zero capacity, so every feasible
    configuration demotes at least ``N_EXPENSIVE`` groups. A distance enumeration
    must exhaust all nearer waves first, and that count dwarfs any sane
    ``max_configs`` — so raising the budget is the same wrong mechanism with a
    bigger constant, which is what #993 records.
    """
    x_relax = _relaxation_point()
    m = _capacity_gdp()
    mask = np.zeros(3 * N_GROUPS, dtype=bool)
    mask[: 2 * N_GROUPS] = True

    groups = _scan_one_hot_rows(m, mask, mask.size)
    assert len(groups) == N_GROUPS, groups

    # The relaxation prefers the cheap disjunct in every single group.
    for i, g in enumerate(groups):
        assert max(g, key=lambda j: x_relax[j]) == 2 * i, f"group {i} argmax is not cheap"

    plans_before_the_answer = sum(comb(N_GROUPS, d) for d in range(N_EXPENSIVE))
    assert plans_before_the_answer > 256 * 1000, plans_before_the_answer


@pytest.mark.smoke
def test_dive_finds_a_point_no_wave_around_one_point_can_reach():
    """The #993 regression: re-solving between choices reaches the answer.

    Before the fix this returns ``[]`` — the wave exhausts ``max_configs`` inside
    the first three demotion levels and never reaches level nine.
    """
    m = _capacity_gdp()
    found = one_hot_config_subnlp(m, _relaxation_point(), deadline=time.perf_counter() + 60.0)
    assert found, "no feasible configuration found; the dive did not reach the answer"

    ev = cached_evaluator(m)
    for x, obj in found:
        x = np.asarray(x, dtype=np.float64)
        assert np.isfinite(obj)
        # Verified independently, not trusted because the heuristic returned it.
        assert _check_constraint_feasibility(ev, x), "returned an infeasible point"
        for i in range(N_GROUPS):
            assert abs(x[2 * i] + x[2 * i + 1] - 1.0) < 1e-6, f"group {i} is not one-hot"
            assert abs(x[2 * i] - round(x[2 * i])) < 1e-5, f"group {i} is fractional"
        # A primal heuristic may not beat the true optimum: that would be a bound
        # violation dressed up as a good incumbent.
        assert obj >= OPTIMUM - 1e-6, f"objective {obj} beats the optimum {OPTIMUM}"


@pytest.mark.smoke
def test_dive_restores_every_bound_it_touched():
    """The dive fixes indicators by rewriting variable bounds; all of it is temporary."""
    m = _capacity_gdp()
    lb_before, ub_before = _get_variable_bounds(m)
    one_hot_config_dive(m, _relaxation_point(), deadline=time.perf_counter() + 20.0)
    lb_after, ub_after = _get_variable_bounds(m)
    assert np.array_equal(lb_before, lb_after), "lower bounds leaked out of the dive"
    assert np.array_equal(ub_before, ub_after), "upper bounds leaked out of the dive"


@pytest.mark.smoke
def test_dive_respects_an_expired_deadline():
    """A past deadline stops it before the first relaxation solve."""
    m = _capacity_gdp()
    t0 = time.perf_counter()
    assert one_hot_config_dive(m, _relaxation_point(), deadline=t0 - 1.0) == []
    assert time.perf_counter() - t0 < 1.0, "an expired deadline still cost a solve"


@pytest.mark.smoke
def test_dive_is_a_noop_without_one_hot_structure():
    """Generality: gated on detected structure, never on a name (CLAUDE.md §2)."""
    m = dm.Model("plain")
    z = m.binary("z", 3)
    m.subject_to(z[0] + z[1] + z[2] <= 2, name="c")
    m.minimize(z[0] + z[1] + z[2])
    assert one_hot_config_dive(m, np.array([0.4, 0.4, 0.4])) == []


@pytest.mark.smoke
def test_dive_is_deterministic_for_a_fixed_seed():
    """Reproducibility: the randomised restarts are seeded, so runs are comparable.

    A heuristic that varies run to run makes node counts unreproducible, which
    would break the §5 bound-neutral comparison the panel relies on.
    """
    x_relax = _relaxation_point()
    a = one_hot_config_dive(_capacity_gdp(), x_relax, max_restarts=3, deadline=None)
    b = one_hot_config_dive(_capacity_gdp(), x_relax, max_restarts=3, deadline=None)
    assert [round(o, 9) for _, o in a] == [round(o, 9) for _, o in b]


@pytest.mark.smoke
def test_wave_hands_over_on_a_solve_count_not_a_clock_slice(monkeypatch):
    """#912: how much of the wave runs must not be a function of machine speed.

    The first wiring reserved half the *wall* grant for the dive
    (``wave_deadline = _now() + 0.5 * (deadline - _now())``), so a slower box tried
    fewer configurations — a different incumbent, and from there a different search
    tree, for the same model. ``_WAVE_SOLVE_CAP`` denominates the handover in
    sub-NLP solves instead. Driving the ``_now()`` seam with two clocks that differ
    by an unbounded factor pins that: the count is identical, and the deadline is
    left doing only the job #912 allows it to do — deciding when to stop.
    """
    m = _capacity_gdp()
    x_relax = _relaxation_point()
    deadline = 120.0  # in the fake clock's own units, not seconds

    def _arm(tick: float) -> tuple[int, float]:
        reading = [0.0]

        def _fake_now() -> float:
            reading[0] += tick
            return reading[0]

        tried: list[object] = []

        def _never_feasible(model, seed, **kwargs):
            # Return nothing so the wave runs to its own bound rather than stopping
            # early on a hit; the dive is a separate mechanism, stubbed out here.
            tried.append(kwargs.get("time_budget"))
            return None

        monkeypatch.setattr(ph, "_now", _fake_now)
        monkeypatch.setattr(ph, "subnlp", _never_feasible)
        monkeypatch.setattr(ph, "one_hot_config_dive", lambda *a, **kw: [])
        ph.one_hot_config_subnlp(m, x_relax, deadline=deadline)
        return len(tried), reading[0]

    slow, slow_clock = _arm(1.0)
    fast, _ = _arm(0.0)

    assert slow == fast == _WAVE_SOLVE_CAP, (
        f"the wave tried {slow} configuration(s) on the slow clock and {fast} on the "
        f"fast one; both must be _WAVE_SOLVE_CAP={_WAVE_SOLVE_CAP}, or how far the "
        "wave gets is a function of the machine"
    )
    # Two checks that a *passing* run was not vacuous (CLAUDE.md §6). The slow arm
    # has to run past the halfway point of the grant, or a half-wall split would not
    # have truncated it and the count above proves nothing; and it has to stay inside
    # the grant, or it stopped on the deadline contract rather than on the cap.
    assert slow_clock > 0.5 * deadline, (
        f"premise stale: the slow arm stopped at clock {slow_clock:.1f}, inside half "
        f"of {deadline}; a wall split would not have cut it, so nothing is pinned"
    )
    assert slow_clock < deadline, (
        f"premise stale: the slow arm's clock reached {slow_clock:.1f} of {deadline}, "
        "so it stopped on the deadline contract and the cap is untested"
    )
