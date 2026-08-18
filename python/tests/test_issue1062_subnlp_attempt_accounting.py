"""Regression test for #1062: ``subnlp_calls`` must count solves, not successes.

The last piece of #1062. After the counters were wired into ``_solve_nlp_bb``
(they had been structurally 0 there), the number they carried was still not the
number the issue asks for: both GDP call sites did ::

    _subnlp_calls += len(_cfg_results)

and ``_cfg_results`` holds only the configurations that came back **feasible**.
So on a model where every configuration is infeasible — the case where the
heuristic works hardest and produces nothing — the solver reported
``subnlp_calls=0``, which is indistinguishable from the heuristic never having
run. That is the same §6 vacuous reading that named the issue, one layer in.

The fix threads an attempt counter out of the constructors themselves. The same
conflation existed at the ``enumerate_binary_seeds_subnlp`` call site and is
fixed with it — the class, not the instance (§2).

Everything here is asserted on small structural models; nothing depends on a
named instance or on any model being hard on the day.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt.solver import solve_model


def _all_configs_infeasible_model():
    """A recognised one-hot disjunction in which *no* configuration is feasible.

    ``demand`` wants at least 1.0 of total flow and ``cap`` allows at most 0.5,
    so every fixing of the disjunction hands the sub-NLP an infeasible problem.
    The one-hot row is a genuine structural match for ``_scan_one_hot_rows``
    (unit-coefficient binaries, ``== 1``), so the constructor really does run.
    """
    m = dm.Model("all_infeasible")
    y = m.binary("y", 3)
    x = m.continuous("x", 3, lb=0.0, ub=4.0)
    m.subject_to(y[0] + y[1] + y[2] == 1, name="pick_one")
    for i in range(3):
        m.subject_to(x[i] <= 4.0 * y[i], name=f"link{i}")
    m.subject_to(x[0] + x[1] + x[2] >= 1.0, name="demand")
    m.subject_to(x[0] + x[1] + x[2] <= 0.5, name="cap")
    m.minimize(sum((i + 1) * x[i] * x[i] for i in range(3)))
    return m


def _mixed_model():
    """The same disjunction with two of its three disjuncts made infeasible.

    ``cap1``/``cap2`` hold ``x[1]`` and ``x[2]`` below the 1.0 that ``demand``
    requires, so picking either of those disjuncts hands the sub-NLP an
    infeasible problem while disjunct 0 solves. That is what separates the two
    counters: the solver issues more sub-NLP solves than it gets points back.
    """
    m = dm.Model("mixed")
    y = m.binary("y", 3)
    x = m.continuous("x", 3, lb=0.0, ub=4.0)
    m.subject_to(y[0] + y[1] + y[2] == 1, name="pick_one")
    for i in range(3):
        m.subject_to(x[i] <= 4.0 * y[i], name=f"link{i}")
    m.subject_to(x[0] + x[1] + x[2] >= 1.0, name="demand")
    m.subject_to(x[1] <= 0.5, name="cap1")
    m.subject_to(x[2] <= 0.5, name="cap2")
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
def test_the_fixture_really_has_a_one_hot_group():
    """§6 guard: with no recognised disjunction every test below is vacuous."""
    model = _all_configs_infeasible_model()
    int_mask = ph._get_integer_mask(model)
    groups = ph._scan_one_hot_rows(model, int_mask, int(int_mask.size))
    assert groups, "no one-hot group detected — the accounting tests would prove nothing"


@pytest.mark.smoke
def test_config_constructor_reports_attempts_when_nothing_is_feasible():
    """The headline case: zero feasible points, but the solves did happen."""
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _all_configs_infeasible_model()
    evaluator, seed = _seed_for(model)

    stats: dict = {}
    results = ph.one_hot_config_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=None,
        stats=stats,
    )

    # The precondition that makes this test the one #1062 needs: the old
    # accounting (``len(results)``) is 0 here, so anything the new counter
    # reports above 0 is exactly the information that used to be lost.
    assert results == [], (
        "the fixture is meant to have no feasible configuration; it returned "
        f"{len(results)} — rebuild the fixture or this test proves nothing"
    )
    assert stats["attempted"] > 0, (
        "the constructor solved sub-NLPs and reported 0 attempts — this is the "
        "#1062 vacuous reading, one layer in"
    )


@pytest.mark.smoke
def test_the_dive_contributes_its_own_attempts():
    """The dive kept no counter at all; the wave's alone would under-report.

    Measured on the all-infeasible fixture the dive truthfully reports 0 — its
    per-level relaxations dead-end before a configuration is ever complete — so
    the fixture here is the mixed one, where the dive really does solve.
    """
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _mixed_model()
    evaluator, seed = _seed_for(model)

    dive_stats: dict = {}
    ph.one_hot_config_dive(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=None,
        stats=dive_stats,
    )
    assert dive_stats["attempted"] > 0, (
        "one_hot_config_dive issued sub-NLP solves but reported none"
    )

    # And the wave's total must include them, not just its own.
    whole: dict = {}
    ph.one_hot_config_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=None,
        stats=whole,
    )
    assert whole["attempted"] >= dive_stats["attempted"], (
        f"the constructor reported {whole['attempted']} attempts, fewer than the "
        f"{dive_stats['attempted']} its dive alone issues"
    )


@pytest.mark.smoke
def test_enumeration_reports_attempts_when_nothing_is_feasible():
    """Same conflation at the ``enumerate_binary_seeds_subnlp`` call site (§2)."""
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _all_configs_infeasible_model()
    evaluator, seed = _seed_for(model)

    stats: dict = {}
    results = ph.enumerate_binary_seeds_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        stats=stats,
    )
    assert results == [], "fixture regression: a configuration became feasible"
    assert stats["attempted"] > 0, "the enumeration solved sub-NLPs and reported none"


@pytest.mark.smoke
def test_stats_is_optional_and_the_default_path_is_unchanged():
    """Callers that pass no dict must behave exactly as before."""
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _all_configs_infeasible_model()
    evaluator, seed = _seed_for(model)
    assert (
        ph.one_hot_config_subnlp(
            model, seed, backend=get_nlp_solver("auto"), evaluator=evaluator, deadline=None
        )
        == []
    )
    assert (
        ph.enumerate_binary_seeds_subnlp(
            model, seed, backend=get_nlp_solver("auto"), evaluator=evaluator
        )
        == []
    )


@pytest.mark.smoke
def test_zero_attempts_is_reported_when_the_constructor_really_does_nothing():
    """The counter must still be able to say 0 — otherwise it is not a measurement.

    A deadline already in the past stops both searches before their first
    sub-NLP, so a truthful counter reads 0 here. Without this the tests above
    would pass against a counter hardwired to a positive number.
    """
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _all_configs_infeasible_model()
    evaluator, seed = _seed_for(model)

    stats: dict = {}
    ph.one_hot_config_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=-1.0,
        stats=stats,
    )
    assert stats["attempted"] == 0, (
        f"an expired deadline should stop before any sub-NLP; reported {stats['attempted']}"
    )


@pytest.mark.smoke
def test_solver_counts_attempts_not_successes():
    """End to end: ``subnlp_calls`` must exceed the number of feasible points.

    Under the old accounting the two were incremented from the same list, so
    ``subnlp_calls == subnlp_feasible`` held identically on every GDP solve. A
    strict inequality is only reachable once attempts are counted.
    """
    res = solve_model(_mixed_model(), nlp_bb=True, time_limit=30.0)

    assert res.nlp_bb is True, "test must exercise the NLP-BB path to be meaningful"
    assert res.subnlp_calls > 0, "the disjunct constructor ran but reported no sub-NLP work"
    assert res.subnlp_feasible <= res.subnlp_calls, (
        "more feasible points than sub-NLP solves — the counters are inconsistent"
    )
    assert res.subnlp_calls > res.subnlp_feasible, (
        f"subnlp_calls={res.subnlp_calls} == subnlp_feasible={res.subnlp_feasible}: "
        "the solver is still counting feasible points as sub-NLP calls (#1062)"
    )
