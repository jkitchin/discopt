"""Regression test for #1062: ``subnlp_incumbent_updates`` must count adoptions.

The last vacuous reading in #1062's own instrument. After ``subnlp_calls`` was
fixed to count attempts rather than successes, the third counter still read a
number nobody could act on: every one of the seven injection sites did ::

    _inject_incumbent(x, obj)      # returns bool -- discarded
    _subnlp_incumbent_updates += 1

``_inject_incumbent`` returns ``tree.inject_incumbent(...)``, and the Rust
``TreeManager::inject_incumbent`` (crates/discopt-core/src/bnb/tree_manager.rs)
updates the incumbent *only if ``obj_val`` is strictly better* and returns
``true`` in exactly that case. Throwing the bool away turned the counter into a
second copy of the attempt count.

Measured before the fix (60 s, published optima from ``minlplib.solu``):

    syn40m      112 calls   78 feasible   78 "updates"   obj 33.19704202508967
    syn20m02m   112 calls   65 feasible   65 "updates"   obj 636.7196818434977

A 100% adoption rate is arithmetically impossible alongside an objective that
does not move: after the first adoption the incumbent can only be replaced by a
strictly better point, so 78 adoptions means 78 distinct improving objectives.
The counter said the heuristic was working on every single point while it was in
fact contributing one point and then repeating itself -- the same §6 failure
(an instrument that cannot distinguish "working" from "no-op") that named the
issue, now two layers in.

The same edit is applied at all seven sites, not just the two the measurement
happened to reach (§2). The log lines move under the same guard: "SubNLP
incumbent: obj=..." asserted an incumbent for points the tree had declined.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt.solver import solve_model


def _three_feasible_disjuncts():
    """A one-hot disjunction whose three disjuncts are all feasible and ranked.

    Fixing ``y[i] = 1`` forces ``x[j] = 0`` for ``j != i`` and ``demand`` then
    pins ``x[i] = 1``, so disjunct ``i`` has objective exactly ``i + 1``: three
    feasible points, well separated, with a known order. That separation is what
    lets the assertions below be exact rather than tolerance-dependent.
    """
    m = dm.Model("ranked_disjuncts")
    y = m.binary("y", 3)
    x = m.continuous("x", 3, lb=0.0, ub=4.0)
    m.subject_to(y[0] + y[1] + y[2] == 1, name="pick_one")
    for i in range(3):
        m.subject_to(x[i] <= 4.0 * y[i], name=f"link{i}")
    m.subject_to(x[0] + x[1] + x[2] >= 1.0, name="demand")
    m.minimize(sum((i + 1) * x[i] * x[i] for i in range(3)))
    return m


def _real_config_points():
    """Run the genuine constructor once; return one real point per disjunct.

    Replaying *real* points keeps the stub below from being a fiction: the
    solver still re-evaluates each one, checks its rows and its integrality, and
    injects it exactly as it would on an unstubbed solve. Only the order in
    which they arrive is controlled.

    The constructor's wave and dive between them return the same configuration
    many times over (66 points for 3 disjuncts on this fixture), so the list is
    deduplicated by objective. Duplicates would be *correctly* declined by the
    tree and would make the improving arm below untestable.
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solvers.nlp_backend import get_nlp_solver

    model = _three_feasible_disjuncts()
    evaluator = NLPEvaluator(model)
    lb, ub = evaluator.variable_bounds
    seed = 0.5 * (np.asarray(lb, dtype=np.float64) + np.asarray(ub, dtype=np.float64))

    stats: dict = {}
    found = ph.one_hot_config_subnlp(
        model,
        seed,
        backend=get_nlp_solver("auto"),
        evaluator=evaluator,
        deadline=None,
        stats=stats,
    )
    objs = [float(evaluator.evaluate_objective(np.asarray(x, dtype=np.float64))) for x, _ in found]

    uniq_points, uniq_objs, seen = [], [], set()
    for pair, obj in sorted(zip(found, objs), key=lambda t: t[1]):
        key = round(obj, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq_points.append(pair)
        uniq_objs.append(obj)
    return uniq_points, uniq_objs


@pytest.mark.smoke
def test_the_fixture_really_yields_three_separated_feasible_configurations():
    """§6 guard: without three ranked points every assertion below is vacuous."""
    found, objs = _real_config_points()
    assert len(found) >= 3, (
        f"the constructor returned {len(found)} distinct feasible objective(s); "
        "the adoption counter cannot be distinguished from the attempt counter "
        "below 3"
    )
    ranked = sorted(objs)
    assert ranked[0] == pytest.approx(1.0, rel=1e-4)
    assert min(b - a for a, b in zip(ranked, ranked[1:])) > 0.5, (
        f"objectives {ranked} are not separated; the ordering below would be "
        "decided by floating-point noise rather than by the fixture"
    )


def _solve_with_replayed_points(monkeypatch, order):
    """Solve with the constructor stubbed to replay real points in a fixed order."""
    found, objs = _real_config_points()
    ordered = [pair for _, pair in sorted(zip(objs, found), key=lambda t: t[0])]
    if order == "worst_first":
        ordered = list(reversed(ordered))

    def _stub(*args, **kwargs):
        stats = kwargs.get("stats")
        if stats is not None:
            stats["attempted"] = len(ordered)
        return list(ordered)

    monkeypatch.setattr(ph, "one_hot_config_subnlp", _stub)
    res = solve_model(_three_feasible_disjuncts(), nlp_bb=True, time_limit=30.0)
    assert res.nlp_bb is True, "test must exercise the NLP-BB path to be meaningful"
    assert res.subnlp_feasible >= 3, (
        f"the stubbed points did not reach the injection loop (feasible="
        f"{res.subnlp_feasible}); this run measures nothing"
    )
    return res, len(ordered)


@pytest.mark.smoke
def test_declined_injections_are_not_counted_as_incumbent_updates(monkeypatch):
    """Best point first: the two that follow are declined and must not be counted.

    Every point here is feasible, integral and injected, so the attempt count is
    3 in both the old code and the new. Only the second and third are worse than
    the standing incumbent, so the tree declines them -- and before the fix the
    counter recorded them anyway.
    """
    res, n = _solve_with_replayed_points(monkeypatch, order="best_first")

    assert res.subnlp_incumbent_updates == 1, (
        f"{res.subnlp_incumbent_updates} adoption(s) reported from {n} injections "
        "of monotonically worsening points -- only the first can improve the "
        "incumbent, so the counter is still counting attempts (#1062)"
    )
    assert res.subnlp_incumbent_updates < res.subnlp_feasible


@pytest.mark.smoke
def test_the_counter_still_rises_when_the_points_genuinely_improve(monkeypatch):
    """The anti-vacuity arm: the fix must not hardwire the counter to 1.

    Same three points, worst first, so each one really does replace the
    incumbent. A counter that reported 1 here would be as useless as the one it
    replaced, in the opposite direction.
    """
    res, n = _solve_with_replayed_points(monkeypatch, order="worst_first")

    assert res.subnlp_incumbent_updates == n, (
        f"{res.subnlp_incumbent_updates} adoption(s) reported from {n} strictly "
        "improving injections -- the counter is under-reporting real work"
    )


@pytest.mark.smoke
def test_no_call_site_discards_the_injection_result():
    """The fix is the class, not the two sites the measurement happened to reach.

    A source assertion rather than a behavioural one: the remaining sites live on
    the spatial path behind heuristics that no small fixture reaches reliably, and
    an unguarded increment there is the identical defect. Reintroducing one is a
    single line, so this makes that line fail here rather than in a benchmark
    report months later.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "discopt" / "solver.py"
    lines = src.read_text().splitlines()
    sites = [i for i, ln in enumerate(lines) if "_subnlp_incumbent_updates += 1" in ln]
    assert len(sites) == 7, (
        f"expected the 7 known injection sites, found {len(sites)}; a new one "
        "must be guarded on the injection result too"
    )

    guard = re.compile(r"^\s*(if .*inject_incumbent\(|if _adopted:)")
    for i in sites:
        # The increment must sit directly under a guard on the injection's result.
        assert guard.match(lines[i - 1]), (
            f"{src.name}:{i + 1} increments the adoption counter unguarded; "
            f"the line above is {lines[i - 1].strip()!r}. inject_incumbent returns "
            "False when the tree declines the point (#1062)."
        )
