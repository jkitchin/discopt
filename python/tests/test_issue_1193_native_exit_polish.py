"""Regression tests for the #1193 native-exit primal polish.

Issue #1193: ``nvs19`` returns -1098.2 against a -1098.4 optimum at the default
budget and exits on ``max_nodes``. The measured cause is *primal*, in two parts:

1. **The Python heuristic layer never runs.** #1153 put ``node_limit`` on the
   native kernel's accepted-status list, which made an uncertified kernel exit
   *terminal*. Measured on nvs19: ``integer_box_search`` fired **0 times** on a
   default solve while ``_try_native_spatial_kernel`` fired once and returned a
   result. So no primal heuristic could ever see that incumbent.
2. **The heuristic that would fix it was priced for the wrong operation.**
   ``integer_box_search`` re-solves a continuous sub-NLP per cell, so its caps
   are ``max_int_vars=3`` / ``max_combos=128``. On a model with **no free
   continuous variables** that sub-NLP is zero-dimensional and reduces exactly to
   "round, verify, evaluate". Measured on nvs19 (8 integers, 0 continuous):
   18.922 ms/cell via ``subnlp`` vs 0.014 ms/cell direct — 1385x — so the whole
   3^8 = 6561-cell radius-1 grid costs 0.09 s, not 124.1 s.

nvs19's incumbent is a **strict 2-opt local optimum**: an exhaustive scan of its
L-inf <= 2 box found 98,429 feasible points and exactly ONE improving one, the
global optimum, three coordinates away. So no unit-move descent reaches it and
only a box enumeration does.

nvs19 is not in the in-repo corpus and a full solve is minutes long, so the class
is exercised here by the smallest model that exhibits it (CLAUDE.md §2: the fix
is for the class, not the instance) -- ``_two_opt_trap`` below is verified to BE
a strict 2-opt local optimum by :func:`test_trap_is_a_strict_two_opt_local_optimum`,
so these tests cannot quietly degrade into passing against a model that no longer
poses the problem.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pytest
from discopt._relax.nlp_evaluator import cached_evaluator
from discopt._relax.primal_heuristics import (
    _check_constraint_feasibility,
    _finalize_candidate,
    _get_integer_mask,
    integer_box_search,
    subnlp,
)
from discopt.modeling.core import Model
from discopt.solver import _native_exit_polish_enabled, _native_exit_primal_polish

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_INTEGER_BOX_POLISH"


def _two_opt_trap(*, with_continuous: bool = False, maximize: bool = False) -> Model:
    """5 integers, no continuous slots; the origin is a strict 2-opt local optimum.

    ``-(x0*x1*x2) + 0.1*sum(x)``: every 1-step move costs +0.1 and every 2-step
    move +0.2, while the 3-step ``(1,1,1,0,0)`` pays -0.7. The trilinear term is
    the general shape that makes the improving move invisible to unit descent --
    no proper subset of ``{x0,x1,x2}`` earns anything.
    """
    m = Model("two_opt_trap")
    xs = [m.integer(f"x{i}", lb=0, ub=4) for i in range(5)]
    if with_continuous:
        c = m.continuous("c", lb=0.0, ub=1.0)
        m.subject_to(c <= 1.0)
    m.subject_to(sum(xs) <= 3)
    body = -(xs[0] * xs[1] * xs[2]) + 0.1 * sum(xs)
    if maximize:
        m.maximize(-body)
    else:
        m.minimize(body)
    return m


ORIGIN = np.zeros(5)
ESCAPE = np.array([1.0, 1.0, 1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# The test instance really does pose the problem (CLAUDE.md §6)
# ---------------------------------------------------------------------------


def test_trap_is_a_strict_two_opt_local_optimum():
    """Guards every other test here: if this fails they are all vacuous."""
    m = _two_opt_trap()
    ev = cached_evaluator(m)
    base = float(ev.evaluate_objective(ORIGIN))
    assert base == pytest.approx(0.0)

    moves = 0
    for k in (1, 2):
        for combo in itertools.combinations(range(5), k):
            for vals in itertools.product((1.0, 2.0, 3.0, 4.0), repeat=k):
                x = ORIGIN.copy()
                for idx, v in zip(combo, vals):
                    x[idx] = v
                if not _check_constraint_feasibility(ev, x):
                    continue
                moves += 1
                assert float(ev.evaluate_objective(x)) > base - 1e-12, (
                    f"{combo}={vals} improves on the origin — not a 2-opt local optimum"
                )
    assert moves > 0, "no feasible 1-/2-opt neighbour was examined; the scan proved nothing"
    # ...and the 3-step escape really is strictly better and feasible.
    assert _check_constraint_feasibility(ev, ESCAPE)
    assert float(ev.evaluate_objective(ESCAPE)) == pytest.approx(-0.7)


# ---------------------------------------------------------------------------
# integer_box_search: the zero-free-continuous direct path
# ---------------------------------------------------------------------------


def test_direct_path_reaches_the_three_step_optimum_at_default_caps():
    """The #1193 fix. Before it, 5 integers > ``max_int_vars=3`` declined outright."""
    m = _two_opt_trap()
    out = integer_box_search(m, ORIGIN, radius=1, evaluator=cached_evaluator(m))
    assert out is not None, "declined — the direct path did not fire"
    x, obj = out
    np.testing.assert_allclose(x, ESCAPE)
    assert obj == pytest.approx(-0.7)


def test_the_flag_gates_the_direct_path_too(monkeypatch):
    """ONE flag governs BOTH halves of the #1193 fix.

    Regression test for a real defect in the first cut: only the exit polish was
    gated, so the re-priced box search stayed live in a graduation panel's OFF
    arm and the A/B measured neither half honestly. With the flag off, this
    5-integer zero-continuous model must fall back to the pre-#1193 caps
    (``max_int_vars=3``) and decline.
    """
    m = _two_opt_trap()
    ev = cached_evaluator(m)
    monkeypatch.setenv(FLAG, "0")
    assert integer_box_search(m, ORIGIN, radius=1, evaluator=ev) is None
    monkeypatch.setenv(FLAG, "1")
    out = integer_box_search(m, ORIGIN, radius=1, evaluator=ev)
    assert out is not None
    np.testing.assert_allclose(out[0], ESCAPE)


def test_direct_path_returns_only_feasible_integral_points():
    m = _two_opt_trap()
    ev = cached_evaluator(m)
    out = integer_box_search(m, ORIGIN, radius=1, evaluator=ev)
    assert out is not None
    x = out[0]
    int_mask = _get_integer_mask(m)
    np.testing.assert_allclose(x[int_mask], np.round(x[int_mask]))
    assert _check_constraint_feasibility(ev, x)
    assert float(ev.evaluate_objective(x)) == pytest.approx(out[1])


def test_a_free_continuous_variable_keeps_the_sub_nlp_caps():
    """The direct path is gated on the *cost* of a cell, not on the model's name.

    One free continuous slot restores the per-cell NLP solve, so the NLP-priced
    caps apply again and 5 integers is over ``max_int_vars=3``.
    """
    m = _two_opt_trap(with_continuous=True)
    out = integer_box_search(m, np.zeros(6), radius=1, evaluator=cached_evaluator(m))
    assert out is None


def test_direct_path_still_honours_its_own_cell_cap():
    """The cap is re-priced, not removed: a grid past it is declined, not run."""
    m = _two_opt_trap()
    out = integer_box_search(m, ORIGIN, radius=1, evaluator=cached_evaluator(m), max_combos=10)
    assert out is None, "3^5 = 243 cells must not run under a 10-cell cap"


def test_an_explicit_cap_is_never_silently_overridden(monkeypatch):
    """An explicitly passed cap binds on the DIRECT path too.

    Regression test for a real defect in the first cut: the direct path read a
    separate ``max_combos_direct`` and ignored ``max_combos``/``max_int_vars``
    entirely, so a caller who asked for a 2-cell search silently got a 20,000-cell
    one. Two pre-existing tests in ``test_gdpopt_heuristics_core_units.py`` caught
    it in CI. Only the DEFAULTS are re-priced by path; an explicit number is
    obeyed verbatim on either.
    """
    m = _two_opt_trap()
    ev = cached_evaluator(m)
    # Defaults: the direct path runs and escapes.
    assert integer_box_search(m, ORIGIN, radius=1, evaluator=ev) is not None
    # Explicit caps: each one alone must veto it.
    assert integer_box_search(m, ORIGIN, radius=1, evaluator=ev, max_combos=2) is None
    assert integer_box_search(m, ORIGIN, radius=1, evaluator=ev, max_int_vars=1) is None


def test_direct_path_is_bounded_by_its_eval_budget():
    """The extent is a deterministic operation count, not a wall clock (#912)."""
    m = _two_opt_trap()
    ev = cached_evaluator(m)
    # Two EVAL charges per cell, so a 4-charge budget buys 2 cells of a 243-cell
    # grid — far too few to reach a 3-step escape.
    starved = integer_box_search(m, ORIGIN, radius=1, evaluator=ev, eval_budget=4)
    assert starved is None or starved[1] > -0.7 + 1e-9
    full = integer_box_search(m, ORIGIN, radius=1, evaluator=ev, eval_budget=100_000)
    assert full is not None and full[1] == pytest.approx(-0.7)


def test_zero_time_budget_still_means_no_budget_at_all():
    m = _two_opt_trap()
    out = integer_box_search(m, ORIGIN, radius=1, evaluator=cached_evaluator(m), time_budget=0.0)
    assert out is None


def test_finalize_candidate_matches_subnlp_when_there_are_no_continuous_slots():
    """The substitution the direct path rests on, checked directly.

    With every integer fixed and no continuous variable left, ``subnlp``'s inner
    solve has zero degrees of freedom, so it must agree with the direct
    "round, verify, evaluate" on both the point and the objective.
    """
    m = _two_opt_trap()
    ev = cached_evaluator(m)
    int_mask = _get_integer_mask(m)
    compared = 0
    for combo in itertools.product((0.0, 1.0), repeat=3):
        seed = np.array([*combo, 0.0, 0.0])
        direct = _finalize_candidate(ev, seed, int_mask, 1e-5, 1e-6)
        viasub = subnlp(m, seed, evaluator=ev, integer_tol=1e-5, feas_tol=1e-6)
        assert (direct is None) == (viasub is None), f"{combo}: disagreement on acceptance"
        if direct is None:
            continue
        compared += 1
        np.testing.assert_allclose(direct[0], viasub[0], atol=1e-9)
        assert direct[1] == pytest.approx(viasub[1], abs=1e-9)
    assert compared > 0, "no cell was accepted by either path; the comparison proved nothing"


# ---------------------------------------------------------------------------
# _native_exit_primal_polish
# ---------------------------------------------------------------------------


@pytest.fixture
def polish_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


def test_polish_adopts_a_strictly_improving_verified_point(polish_on):
    m = _two_opt_trap()
    out = _native_exit_primal_polish(m, ORIGIN, 0.0, -10.0, 5, None)
    assert out is not None
    np.testing.assert_allclose(out[0][:5], ESCAPE)
    assert out[1] == pytest.approx(-0.7)


def test_polish_declines_when_nothing_improves(polish_on):
    """An incumbent already better than anything in the box is left alone."""
    m = _two_opt_trap()
    assert _native_exit_primal_polish(m, ORIGIN, -1e6, -1e9, 5, None) is None


def test_polish_declines_a_candidate_that_crosses_the_reported_bound(polish_on, caplog):
    """A verified point beyond a *valid* bound means the BOUND is wrong.

    The polish must surface that loudly and decline, leaving the run identical to
    what it would have reported anyway — never silently ship a negative gap, and
    never weaken the check to let the improvement through (CLAUDE.md §1/§3).
    """
    m = _two_opt_trap()
    with caplog.at_level(logging.ERROR, logger="discopt.solver"):
        out = _native_exit_primal_polish(m, ORIGIN, 0.0, -0.1, 5, None)
    assert out is None
    assert any("bound is invalid" in r.getMessage() for r in caplog.records)


def test_polish_respects_maximize_sense(polish_on):
    """Improvement is 'better in the model's own sense', not 'numerically smaller'."""
    m = _two_opt_trap(maximize=True)
    out = _native_exit_primal_polish(m, ORIGIN, 0.0, 10.0, 5, None)
    assert out is not None
    np.testing.assert_allclose(out[0][:5], ESCAPE)
    assert out[1] == pytest.approx(0.7)
    # And the same point must NOT be adopted against an already-better incumbent.
    assert _native_exit_primal_polish(m, ORIGIN, 1e6, 1e9, 5, None) is None


def test_polish_is_off_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    assert not _native_exit_polish_enabled()
    m = _two_opt_trap()
    assert _native_exit_primal_polish(m, ORIGIN, 0.0, -10.0, 5, None) is None


def test_polish_defaults_on(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert _native_exit_polish_enabled()


# ---------------------------------------------------------------------------
# End-to-end: the polish is bound-, node- and certificate-neutral
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_nodes", [1, 50, 5000])
def test_solve_is_bound_and_node_neutral_across_the_flag(max_nodes, monkeypatch):
    """CLAUDE.md §5 bound-neutral regime: the polish runs AFTER the tree, so it
    cannot move the dual bound or the node count. Any drift is a plumbing bug."""
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        r = _two_opt_trap().solve(max_nodes=max_nodes, time_limit=30.0)
        results[flag] = r
    off, on = results["0"], results["1"]
    assert off.node_count == on.node_count, "node count moved"
    if off.bound is not None and on.bound is not None:
        assert off.bound == pytest.approx(on.bound, rel=1e-12, abs=1e-12), "dual bound moved"
    assert off.gap_certified == on.gap_certified, "certification changed"
    # A certified exit must not notice the flag at all.
    if off.status == "optimal":
        assert on.status == "optimal"
        assert off.objective == pytest.approx(on.objective, rel=1e-12, abs=1e-12)
    # The incumbent may only ever get BETTER (minimize sense).
    if off.objective is not None and on.objective is not None:
        assert on.objective <= off.objective + 1e-9
