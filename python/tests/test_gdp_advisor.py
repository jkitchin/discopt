"""Tests for the F1 GDP big-M-vs-hull advisor (issue #53)."""

from __future__ import annotations

import discopt.modeling as dm
from discopt._jax.gdp_advisor import (
    GdpAdvice,
    recommend_method,
    recommend_methods,
)
from discopt._jax.gdp_reformulate import reformulate_gdp


def test_advisor_picks_hull_for_small_linear_disjunction():
    """Two-disjunct linear modes on a single variable: hull wins on
    tightness with a tiny disaggregation cost."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.minimize(x)
    m.either_or([[x <= 5.0], [x >= 15.0]], name="modes")

    advice = recommend_methods(m)
    assert len(advice) == 1
    a = advice[0]
    assert isinstance(a, GdpAdvice)
    assert a.recommendation == "hull"
    assert a.has_nonlinear is False
    assert a.n_disjuncts == 2
    assert a.n_common_vars == 1
    assert a.hull_aux_cost == 2
    assert a.name == "modes"


def test_advisor_picks_mbigm_for_large_linear_disjunction():
    """Many disjuncts with many variables: hull cost exceeds the
    budget, so the advisor falls back to mbigm."""
    m = dm.Model("big-linear")
    n_vars = 8
    n_disj = 8  # 8 * 8 = 64 > default budget of 50
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_vars)]
    m.minimize(xs[0])
    disjuncts = []
    for k in range(n_disj):
        # Each disjunct is a single linear constraint touching every var.
        body = xs[0]
        for j in range(1, n_vars):
            body = body + xs[j]
        disjuncts.append([body <= float(k + 1)])
    m.either_or(disjuncts, name="big")

    advice = recommend_methods(m)
    assert len(advice) == 1
    a = advice[0]
    assert a.recommendation == "mbigm"
    assert a.hull_aux_cost == n_vars * n_disj
    assert a.has_nonlinear is False


def test_advisor_picks_big_m_for_nonlinear():
    """A disjunct with a nonlinear body: advisor picks big-m or mbigm,
    never hull."""
    m = dm.Model("nl")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.either_or([[x * x <= 4.0], [x >= 5.0]], name="nl_modes")

    advice = recommend_methods(m)
    assert len(advice) == 1
    a = advice[0]
    assert a.recommendation in {"big-m", "mbigm"}
    assert a.has_nonlinear is True


def test_advisor_picks_mbigm_when_nonlinear_M_is_loose():
    """Nonlinear body with a loose interval-arithmetic M: prefer mbigm
    so LP-tightening can recoup some of the relaxation gap."""
    m = dm.Model("nl-loose")
    x = m.continuous("x", lb=-200.0, ub=200.0)  # x*x ∈ [0, 40000]
    m.minimize(x)
    m.either_or([[x * x <= 100.0], [x >= 5.0]], name="nl_loose")

    advice = recommend_methods(m)
    assert len(advice) == 1
    a = advice[0]
    assert a.recommendation == "mbigm"
    assert a.has_nonlinear is True
    assert a.max_big_m > 100.0


def test_advisor_returns_empty_for_model_without_disjunctions():
    m = dm.Model("plain")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    m.subject_to(x <= 0.5)
    assert recommend_methods(m) == []


def test_recommend_method_respects_custom_thresholds():
    """The hull-aux-budget knob can flip a small disjunction off hull."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.minimize(x)
    m.either_or([[x <= 5.0], [x >= 15.0]], name="modes")

    # With budget=1, hull's cost (2) exceeds budget — should pick mbigm.
    a = recommend_method(m._constraints[-1], m, hull_aux_budget=1)
    assert a.recommendation == "mbigm"


def test_reformulate_gdp_auto_dispatches_per_disjunction():
    """In auto mode, a small linear disjunction goes to hull and a
    nonlinear one goes to big-m within the same model."""
    m = dm.Model("mixed")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    m.either_or([[x <= 3.0], [x >= 7.0]], name="lin")
    m.either_or([[y * y <= 4.0], [y >= 5.0]], name="nl")

    new_m = reformulate_gdp(m, method="auto")

    # Hull dispatch on the linear branch creates disaggregated copies
    # whose names start with "_hull_". Big-M dispatch on the nonlinear
    # branch creates selector binaries named "_gdp_aux_disj_*".
    new_var_names = {v.name for v in new_m._variables}
    has_hull = any(n.startswith("_hull_lin_") for n in new_var_names)
    has_bigm = any(n.startswith("_gdp_aux_disj_nl") for n in new_var_names)
    assert has_hull, f"expected hull aux for the linear disjunction; got {new_var_names}"
    assert has_bigm, f"expected big-m selector for the nonlinear disjunction; got {new_var_names}"


def test_reformulate_gdp_auto_matches_hull_when_advisor_picks_hull():
    """Sanity: for a disjunction the advisor flags as hull, ``method='auto'``
    yields the same constraint count as ``method='hull'``."""
    m_auto = dm.Model("a")
    x_a = m_auto.continuous("x", lb=0.0, ub=20.0)
    m_auto.minimize(x_a)
    m_auto.either_or([[x_a <= 5.0], [x_a >= 15.0]], name="modes")

    m_hull = dm.Model("h")
    x_h = m_hull.continuous("x", lb=0.0, ub=20.0)
    m_hull.minimize(x_h)
    m_hull.either_or([[x_h <= 5.0], [x_h >= 15.0]], name="modes")

    auto_out = reformulate_gdp(m_auto, method="auto")
    hull_out = reformulate_gdp(m_hull, method="hull")

    assert len(auto_out._constraints) == len(hull_out._constraints)
    assert len(auto_out._variables) == len(hull_out._variables)
