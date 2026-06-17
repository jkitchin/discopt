"""Engage the LP relaxer for general transcendental nonlinearity.

Models whose only nonlinearity is a general transcendental term (sqrt, exp,
sin/cos, ...) were routed away from the McCormick LP relaxer, so spatial B&B got
*no* dual bound and could not prove optimality (minlptests_nlp_003: 179 nodes,
status "feasible", root bound None). ``build_milp_relaxation`` already emits a
valid polyhedral outer approximation for these terms, so engaging the relaxer
yields a sound dual bound. These tests pin that the relaxer now engages and the
resulting bound is valid (never beyond the true optimum).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer


def test_relaxer_engages_on_general_transcendental():
    m = dm.Model("sqrt_only")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.minimize(-dm.sqrt(x + 0.1))
    assert MccormickLPRelaxer(m).has_relaxable_nonlinearity is True


@pytest.mark.slow
def test_transcendental_model_proves_optimality_with_valid_bound():
    # nlp_003-style: sqrt objective, exp + sin^2 constraints.
    m = dm.Model("nlp003")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(dm.sqrt(x + 0.1))
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    res = m.solve(time_limit=60)
    assert res.status == "optimal"  # was "feasible" (no bound) before
    # The dual bound is valid: for a maximization it never *under*-states the
    # optimum, and at proven optimality it matches the objective.
    assert res.bound is not None
    assert float(res.bound) >= float(res.objective) - 1e-4
    assert res.node_count <= 50  # was 179 with no bound to prune on


@pytest.mark.slow
def test_exp_objective_minimization_bound_is_valid():
    # min exp(x) over [-2, 2]: optimum exp(-2) ~ 0.1353; relaxer must not
    # over-state the bound (a valid lower bound is <= the optimum).
    m = dm.Model("exp_min")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(dm.exp(x))
    res = m.solve(time_limit=30)
    assert res.status == "optimal"
    assert abs(float(res.objective) - 0.13533528) < 1e-4
    if res.bound is not None:
        assert float(res.bound) <= float(res.objective) + 1e-4  # valid lower bound
