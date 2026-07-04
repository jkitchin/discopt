"""SP Phase 0: scenarios, risk measures, and the extensive form.

Validated without the global solver: (a) the extensive-form objective actually
carries the scenario probabilities (Σ_s p_s Q_s), checked by compiling and
evaluating it; (b) the CVaR Rockafellar–Uryasev expression equals the analytic
CVaR (brute-force η-grid oracle); (c) scenario-set validation and SAA; (d) the
chance-constraint coverage encoding. Needs JAX (to evaluate expressions); no Rust.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from discopt._jax.dag_compiler import compile_expression_params  # noqa: E402
from discopt.modeling.core import Constant, Model  # noqa: E402
from discopt.stochastic import (  # noqa: E402
    CVaR,
    Expectation,
    ScenarioSet,
    build_extensive_form,
    chance_constraint,
)

pytestmark = pytest.mark.smoke


def _eval(body, model, values):
    flat = [float(values.get(v, 0.0)) for v in model._variables]
    params = tuple(jnp.asarray(p.value) for p in model._parameters)
    return float(compile_expression_params(body, model)(jnp.asarray(flat), params))


# ---------------------------------------------------------------------------
# 1. ScenarioSet.
# ---------------------------------------------------------------------------


def test_scenarioset_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ScenarioSet.from_list([(0.3, {"d": 1}), (0.3, {"d": 2})])


def test_scenarioset_from_samples_uniform_and_from_list():
    s = ScenarioSet.from_samples({"d": np.array([10.0, 20.0, 30.0])})
    assert len(s) == 3
    assert np.allclose(s.probabilities, 1.0 / 3)
    assert s[1].data["d"] == 20.0
    explicit = ScenarioSet.from_list([(0.25, {"d": 5}), (0.75, {"d": 9})])
    assert np.allclose(explicit.probabilities, [0.25, 0.75])


def test_scenarioset_sample_is_reproducible():
    samplers = {"d": lambda rng: float(rng.normal(100, 10))}
    a = ScenarioSet.sample(samplers, n=50, seed=7)
    b = ScenarioSet.sample(samplers, n=50, seed=7)
    assert a.seed == 7 and a.n_sampled == 50
    assert [s.data["d"] for s in a] == [s.data["d"] for s in b]


# ---------------------------------------------------------------------------
# 2. Extensive form carries the scenario probabilities (the key primitive).
# ---------------------------------------------------------------------------


def _newsvendor(scenarios, cost=2.0, price=5.0):
    m = Model("newsvendor")
    q = m.continuous("q", lb=0, ub=100)  # first-stage order

    def recourse(model, data, s):
        d = float(data["demand"])
        sales = model.continuous(f"sales_{s}", lb=0, ub=100)
        model.subject_to(sales <= q)
        model.subject_to(sales <= Constant(d))
        return Constant(-price) * sales  # recourse cost = -revenue

    ef = build_extensive_form(
        m,
        scenarios=scenarios,
        recourse_builder=recourse,
        first_stage_cost=Constant(cost) * q,
        risk=Expectation(),
    )
    return m, q, ef


def test_extensive_form_objective_is_probability_weighted():
    scen = ScenarioSet.from_list(
        [(0.5, {"demand": 40}), (0.3, {"demand": 60}), (0.2, {"demand": 80})]
    )
    m, q, ef = _newsvendor(scen)
    # pick a feasible assignment and check obj == c*q + Σ p_s (-price * sales_s)
    cost, price = 2.0, 5.0
    qv = 50.0
    sales = {0: 40.0, 1: 50.0, 2: 50.0}  # <= q and <= demand
    assign = {q: qv}
    sales_vars = [v for v in m._variables if v.name.startswith("sales_")]
    for s, sv in enumerate(sales_vars):
        assign[sv] = sales[s]
    got = _eval(ef.objective, m, assign)
    probs = scen.probabilities
    expected = cost * qv + sum(probs[s] * (-price * sales[s]) for s in range(3))
    assert abs(got - expected) < 1e-7


def test_expectation_reduces_to_first_stage_when_no_recourse_cost():
    scen = ScenarioSet.from_list([(0.6, {"demand": 10}), (0.4, {"demand": 20})])
    m = Model("t")
    q = m.continuous("q", lb=0, ub=10)
    ef = build_extensive_form(
        m,
        scenarios=scen,
        recourse_builder=lambda mm, d, s: Constant(0.0),
        first_stage_cost=Constant(3.0) * q,
    )
    assert abs(_eval(ef.objective, m, {q: 4.0}) - 12.0) < 1e-9


# ---------------------------------------------------------------------------
# 3. CVaR Rockafellar–Uryasev expression == analytic CVaR (brute-force oracle).
# ---------------------------------------------------------------------------


def _analytic_cvar(costs, probs, alpha):
    """min_η η + 1/(1-α) Σ p_s (c_s-η)^+ via a fine η grid, returning (value, η*)."""
    inv = 1.0 / (1.0 - alpha)
    grid = np.linspace(min(costs) - 1, max(costs) + 1, 20001)
    vals = grid + inv * np.array([np.sum(probs * np.maximum(costs - e, 0.0)) for e in grid])
    k = int(np.argmin(vals))
    return float(vals[k]), float(grid[k])


@pytest.mark.parametrize("alpha", [0.5, 0.8, 0.95])
def test_cvar_expression_matches_analytic(alpha):
    costs = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cvar_val, eta_star = _analytic_cvar(costs, probs, alpha)

    m = Model("cvar")
    risk = CVaR(alpha)
    obj = risk.build(m, Constant(0.0), [Constant(float(c)) for c in costs], probs)

    assign = {}
    for v in m._variables:
        if v.name == "cvar_eta":
            assign[v] = eta_star
        elif v.name.startswith("cvar_u"):
            s = int(v.name[len("cvar_u") :])
            assign[v] = max(costs[s] - eta_star, 0.0)
    got = _eval(obj, m, assign)
    assert abs(got - cvar_val) < 1e-3
    # risk-averse: CVaR of the tail >= the mean
    assert cvar_val >= float(np.dot(probs, costs)) - 1e-9


# ---------------------------------------------------------------------------
# 4. Chance constraint (SAA) coverage encoding.
# ---------------------------------------------------------------------------


def test_chance_constraint_structure_and_coverage():
    m = Model("cc")
    x = m.continuous("x", lb=0, ub=100)
    demands = [30.0, 50.0, 70.0, 90.0]
    probs = [0.25, 0.25, 0.25, 0.25]
    bodies = [x - Constant(d) for d in demands]  # want x <= d in each scenario
    zs = chance_constraint(m, bodies, probs, epsilon=0.25, big_m=1000.0)
    assert len(zs) == 4
    # coverage constraint present
    cov = [c for c in m._constraints if (c.name or "").endswith("coverage")]
    assert len(cov) == 1
    # at x=60, scenarios with d<60 (30,50) are violated -> need z=1 there:
    # Σ p_s z_s = 0.5 > ε=0.25, so coverage is violated (as it should be at x=60).
    z_on = {zs[i]: 1.0 for i in range(4) if demands[i] < 60.0}
    assign = {x: 60.0, **z_on}
    # coverage body is (Σ p_s z_s) - ε <= 0 ; here 0.5 - 0.25 = 0.25 > 0 -> infeasible
    assert _eval(cov[0].body, m, assign) > 0.0
    # at x=45, only d=30 violated -> Σ p_s z_s = 0.25 <= ε -> feasible
    z_on2 = {zs[i]: 1.0 for i in range(4) if demands[i] < 45.0}
    assert _eval(cov[0].body, m, {x: 45.0, **z_on2}) <= 1e-12
