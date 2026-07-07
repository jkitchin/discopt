"""SP Phase 1: L-shaped setup (probability-weighted Benders).

The certified `L-shaped == extensive-form` solve needs the Rust+pounce backend and
runs in CI. Validated here without the solver: the L-shaped *decomposition
structure* is correct (via the Python-side `detect_decomposition` — first-stage
complicating vars, one recourse block per scenario, separable), the objective is
the probability-weighted extensive form, and the risk-neutral gate holds. Needs
JAX to evaluate the objective; no Rust.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from discopt._jax.dag_compiler import compile_expression_params  # noqa: E402
from discopt.modeling.core import Constant, Model  # noqa: E402
from discopt.stochastic import CVaR, ScenarioSet, solve_lshaped  # noqa: E402

pytestmark = pytest.mark.smoke


def _eval(body, model, values):
    flat = [float(values.get(v, 0.0)) for v in model._variables]
    params = tuple(jnp.asarray(p.value) for p in model._parameters)
    return float(compile_expression_params(body, model)(jnp.asarray(flat), params))


def _newsvendor_lshaped(scenarios, cost=2.0, price=5.0):
    m = Model("nv_lshaped")
    q = m.continuous("q", lb=0, ub=100)

    def recourse(model, data, s):
        d = float(data["demand"])
        sales = model.continuous(f"sales_{s}", lb=0, ub=100)
        model.subject_to(sales <= q)
        model.subject_to(sales <= Constant(d))
        return Constant(-price) * sales

    res = solve_lshaped(
        m,
        first_stage_vars=[q],
        scenarios=scenarios,
        recourse_builder=recourse,
        first_stage_cost=Constant(cost) * q,
        solve=False,  # backend not available in this sandbox; structure only
    )
    return m, q, res


def test_lshaped_decomposition_structure_is_per_scenario():
    scen = ScenarioSet.from_list(
        [(0.5, {"demand": 40}), (0.3, {"demand": 60}), (0.2, {"demand": 80})]
    )
    m, q, res = _newsvendor_lshaped(scen)
    st = res.structure
    # first stage is the complicating variable
    assert st.complicating_vars == ["q"]
    # each scenario's recourse is its own block, separable from the others
    recourse_blocks = [b for b in st.blocks if "q" not in b]
    assert len(recourse_blocks) == 3
    assert st.is_separable
    # every recourse variable landed in a recourse block
    recourse_vars = {v.name for v in m._variables if v.name.startswith("sales_")}
    assert recourse_vars == {v for b in recourse_blocks for v in b}


def test_lshaped_first_stage_annotation_set():
    scen = ScenarioSet.from_list([(0.5, {"demand": 40}), (0.5, {"demand": 80})])
    m, q, res = _newsvendor_lshaped(scen)
    assert m._decomp_stages.get("q") == 1


def test_lshaped_objective_is_probability_weighted_extensive_form():
    scen = ScenarioSet.from_list(
        [(0.5, {"demand": 40}), (0.3, {"demand": 60}), (0.2, {"demand": 80})]
    )
    m, q, res = _newsvendor_lshaped(scen)
    cost, price = 2.0, 5.0
    qv = 50.0
    sales = {0: 40.0, 1: 50.0, 2: 50.0}
    assign = {q: qv}
    for sv in [v for v in m._variables if v.name.startswith("sales_")]:
        assign[sv] = sales[int(sv.name.split("_")[1])]
    got = _eval(res.extensive_form.objective, m, assign)
    probs = scen.probabilities
    expected = cost * qv + sum(probs[s] * (-price * sales[s]) for s in range(3))
    assert abs(got - expected) < 1e-7


def test_lshaped_rejects_risk_averse_in_phase1():
    scen = ScenarioSet.from_list([(0.5, {"demand": 40}), (0.5, {"demand": 80})])
    m = Model("ra")
    q = m.continuous("q", lb=0, ub=100)
    with pytest.raises(NotImplementedError, match="risk-neutral"):
        solve_lshaped(
            m,
            first_stage_vars=[q],
            scenarios=scen,
            recourse_builder=lambda mm, d, s: Constant(0.0),
            first_stage_cost=Constant(2.0) * q,
            risk=CVaR(0.9),
            solve=False,
        )
