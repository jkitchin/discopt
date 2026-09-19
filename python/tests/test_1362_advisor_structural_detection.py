"""Issue #1362: the rule-based advisor reported bilinear terms in a pure MILP.

Found re-running ``docs/notebooks/llm_integration.ipynb``. Its facility-location
model is a **MILP with zero bilinear terms** (``classify_nonlinear_terms`` returns
an empty ``bilinear`` list, ``classify_problem`` says ``MILP``), and both advisory
modules said otherwise:

* ``advisor.suggest_solver_params`` -> ``partitions: 4`` with the reasoning
  "Enabling 4-partition McCormick (bilinear terms detected)";
* ``reformulation.analyze_reformulations`` -> "Found bilinear terms in 3
  constraints -- use partitions=4 or partitions=8".

Both decided it with ``" * " in str(constraint)``, which matches every
``constant * variable`` -- i.e. every linearly-scaled term in every model. The
suggestions are advisory, so this never touched solver math, but a user pasting
them into ``solve()`` pays for a piecewise-McCormick relaxation of a model with
nothing to partition.

The same function also recommended ``nlp_solver="ipm"``, describing it as a
"pure-JAX IPM" -- an engine that no longer exists. ``"ipm"`` is a back-compat
alias that resolves to POUNCE silently, so the advice was a no-op described
wrongly.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.llm.advisor import model_has_bilinear, suggest_solver_params
from discopt.llm.reformulation import analyze_reformulations


def _facility_location():
    """The llm_integration notebook's model: linear throughout."""
    m = dm.Model("facility_location")
    x = m.continuous("shipment", shape=(3, 5), lb=0, ub=200)
    y = m.binary("open", shape=(3,))
    supply = np.array([100, 150, 200])
    demand = np.array([80, 60, 70, 40, 50])
    cost = np.random.RandomState(42).rand(3, 5) * 100
    m.minimize(
        dm.sum(lambda i: dm.sum(lambda j: cost[i, j] * x[i, j], over=range(5)), over=range(3))
        + dm.sum(lambda i: 1000 * y[i], over=range(3))
    )
    for j in range(5):
        m.subject_to(dm.sum(lambda i: x[i, j], over=range(3)) >= demand[j], name=f"demand_{j}")
    for i in range(3):
        m.subject_to(dm.sum(lambda j: x[i, j], over=range(5)) <= supply[i] * y[i], name=f"cap_{i}")
    return m


def _bilinear_constraint_model():
    m = dm.Model("bilinear_con")
    a = m.continuous("a", lb=0, ub=4)
    b = m.continuous("b", lb=0, ub=4)
    z = m.binary("z")
    m.minimize(a + b + z)
    m.subject_to(a * b >= 2, name="prod")
    return m


def _bilinear_objective_model():
    m = dm.Model("bilinear_obj")
    a = m.continuous("a", lb=0, ub=4)
    b = m.continuous("b", lb=0, ub=4)
    m.minimize(a * b)
    m.subject_to(a + b >= 2, name="lin")
    return m


def test_the_precondition_the_model_really_is_linear():
    """Without this the rest of the file proves nothing."""
    from discopt._relax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(_facility_location())
    assert terms.bilinear == [], terms.bilinear


def test_a_scaled_linear_term_is_not_bilinear():
    assert model_has_bilinear(_facility_location()) is False


@pytest.mark.parametrize(
    "factory",
    [_bilinear_constraint_model, _bilinear_objective_model],
    ids=["constraint", "objective"],
)
def test_a_real_product_of_variables_is_still_detected(factory):
    assert model_has_bilinear(factory()) is True


def test_advisor_does_not_recommend_partitions_for_a_milp():
    params = suggest_solver_params(_facility_location())
    assert params["partitions"] == 0
    assert "bilinear" not in params["reasoning"].lower()


def test_advisor_still_recommends_partitions_for_a_bilinear_minlp():
    params = suggest_solver_params(_bilinear_constraint_model())
    assert params["partitions"] == 4


def test_advisor_recommends_the_engine_that_exists():
    """``"ipm"`` names a retired solver; the default is POUNCE."""
    for factory in (_facility_location, _bilinear_constraint_model):
        params = suggest_solver_params(factory())
        assert params["nlp_solver"] == "pounce", params
        assert "jax" not in params["reasoning"].lower(), params["reasoning"]


def test_reformulation_reports_no_bilinear_suggestion_for_a_milp():
    categories = [s.category for s in analyze_reformulations(_facility_location())]
    assert "mccormick_partitioning" not in categories, categories


@pytest.mark.parametrize(
    "factory",
    [_bilinear_constraint_model, _bilinear_objective_model],
    ids=["constraint", "objective"],
)
def test_reformulation_still_reports_a_real_bilinear_model(factory):
    suggestions = [
        s for s in analyze_reformulations(factory()) if s.category == "mccormick_partitioning"
    ]
    assert len(suggestions) == 1
    # It must never claim a count it did not find (the objective-only case used
    # to be reachable as "bilinear terms in 0 constraints").
    assert " 0 constraints" not in suggestions[0].description
