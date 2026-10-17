"""#1755: reformulation-added multipliers are not named by the large-bounds warning.

The budget (polyhedral) robust counterpart's ``ro_lam*`` duals and the bilevel
KKT multipliers ``bl_mu*`` have no a-priori upper bound; the warning told the
user to declare tighter bounds on columns they never declared.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.bilevel import BilevelProblem
from discopt.ro import RobustCounterpart, budget_uncertainty_set

pytestmark = pytest.mark.smoke

A_HAT = np.array([2.0, 3.0, 4.0])


def _large_bound_warnings(solve):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = solve()
    return r, [str(x.message) for x in w if "very large or infinite" in str(x.message)]


def _budget_model(ub):
    m = dm.Model("b")
    x = m.continuous("x", shape=3, lb=0, ub=ub)
    a = m.parameter("a", value=A_HAT)
    m.subject_to(a @ x >= 10)
    m.minimize(dm.sum(x))
    RobustCounterpart(m, budget_uncertainty_set(a, delta=np.full(3, 0.5), gamma=1.5)).formulate()
    return m


def test_budget_counterpart_duals_not_named():
    r, msgs = _large_bound_warnings(_budget_model(10).solve)
    assert r.status == "optimal"
    assert msgs == []


def test_user_declared_unbounded_still_named():
    _r, msgs = _large_bound_warnings(_budget_model(None).solve)
    assert len(msgs) == 1
    assert "x[0]" in msgs[0] and "ro_lam" not in msgs[0]


def test_bilevel_multipliers_not_named():
    m = dm.Model("sd")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
    )
    bl.formulate(method="strong_duality")
    _r, msgs = _large_bound_warnings(lambda: bl.solve(time_limit=30))
    assert msgs == []
