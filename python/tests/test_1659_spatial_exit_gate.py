"""#1659: the spatial B&B must not certify a point outside the declared rows.

On a small nonlinear GDP under ``gdp_method="hull"`` the spatial tree certified
``optimal`` at -0.386349, 5.4e-5 below the true optimum ``1 - 2 ln 2``, on a point
violating ``y >= exp(x) - 1`` by 5.9e-5. Three defects compounded:

* a raw relaxation point was injected as an incumbent under the search's 1e-4
  row tolerance, which it spent;
* the terminal polish found the true optimum but was judged on the LIFTED rows,
  whose ``1/dmin``-scaled aux definitions amplify 4e-9 of IPM noise in an
  inactive disjunct into a 1.3e-4 residual, so it was rejected;
* nothing checked the point that left against the declared 1e-6 tolerance.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import discopt.modeling as dm
from discopt.solver import _nonlinear_point_excess, _spatial_row_arbiter

TRUE_OPT = 1.0 - 2.0 * math.log(2.0)


def _build():
    m = dm.Model("nl_disj_1659")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=20)
    m.either_or([[y >= dm.exp(x) - 1, x <= 1], [y >= x**2 + 3, x >= 2]])
    m.minimize(y - 2 * x)
    return m, x, y


@pytest.mark.slow
def test_hull_gdp_returns_a_feasible_point_at_the_true_optimum():
    m, x, y = _build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(gdp_method="hull")
    assert r.objective is not None
    viol = math.exp(r.value(x)) - 1 - r.value(y)
    assert viol <= 1e-6, viol
    # The reported objective may not beat the true optimum by more than tolerance.
    assert r.objective >= TRUE_OPT - 1e-6 * (1 + abs(TRUE_OPT))
    if r.status == "optimal":
        assert r.bound <= r.objective + 1e-9
        assert r.bound <= TRUE_OPT + 1e-6


def test_row_arbiter_reads_the_prelift_rows():
    """With a lift, the arbiter is the pre-lift model over the original slice."""
    from discopt._relax.factorable_reform import factorable_reformulate
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.solver import _infer_constraint_bounds, _make_evaluator

    m, _, _ = _build()
    hull = reformulate_gdp(m, method="hull")
    nv = sum(v.size for v in hull._variables)
    lifted = factorable_reformulate(hull)
    assert sum(v.size for v in lifted._variables) > nv  # the lift fired
    ev = _make_evaluator(lifted)
    cl, cu = _infer_constraint_bounds(lifted, ev)

    arb = _spatial_row_arbiter(ev, cl, cu, hull, nv)
    assert arb.n_cols == nv
    assert arb.evaluator.n_constraints == len(hull._constraints)
    x_full = np.zeros(sum(v.size for v in lifted._variables))
    assert arb.view(x_full).shape == (nv,)

    same = _spatial_row_arbiter(ev, cl, cu, None, 0)
    assert same.evaluator is ev and same.n_cols is None


def test_exit_gate_measures_the_reported_violation():
    """The gate's own arithmetic flags a 5.9e-5 row violation (and passes 0)."""
    from discopt.solver import _infer_constraint_bounds, _make_evaluator

    m = dm.Model("gate")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=20)
    m.subject_to(y >= dm.exp(x) - 1)
    m.minimize(y - 2 * x)
    ev = _make_evaluator(m)
    cl, cu = _infer_constraint_bounds(m, ev)
    xv = 0.695197
    bad = np.array([xv, math.exp(xv) - 1 - 5.857e-5])
    good = np.array([xv, math.exp(xv) - 1])
    exc_bad, _, n_bad = _nonlinear_point_excess(ev, bad, cl, cu)
    exc_good, _, n_good = _nonlinear_point_excess(ev, good, cl, cu)
    assert n_bad > 0 and n_good > 0
    assert exc_bad > 1e-6
    assert exc_good <= 1e-6


def test_exit_gate_refuses_a_row_that_is_not_a_number():
    """A NaN row is an infinite violation, never "nothing checked" (#1659).

    ``max(-inf, nan)`` is ``-inf`` in Python, so the gate used to fold a NaN row
    away and pass the point -- which let the spatial exit certify ``y = 20`` on
    ``max y, (y == exp(x) + 2) or (y == log(x + 3))`` (true optimum e^2 + 2).
    """
    from discopt.solver import _infer_constraint_bounds, _make_evaluator

    m = dm.Model("nan_row")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(dm.log(x + 3) <= 5)
    m.minimize(x)
    ev = _make_evaluator(m)
    cl, cu = _infer_constraint_bounds(m, ev)
    with np.errstate(invalid="ignore"):
        exc, where, n = _nonlinear_point_excess(ev, np.array([-5.0]), cl, cu)
    assert n > 0
    assert exc == np.inf and "not finite" in where


@pytest.mark.slow
def test_hull_polish_outside_a_perspective_domain_is_not_adopted():
    """The #1043 two-branch model: the polish may not certify y at its bound."""
    from discopt.modeling.core import Constraint

    m = dm.Model("off_log_1659")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-10.0, ub=20.0)
    m.maximize(y)
    m.either_or(
        [
            [Constraint(body=y - (dm.exp(x) + 2.0), sense="==", rhs=0.0)],
            [Constraint(body=y - dm.log(x + 3.0), sense="==", rhs=0.0)],
        ],
        name="br",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(gdp_method="hull")
    assert r.objective == pytest.approx(math.exp(2.0) + 2.0, rel=1e-6)
