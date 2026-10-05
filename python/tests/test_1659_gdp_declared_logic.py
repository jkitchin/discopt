"""#1659: a nonlinear GDP hull certified a point violating its disjunct row.

``min y - 2x`` over ``either_or([[y >= exp(x) - 1, x <= 1], [y >= x**2 + 3, x >= 2]])``
has optimum ``1 - 2 ln 2`` at ``(ln 2, 1)``. With ``gdp_method="hull"`` the solve
reported ``optimal`` at -0.386348729, a point violating ``y >= exp(x) - 1`` by
5.9e-5. Two defects let it through:

* the terminal polish found the true optimum but judged it on the factorable
  lift's rows, where clearing the hull's ``v / (lam + 1e-8)`` scales a row by
  1e8 and turns the inactive disjunct's 4e-9 residue into a 1.3e-4 "violation";
* ``verify_point`` judged only algebraic rows, so a model declared with
  ``either_or`` had zero rows to check and the certificate backstop passed it.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import discopt.solver as solver
import numpy as np
import pytest
from discopt.modeling.core import _DisjunctiveConstraint
from discopt.validation.feasibility import verify_point

TRUE_OPT = 1.0 - 2.0 * math.log(2.0)


def _witness():
    m = dm.Model("nl_disj")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=20)
    m.either_or([[y >= dm.exp(x) - 1, x <= 1], [y >= x**2 + 3, x >= 2]])
    m.minimize(y - 2 * x)
    return m, x, y


def _row_violation(r, x, y):
    xv, yv = float(r.value(x)), float(r.value(y))
    if xv <= 1 + 1e-6:
        return max(0.0, math.exp(xv) - 1 - yv, xv - 1)
    return max(0.0, xv**2 + 3 - yv, 2 - xv)


@pytest.mark.parametrize("method", ["hull", "big-m"])
def test_reported_point_meets_its_disjunct(method):
    m, x, y = _witness()
    r = m.solve(gdp_method=method, time_limit=60)
    assert r.x is not None
    assert _row_violation(r, x, y) <= 1e-6
    assert r.objective >= TRUE_OPT - 1e-6
    assert r.bound <= TRUE_OPT + 1e-6


def test_backstop_decertifies_and_repairs_an_infeasible_disjunct(monkeypatch):
    """With the polish refusing to move the point (main's behaviour), the
    declared-model check must withhold the certificate and the repair must
    publish a point that meets the disjunct."""
    monkeypatch.setattr(solver, "_polish_preserves_feasibility", lambda *a, **k: False)
    m, x, y = _witness()
    r = m.solve(gdp_method="hull", time_limit=60)
    stats = r.solver_stats or {}
    assert stats.get("certificate/incumbent_unverified") == 1.0
    assert not r.gap_certified and r.status != "optimal"
    assert stats.get("certificate/incumbent_repaired") == 1.0
    assert _row_violation(r, x, y) <= 1e-6
    assert r.objective >= TRUE_OPT - 1e-9


# ── verify_point on declared logic ────────────────────────────────────────────


def test_disjunction_points():
    m, _x, _y = _witness()
    checks = 0
    for point, ok in [
        ((0.6951966603219929, 1.0040445913119824), False),  # the #1659 incumbent
        ((math.log(2.0), 1.0), True),  # first disjunct
        ((2.0, 7.0), True),  # second disjunct
        ((1.5, 10.0), False),  # between them
    ]:
        res = verify_point(m, np.array(point))
        assert res.ok is ok, (point, res)
        checks += 1
    assert "disjunction" in verify_point(m, np.array([1.5, 10.0])).reason
    assert checks == 4


def test_nested_disjunction():
    m = dm.Model("nested")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 5], [x >= 8]], name="outer")
    # Nesting is an IR feature: a disjunction inside a disjunct's row list.
    inner = _DisjunctiveConstraint(disjuncts=[[x <= 1], [x >= 3, x <= 4]], name="inner")
    m._constraints[-1].disjuncts[0].append(inner)
    m.minimize(x)
    assert verify_point(m, np.array([0.5])).ok
    assert verify_point(m, np.array([3.5])).ok
    assert verify_point(m, np.array([9.0])).ok
    assert not verify_point(m, np.array([2.0])).ok  # outer arm 0 holds, inner fails
    assert not verify_point(m, np.array([6.0])).ok


def test_indicator_row_binds_only_when_active():
    m = dm.Model("ind")
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=10)
    m.if_then(z, [x >= 5])
    m.minimize(x)
    assert verify_point(m, np.array([1.0, 6.0])).ok
    assert verify_point(m, np.array([0.0, 1.0])).ok
    assert not verify_point(m, np.array([1.0, 1.0])).ok


def test_sos_cardinality():
    m = dm.Model("sos")
    v = [m.continuous(f"v{i}", lb=0, ub=1) for i in range(3)]
    m.sos1(v[:2])
    m.sos2(v)
    m.minimize(v[0])
    assert verify_point(m, np.array([0.0, 0.5, 0.5])).ok
    assert not verify_point(m, np.array([0.5, 0.5, 0.0])).ok  # SOS1 on v0, v1
    m2 = dm.Model("sos2")
    w = [m2.continuous(f"w{i}", lb=0, ub=1) for i in range(3)]
    m2.sos2(w)
    m2.minimize(w[0])
    assert verify_point(m2, np.array([0.5, 0.5, 0.0])).ok
    assert not verify_point(m2, np.array([0.5, 0.0, 0.5])).ok  # not adjacent


def test_logical_constraint():
    m = dm.Model("logic")
    a = m.boolean("a")
    b = m.boolean("b")
    m.logical(a.implies(b))
    m.minimize(0.0 * a.variable)
    assert verify_point(m, np.array([1.0, 1.0])).ok
    assert verify_point(m, np.array([0.0, 0.0])).ok
    assert not verify_point(m, np.array([1.0, 0.0])).ok
