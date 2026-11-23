"""#2314: ``_nonlinear_point_excess`` with an infinite Jacobian entry at the point.

The term scale ``|J| @ |x|`` was ``inf * 0 = nan`` for an infinite slope at a
zero coordinate (``d/dx sqrt(x)`` at 0), and ``inf`` for one at a nonzero
coordinate. The excess then came back ``nan`` (which ``excess > tol`` accepts) or
``-inf`` (forgiving the row whatever its violation). Now such a row gets no
relative allowance, and a non-finite net residual is a violation. The fixed-node
shortcut, which turns the excess into an INFEASIBLE certificate, passes
``certify=True`` and defers to the node NLP on such a row instead.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _infer_constraint_bounds, _make_evaluator, _nonlinear_point_excess


def _excess(model, point, **kw):
    ev = _make_evaluator(model)
    cl, cu = _infer_constraint_bounds(model, ev)
    return _nonlinear_point_excess(ev, np.array(point, dtype=float), cl, cu, **kw)


def _nan_scale_model():
    m = dm.Model("nan_scale")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(dm.sqrt(x) + y <= 3)
    m.subject_to(y <= 1)
    m.minimize(-y)
    return m


def _inf_scale_model():
    m = dm.Model("inf_scale")
    x = m.continuous("x", lb=1, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(dm.sqrt(x - 1) + y >= 2)
    m.minimize(x + y)
    return m


@pytest.mark.unit
def test_nan_scale_row_does_not_hide_the_violated_row():
    excess, where, n_cmp = _excess(_nan_scale_model(), [0.0, 1.00005])
    assert n_cmp == 4
    assert math.isfinite(excess)
    assert excess == pytest.approx(5e-5, rel=1e-3)
    assert where == "constraint row 1"


@pytest.mark.unit
def test_inf_scale_row_is_not_forgiven():
    excess, where, n_cmp = _excess(_inf_scale_model(), [1.0, 0.0])
    assert n_cmp == 2
    assert excess == pytest.approx(2.0)
    assert where == "constraint row 0"


@pytest.mark.unit
def test_certify_refuses_an_unreliable_scale():
    with pytest.raises(FloatingPointError):
        _excess(_inf_scale_model(), [1.0, 0.0], certify=True)
    with pytest.raises(FloatingPointError):
        _excess(_nan_scale_model(), [0.0, 1.00005], certify=True)


@pytest.mark.unit
def test_finite_scale_rows_are_unchanged_under_certify():
    m = dm.Model("finite")
    x = m.continuous("x", lb=0, ub=4)
    m.subject_to(x * x <= 1)
    m.minimize(x)
    a = _excess(m, [2.0])
    b = _excess(m, [2.0], certify=True)
    assert a == b
    assert a[0] == pytest.approx(3.0 - 1e-9 * 4.0 * 2.0, rel=1e-12)


@pytest.mark.correctness
def test_fixed_box_shortcut_does_not_certify_a_feasible_point_infeasible(monkeypatch):
    """Exact residual 0 at the only point; float arithmetic says 4 and the slope
    in the fixed ``x = 0`` column is infinite. Must not come back ``infeasible``."""
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS", "0")
    a = 2**54 + 12
    b = 3 * a + 4
    m = dm.Model("fixed_convex")
    x = m.integer("x", lb=0, ub=0)
    y = m.continuous("y", lb=2.25, ub=2.25)
    z = m.continuous("z", lb=2.25, ub=2.25)
    w = m.continuous("w", lb=1, ub=1)
    m.subject_to(
        float(b) - float(a) * dm.sqrt(y) - float(a) * dm.sqrt(z) - 4 * dm.sqrt(w) - dm.sqrt(x) <= 0
    )
    m.minimize(y)
    r = m.solve(threads=1, time_limit=10)
    assert r.status != "infeasible"


@pytest.mark.slow
@pytest.mark.correctness
def test_hull_solve_with_sqrt_row_is_certified(monkeypatch):
    monkeypatch.setenv("DISCOPT_DEFINED_VARS", "0")
    m = dm.Model("exit_gate_solve")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=20)
    w = m.continuous("w", lb=0, ub=1)
    m.either_or([[y >= dm.exp(x) - 1, x <= 1], [y >= x**2 + 3, x >= 2]])
    m.subject_to(dm.sqrt(w) <= 0.5)
    m.minimize(y - 2 * x + w)
    r = m.solve(gdp_method="hull", time_limit=120, threads=1)
    assert r.status == "optimal"
    assert r.gap_certified
    assert r.objective == pytest.approx(1 - 2 * math.log(2), abs=1e-6)
