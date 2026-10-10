"""#1677: ``if_then(1 - z, ...)`` ("if z == 0") must solve, not raise TypeError."""

import discopt.modeling as dm
import pytest


def _model(indicator_of):
    m = dm.Model("t")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.if_then(indicator_of(z), [x <= 3])
    m.maximize(x)
    return m, x, z


def test_complemented_binary_indicator_solves():
    m, _, _ = _model(lambda z: 1 - z)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(10.0, abs=1e-6)  # z = 1 relaxes x <= 3


def test_complemented_indicator_enforces_when_z_zero():
    m, x, z = _model(lambda z: 1 - z)
    m.subject_to(z <= 0)  # force z == 0 -> x <= 3 must hold
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0, abs=1e-6)


def test_non_indicator_expression_rejected_at_call():
    m = dm.Model("t")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    with pytest.raises(TypeError, match="indicator"):
        m.if_then(2 * z, [x <= 3])
