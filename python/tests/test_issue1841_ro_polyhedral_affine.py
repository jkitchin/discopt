"""#1841: the polyhedral counterpart dualized secants of rows nonlinear in p.

Per-component coefficients were finite differences ``g(p + e_j) - g(p)``,
exact only for a row affine in ``p``. ``x + exp(p) <= 30`` over ``|xi| <= 2``
around ``p = 1`` was treated as ``x + e*(e-1)*xi + e`` and certified
``x = 17.94``; the robust truth is ``30 - e**3 = 9.91``. Non-affine rows are now
refused on the polyhedral route (decided structurally), and the box
linearization route worst-cases them by sign tracking instead of the secant.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, PolyhedralUncertaintySet, RobustCounterpart

pytestmark = pytest.mark.smoke

A1 = np.array([[1.0], [-1.0]])
B1 = np.array([2.0, 2.0])

NONLINEAR = {
    "exp": lambda p, x: x + dm.exp(p),
    "square": lambda p, x: x + p * p,
    "reciprocal": lambda p, x: x + 1.0 / p,
    "abs": lambda p, x: x + abs(p),
    "power": lambda p, x: x + p**2,
    "bilinear_exp": lambda p, x: x * dm.exp(p),
}


@pytest.mark.parametrize("name", sorted(NONLINEAR))
def test_polyhedral_refuses_non_affine_row(name):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=30)
    p = m.parameter("p", value=3.0)
    m.subject_to(NONLINEAR[name](p, x) <= 30)
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="#1841"):
        RobustCounterpart(m, PolyhedralUncertaintySet(p, A=A1, b=B1)).formulate()


def test_polyhedral_affine_rows_still_supported():
    m = dm.Model("affine")
    x = m.continuous("x", lb=0, ub=30)
    y = m.continuous("y", lb=-5, ub=5)
    p = m.parameter("p", value=1.0)
    m.subject_to(x + 2.0 * p - (p - 1.0) * y / 4.0 <= 30)
    m.maximize(x)
    RobustCounterpart(m, PolyhedralUncertaintySet(p, A=A1, b=B1)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    # The row is x + 2 + xi*(2 - y/4) over xi in [-2, 2]: worst case
    # 2 + 2*|2 - y/4|, smallest at y = 5 (3.5), so x = 26.5.
    assert float(r.value(x)) == pytest.approx(26.5, abs=1e-4)


def test_box_bilinear_route_does_not_use_secant():
    """In a model routed to the box |coeff| linearization (a ``y * p`` row with
    ``y`` sign-indefinite), a separate ``x + exp(p)`` row must still get its
    true worst case ``exp(p_max)``, not the secant."""
    m = dm.Model("box_mix")
    x = m.continuous("x", lb=0, ub=30)
    y = m.continuous("y", lb=-1, ub=1)
    p = m.parameter("p", value=1.0)
    m.subject_to(x + dm.exp(p) <= 30)
    m.subject_to(y * p <= 5)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=2.0)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(30.0 - np.exp(3.0), abs=1e-4)


def test_affinity_is_structural():
    from discopt.ro.formulations._common import is_affine_in_params

    m = dm.Model("s")
    x = m.continuous("x", lb=0, ub=1)
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    names = {"p"}
    assert is_affine_in_params(x + dm.sum(np.array([1.0, -1.0]) * p), names)
    assert is_affine_in_params(x * p[0] - p[1] / 3.0, names)
    assert not is_affine_in_params(abs(p[0] - 1.0), names)
    assert not is_affine_in_params(p[0] * p[1], names)
    assert not is_affine_in_params(dm.exp(p[0]), names)
    assert not is_affine_in_params(1.0 / p[0], names)
