"""#1922: a negative fractional power whose base has lower bound 0 is declined by
the factorable lift instead of raising ``ZeroDivisionError`` from ``0.0 ** p``."""

import discopt.modeling as dm
import pytest
from discopt._relax.factorable_reform import _Lifter


@pytest.mark.unit
def test_lifter_declines_negative_power_on_zero_lower_bound():
    m = dm.Model("lift")
    t = m.continuous("t", lb=0.0, ub=4.0)
    lifter = _Lifter(m)
    assert lifter.fractional_power(t, -0.5) is None
    # A positive power on the same base is still lifted.
    assert lifter.fractional_power(t, 0.5) is not None


@pytest.mark.smoke
def test_negative_power_of_lifted_sum_solves():
    m = dm.Model("zerodiv")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=10.0)
    m.subject_to((x + y) ** -0.5 <= z)
    m.minimize(z + x)
    r = m.solve(time_limit=30)
    assert r.status in ("optimal", "feasible")
    assert r.objective == pytest.approx(1.0, abs=1e-4)
    assert r.bound <= 1.0 + 1e-6


def test_negative_power_of_original_variable_with_other_lift_solves():
    m = dm.Model("zerodiv_var")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x**-0.5 + x + (y + 1) ** 0.5)
    r = m.solve(time_limit=30)
    # x^-0.5 + x is smallest at x = 2^(-2/3) (value 3 * 2^(-2/3)); (y + 1)^0.5
    # is smallest (1) at y = 0.
    opt = 3 * 2 ** (-2 / 3) + 1.0
    assert r.objective == pytest.approx(opt, abs=1e-4)
    assert r.bound <= opt + 1e-6
