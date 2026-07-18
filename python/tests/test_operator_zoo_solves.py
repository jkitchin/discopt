"""Operator-zoo certification battery for the uniform relaxation engine (#87).

Each test solves a tiny model whose objective/constraints use one of the
less-traveled atom classes (tan on a branch-safe box, inverse trig, hyperbolics,
fractional powers, variable division, relative entropy, integer trig squares)
end-to-end and asserts the closed-form optimum — exercising the engine's
analytic-envelope emitters with a certificate check, not just execution.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke


def _certify(res, opt, tol=1e-3):
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(opt, abs=tol)
    if res.bound is not None:
        assert res.bound <= res.objective + 1e-6


def test_tan_on_branch_safe_box():
    # tan is monotone on (-pi/2, pi/2); min tan(x) + x^2 on [-1, 1]:
    # f'(x) = sec^2 x + 2x = 0 -> x ~ -0.4795; f(x*) ~ -0.29144.
    m = Model("tan")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(dm.tan(x) + x**2)
    res = m.solve(time_limit=60.0)
    xs = np.linspace(-1, 1, 20001)
    true_min = float(np.min(np.tan(xs) + xs**2))
    _certify(res, true_min)


def test_inverse_trig_objective():
    # min asin(x) + acos(x) is constant pi/2 on [-1,1] — a brutal flatness
    # test: any certified value must be pi/2.
    m = Model("invtrig")
    x = m.continuous("x", lb=-0.9, ub=0.9)
    m.minimize(dm.asin(x) + dm.acos(x))
    res = m.solve(time_limit=60.0)
    _certify(res, float(np.pi / 2))


def test_atan_constraint():
    # min x s.t. atan(x) >= 0.5 -> x = tan(0.5).
    m = Model("atan")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    m.subject_to(dm.atan(x) >= 0.5)
    m.minimize(x)
    res = m.solve(time_limit=60.0)
    _certify(res, float(np.tan(0.5)))


def test_hyperbolics():
    # min sinh(x) + cosh(x) = e^x on [-1, 1] -> e^-1 at x = -1.
    m = Model("hyper")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(dm.sinh(x) + dm.cosh(x))
    res = m.solve(time_limit=60.0)
    _certify(res, float(np.exp(-1.0)))


def test_fractional_power_objective():
    # min x^0.5 + (2-x)^0.5 on [0, 2]: symmetric concave sum, minimized at
    # the ENDPOINTS (0 or 2) with value sqrt(2).
    m = Model("fracpow")
    x = m.continuous("x", lb=0.0, ub=2.0)
    m.minimize(x**0.5 + (2.0 - x) ** 0.5)
    res = m.solve(time_limit=60.0)
    _certify(res, float(np.sqrt(2.0)))


def test_variable_division_constraint():
    # min x + y s.t. x/y >= 2, y >= 0.5 -> x = 2y, minimized at y = 0.5:
    # objective 1.5.
    m = Model("vardiv")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.subject_to(x / y >= 2.0)
    m.minimize(x + y)
    res = m.solve(time_limit=60.0)
    _certify(res, 1.5)


def test_relative_entropy_objective():
    # min x*log(x/y) + (y-1)^2 with x fixed by constraint x == 0.5:
    # inner opt over y: d/dy [0.5 log(0.5/y) + (y-1)^2] = -0.5/y + 2(y-1) = 0
    # -> 2y^2 - 2y - 0.5 = 0 -> y = (2 + sqrt(8))/4 ~ 1.20711.
    m = Model("relent")
    x = m.continuous("x", lb=0.1, ub=1.0)
    y = m.continuous("y", lb=0.1, ub=2.0)
    m.subject_to(x == 0.5)
    m.minimize(x * dm.log(x / y) + (y - 1.0) ** 2)
    res = m.solve(time_limit=90.0)
    ys = np.linspace(0.1, 2.0, 200001)
    true_min = float(np.min(0.5 * np.log(0.5 / ys) + (ys - 1.0) ** 2))
    _certify(res, true_min)


def test_integer_trig_square():
    # min sin(i)^2 over integer i in [1, 6]: exact enumeration -> i = 3
    # (sin(3)^2 ~ 0.0199) beats the others.
    m = Model("trigsq")
    i = m.integer("i", lb=1, ub=6)
    m.minimize(dm.sin(i) ** 2)
    res = m.solve(time_limit=60.0)
    true_min = min(float(np.sin(k) ** 2) for k in range(1, 7))
    _certify(res, true_min, tol=1e-4)


def test_mixed_power_bilinear_equality():
    # x^1.5 * y appears via a scaled equality lift: min x + y s.t.
    # x^1.5 * y >= 1 on [0.5, 2]^2. At the boundary y = x^-1.5; minimize
    # x + x^-1.5 -> derivative 1 - 1.5 x^-2.5 = 0 -> x = 1.5^(1/2.5).
    m = Model("powbil")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.subject_to((x**1.5) * y >= 1.0)
    m.minimize(x + y)
    res = m.solve(time_limit=90.0)
    xs = np.linspace(0.5, 2.0, 200001)
    feas = np.clip(xs**-1.5, 0.5, None)
    mask = feas <= 2.0
    true_min = float(np.min(xs[mask] + feas[mask]))
    _certify(res, true_min)
