"""``verify_point`` must not accept an integer outside its declared bounds (#2253).

``check_variable_bounds`` applied the continuous bound tolerance
``ABS_TOL + BOUND_REL_TOL * |bound|`` (1e-6 + 1e-4 * |b|) to INTEGER and BINARY
columns too. On a bound of magnitude 1e4 or more that slack is a whole integer or
more, so ``x = 99990`` passed for ``lb = 100000`` and ``x = 100020`` for
``ub = 100010``. ``Model.solve`` checks its incumbent with ``verify_point`` before
it reports a certificate, so that check could not catch an integer column out of
range by up to ``1e-4 * |bound|``.

An integer column is now judged on its rounded value against the integer hull of
its box, ``[ceil(lb - INT_TOL), floor(ub + INT_TOL)]``. The continuous arm is
unchanged; the control tests below pin that, so these tests cannot pass against a
verifier that simply got stricter everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import modeling as dm
from discopt.validation.feasibility import BOUND_REL_TOL, INT_TOL, verify_point


def _int_model(lb, ub, *, binary=False):
    m = dm.Model("i2253")
    x = m.binary("x") if binary else m.integer("x", lb=lb, ub=ub)
    m.minimize(x)
    return m


# -- the reproducer from the issue -------------------------------------------


@pytest.mark.parametrize("v", [99990.0, 99989.0, 99999.0, 100011.0, 100020.0])
def test_large_magnitude_integer_outside_bounds_rejected(v):
    m = _int_model(100000, 100010)
    res = verify_point(m, [v])
    assert not res.ok
    assert "out of bounds" in res.reason


@pytest.mark.parametrize("v", [100000.0, 100005.0, 100010.0])
def test_large_magnitude_integer_inside_bounds_accepted(v):
    m = _int_model(100000, 100010)
    assert verify_point(m, [v]).ok


def test_integer_within_int_tol_of_bound_accepted():
    """Rounding noise inside INT_TOL on a bound-active integer is still accepted."""
    m = _int_model(100000, 100010)
    assert verify_point(m, [100000.0 - 0.5 * INT_TOL]).ok
    assert verify_point(m, [100010.0 + 0.5 * INT_TOL]).ok


def test_solve_still_certifies_true_optimum():
    m = _int_model(100000, 100010)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(100000.0, abs=1e-6)


# -- integer hull of a non-integral or noisy box ------------------------------


def test_fractional_bounds_use_integer_hull():
    m = _int_model(0.5, 3.5)
    assert not verify_point(m, [0.0]).ok
    assert verify_point(m, [1.0]).ok
    assert verify_point(m, [3.0]).ok
    assert not verify_point(m, [4.0]).ok


def test_bound_stored_with_float_noise_admits_the_integer():
    """``lb = 2 + 1e-10`` (a presolve/round-trip artifact) must still admit 2."""
    m = _int_model(2.0 + 1e-10, 5.0 - 1e-10)
    assert verify_point(m, [2.0]).ok
    assert verify_point(m, [5.0]).ok


def test_binary_column():
    m = _int_model(None, None, binary=True)
    assert verify_point(m, [0.0]).ok
    assert verify_point(m, [1.0]).ok
    assert not verify_point(m, [2.0]).ok
    assert not verify_point(m, [-1.0]).ok


def test_unbounded_integer_column():
    m = dm.Model("i2253_unbounded")
    x = m.integer("x", lb=-1e20, ub=1e20)
    m.minimize(x)
    assert verify_point(m, [-123456789.0]).ok
    assert verify_point(m, [123456789.0]).ok


def test_integer_array_variable_one_entry_out():
    m = dm.Model("i2253_array")
    x = m.integer("x", shape=(3,), lb=np.array([0.0, 50000.0, -50000.0]), ub=50010.0)
    m.minimize(dm.sum(x))
    assert verify_point(m, [0.0, 50000.0, -50000.0]).ok
    # 4 below lb=50000 sits inside the old relative slack (5.000001).
    assert not verify_point(m, [0.0, 49996.0, -50000.0]).ok
    # 4 below lb=-50000 likewise.
    assert not verify_point(m, [0.0, 50000.0, -50004.0]).ok


def test_non_integral_value_still_rejected():
    m = _int_model(100000, 100010)
    res = verify_point(m, [100005.5])
    assert not res.ok
    assert "not integral" in res.reason


# -- control: the continuous arm keeps its relative bound tolerance ----------


def test_continuous_column_keeps_relative_bound_tolerance():
    m = dm.Model("c2253")
    x = m.continuous("x", lb=100000.0, ub=100010.0)
    m.minimize(x)
    slack = BOUND_REL_TOL * 100000.0  # 10.0
    assert verify_point(m, [100000.0 - 0.5 * slack]).ok
    assert not verify_point(m, [100000.0 - 2.0 * slack]).ok
