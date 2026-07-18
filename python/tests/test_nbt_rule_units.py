"""Unit tests for nonlinear bound-tightening helpers and rule behavior (#87).

The monotone-function inverse helpers are checked against their defining
property (``f(inverse(rhs)) == rhs`` on the interior of the domain, and
conservative abstention at the edges). Rule behavior is exercised through
``tighten_nonlinear_bounds`` on tiny models with the documented constraint
shapes; every tightened box must (a) be a subset of the input box and
(b) retain sampled feasible points — bound tightening may never cut a
feasible solution.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nonlinear_bound_tightening import (
    _inverse_monotone_lower,
    _inverse_monotone_upper,
    _monotone_function_value,
    _safe_exp,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Monotone-function helpers
# ---------------------------------------------------------------------------


def test_safe_exp_saturation():
    assert _safe_exp(800.0) == np.inf
    assert _safe_exp(-800.0) == 0.0
    assert _safe_exp(1.0) == pytest.approx(np.e)


def test_monotone_function_value_all_functions():
    assert _monotone_function_value("exp", 0.0) == 1.0
    assert _monotone_function_value("log", np.e) == pytest.approx(1.0)
    assert _monotone_function_value("log", 0.0) == -np.inf
    assert _monotone_function_value("log2", 8.0) == pytest.approx(3.0)
    assert _monotone_function_value("log2", -1.0) == -np.inf
    assert _monotone_function_value("log10", 100.0) == pytest.approx(2.0)
    assert _monotone_function_value("log10", 0.0) == -np.inf
    assert _monotone_function_value("log1p", 0.0) == 0.0
    assert _monotone_function_value("log1p", -1.0) == -np.inf
    assert _monotone_function_value("sqrt", 9.0) == pytest.approx(3.0)
    assert np.isnan(_monotone_function_value("sqrt", -1.0))
    with pytest.raises(ValueError, match="Unsupported monotone function"):
        _monotone_function_value("sin", 0.0)


@pytest.mark.parametrize(
    "func,rhs",
    [
        ("exp", 5.0),
        ("log", 1.3),
        ("log2", 2.5),
        ("log10", 1.5),
        ("log1p", 0.7),
        ("sqrt", 3.0),
    ],
)
def test_inverse_monotone_upper_is_true_inverse(func, rhs):
    upper = _inverse_monotone_upper(func, rhs)
    assert upper is not None
    # f(U) == rhs: any arg <= U keeps f(arg) <= rhs for increasing f.
    assert _monotone_function_value(func, upper) == pytest.approx(rhs, rel=1e-12)


@pytest.mark.parametrize(
    "func,rhs",
    [
        ("exp", 5.0),
        ("log", -0.4),
        ("log2", -1.5),
        ("log10", -0.5),
        ("log1p", -0.3),
        ("sqrt", 2.0),
    ],
)
def test_inverse_monotone_lower_is_true_inverse(func, rhs):
    lower = _inverse_monotone_lower(func, rhs)
    assert lower is not None
    assert _monotone_function_value(func, lower) == pytest.approx(rhs, rel=1e-12)


def test_inverse_monotone_guard_rails():
    # exp(arg) <= 0 has no solution: abstain rather than fabricate.
    assert _inverse_monotone_upper("exp", -1.0) is None
    assert _inverse_monotone_lower("exp", 0.0) is None
    # sqrt(arg) <= negative is infeasible; >= 0 gives no lower-bound info.
    assert _inverse_monotone_upper("sqrt", -1.0) is None
    assert _inverse_monotone_lower("sqrt", 0.0) is None
    # Saturation guards for the base-2/base-10 inverses.
    assert _inverse_monotone_upper("log2", 2000.0) == np.inf
    assert _inverse_monotone_lower("log2", -2000.0) == 0.0
    assert _inverse_monotone_upper("log10", 400.0) == np.inf
    assert _inverse_monotone_lower("log10", -400.0) == 0.0
    assert _inverse_monotone_upper("cosh", 1.0) is None
    assert _inverse_monotone_lower("cosh", 1.0) is None


# ---------------------------------------------------------------------------
# Rule behavior through tighten_nonlinear_bounds
# ---------------------------------------------------------------------------


def _tighten(model):
    from discopt._jax.model_utils import flat_variable_bounds

    flat_lb, flat_ub = flat_variable_bounds(model)
    return flat_lb, flat_ub, tighten_nonlinear_bounds(model, flat_lb, flat_ub)


def _assert_subset_box(new_lb, new_ub, old_lb, old_ub):
    assert np.all(new_lb >= old_lb - 1e-12)
    assert np.all(new_ub <= old_ub + 1e-12)


def test_sqrt_sum_of_squares_rule_tightens_soundly():
    m = Model("sqrtsos")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(dm.sqrt(x**2 + 2.0 * y**2) <= 3.0)
    m.minimize(x + y)
    lb0, ub0, (lb, ub, stats) = _tighten(m)
    assert not stats.infeasible
    assert "sqrt_sum_of_squares_upper_bound" in stats.applied_rules
    _assert_subset_box(lb, ub, lb0, ub0)
    # |x| <= 3, |y| <= 3/sqrt(2).
    assert ub[0] == pytest.approx(3.0)
    assert lb[0] == pytest.approx(-3.0)
    assert ub[1] == pytest.approx(3.0 / np.sqrt(2.0))
    # Soundness: feasible points survive.
    for pt in ([0.0, 0.0], [2.9, 0.0], [0.0, 2.0]):
        if np.sqrt(pt[0] ** 2 + 2 * pt[1] ** 2) <= 3.0:
            assert lb[0] - 1e-9 <= pt[0] <= ub[0] + 1e-9
            assert lb[1] - 1e-9 <= pt[1] <= ub[1] + 1e-9


def test_sqrt_sum_of_squares_rule_proves_infeasibility():
    m = Model("sqrtneg")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.subject_to(dm.sqrt(x**2) <= -1.0)
    m.minimize(x)
    _lb0, _ub0, (lb, ub, stats) = _tighten(m)
    assert stats.infeasible
    assert "negative upper bound" in stats.infeasibility_reason


def test_monotone_equality_rule_propagates_exp():
    m = Model("monoeq")
    x = m.continuous("x", lb=-100.0, ub=100.0)
    y = m.continuous("y", lb=1.0, ub=float(np.exp(2.0)))
    m.subject_to(y == dm.exp(x))
    m.minimize(x)
    lb0, ub0, (lb, ub, stats) = _tighten(m)
    assert not stats.infeasible
    _assert_subset_box(lb, ub, lb0, ub0)
    # y in [1, e^2] forces x in [0, 2].
    assert lb[0] == pytest.approx(0.0, abs=1e-9)
    assert ub[0] == pytest.approx(2.0, abs=1e-9)
    # Feasible pairs survive.
    for xv in (0.0, 1.0, 2.0):
        assert lb[0] - 1e-9 <= xv <= ub[0] + 1e-9
        yv = float(np.exp(xv))
        assert lb[1] - 1e-9 <= yv <= ub[1] + 1e-9


def test_monotone_bound_rule_log_inequality():
    m = Model("monole")
    x = m.continuous("x", lb=0.0, ub=1e6)
    m.subject_to(dm.log(x) <= 2.0)
    m.minimize(-x)
    lb0, ub0, (lb, ub, stats) = _tighten(m)
    assert not stats.infeasible
    _assert_subset_box(lb, ub, lb0, ub0)
    assert ub[0] == pytest.approx(np.exp(2.0), rel=1e-9)
    # x = e^2 itself remains feasible.
    assert lb[0] <= np.exp(2.0) <= ub[0] + 1e-9


def test_bilinear_product_equality_rule_bounds_free_factor():
    # v * (x + 1) == 2 with x in [1, 3]: the affine factor F = x + 1 lies in
    # [2, 4] (sign-stable), so v = 2 / F is enclosed by [0.5, 1.0]. This is
    # the only rule able to bound a variable that appears solely in a
    # nonlinear product (cf. the ex6_1_4 mole-fraction ratios).
    m = Model("bilin")
    v = m.continuous("v", lb=-1000.0, ub=1000.0)
    x = m.continuous("x", lb=1.0, ub=3.0)
    m.subject_to(v * (x + 1.0) == 2.0)
    m.minimize(v)
    lb0, ub0, (lb, ub, stats) = _tighten(m)
    assert not stats.infeasible
    assert "bilinear_product_equality" in stats.applied_rules
    _assert_subset_box(lb, ub, lb0, ub0)
    assert lb[0] >= 0.5 - 1e-9
    assert ub[0] <= 1.0 + 1e-9
    # Every feasible (v, x) pair survives.
    for xv in np.linspace(1.0, 3.0, 9):
        vv = 2.0 / (xv + 1.0)
        assert lb[0] - 1e-9 <= vv <= ub[0] + 1e-9


def test_tighten_nonlinear_bounds_reports_empty_initial_interval():
    m = Model("empty")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    lb = np.array([2.0])
    ub = np.array([1.0])
    _new_lb, _new_ub, stats = tighten_nonlinear_bounds(m, lb, ub)
    assert stats.infeasible
    assert "initial interval is empty" in stats.infeasibility_reason


def test_tighten_nonlinear_bounds_noop_on_linear_model():
    m = Model("lin")
    x = m.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m.subject_to(x[0] + x[1] <= 1.0)
    m.minimize(x[0])
    lb0, ub0, (lb, ub, stats) = _tighten(m)
    assert not stats.infeasible
    assert stats.n_tightened == 0
    np.testing.assert_allclose(lb, lb0)
    np.testing.assert_allclose(ub, ub0)
