"""Unit tests for single-variable power / reciprocal-power canonicalization.

These back the nvs08 fix: ``term_classifier.extract_single_var_power`` folds a
product/power/sqrt tree over one variable into ``(flat_idx, exponent)``, and
``extract_reciprocal_power`` turns ``c/(x**p)`` into ``(flat_idx, -p, c)`` so a
reciprocal of a monomial product is relaxed as a fractional power rather than
dropped.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
from discopt._jax.term_classifier import extract_reciprocal_power, extract_single_var_power


def _model_one_var():
    m = dm.Model("m")
    x = m.continuous("x", lb=0.1, ub=10.0)
    return m, x


def test_bare_variable_is_power_one():
    m, x = _model_one_var()
    assert extract_single_var_power(x, m) == (0, 1.0)


def test_sqrt_is_half_power():
    m, x = _model_one_var()
    assert extract_single_var_power(dm.sqrt(x), m) == (0, 0.5)


def test_integer_power():
    m, x = _model_one_var()
    assert extract_single_var_power(x**3, m) == (0, 3.0)


def test_product_of_powers_same_var():
    m, x = _model_one_var()
    # x**3 * sqrt(x) == x**3.5
    assert extract_single_var_power(x**3 * dm.sqrt(x), m) == (0, 3.5)


def test_nested_power_of_sqrt():
    m, x = _model_one_var()
    # sqrt(x)**3 == x**1.5
    assert extract_single_var_power(dm.sqrt(x) ** 3, m) == (0, 1.5)


def test_multi_variable_product_rejected():
    m = dm.Model("m")
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    assert extract_single_var_power(x * y, m) is None


def test_reciprocal_of_monomial_product():
    m, x = _model_one_var()
    # 1/(x**3 * sqrt(x)) == x**-3.5  (the nvs08 shape)
    assert extract_reciprocal_power(1 / (x**3 * dm.sqrt(x)), m) == (0, -3.5, 1.0)


def test_reciprocal_constant_numerator_scale():
    m, x = _model_one_var()
    assert extract_reciprocal_power(5 / (x**2), m) == (0, -2.0, 5.0)


def test_non_reciprocal_returns_none():
    m, x = _model_one_var()
    assert extract_reciprocal_power(x**2, m) is None


def test_reciprocal_non_constant_numerator_rejected():
    m = dm.Model("m")
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    # y / x**2 — non-constant numerator is not a pure fractional power.
    assert extract_reciprocal_power(y / (x**2), m) is None
