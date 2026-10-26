"""Sweep A3: a tiny coefficient is not a zero coefficient.

The separable objective lower bound (``_expression_lower_bound_for_lift``, which
feeds the ``obj_lin >= sep_lb`` row) and its helpers skipped any term whose
coefficient was ``<= 1e-12`` in magnitude, whatever the range it multiplies. The
"rigorous" bound then came out too high. The positive-product extractor (H-LOG
lift) likewise read ``(x + 1e-12)`` as ``x``, and the even-power matcher read an
exponent within 1e-12 of an even integer as that integer. Only exact zeros may be
dropped. Each case below was unsound before and is sound (or abstains) after.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax import milp_relaxation as mr


def _box(m):
    lo = np.array([v.lb for v in m._variables], dtype=float).ravel()
    hi = np.array([v.ub for v in m._variables], dtype=float).ravel()
    return lo, hi


def _sound(bound, true_min):
    return bound is None or bound <= true_min


@pytest.mark.smoke
def test_tiny_negative_x_exp_term_is_bounded():
    m = Model("xexp")
    x = m.continuous("x", lb=0, ub=40)
    y = m.continuous("y", lb=-1, ub=1)
    true_min = -1e-13 * 40 * math.exp(40)  # about -9.4e5
    b = mr._expression_lower_bound_for_lift(y**2 - 1e-13 * (x * dm.exp(x)), m, *_box(m))
    assert _sound(b, true_min), b  # was 0.0
    assert b is not None


@pytest.mark.smoke
def test_tiny_reciprocal_numerator_over_tiny_denominator():
    m = Model("recip")
    x = m.continuous("x", lb=-5, ub=5)
    d = m.continuous("d", lb=2.0**-40, ub=1)
    true_min = -1 / math.e - 2.0**-44 * 2.0**40
    b = mr._expression_lower_bound_for_lift(x * dm.exp(x) - 2.0**-44 * (1 / d), m, *_box(m))
    assert _sound(b, true_min), b  # was -1/e: the -0.0625 term was dropped


@pytest.mark.smoke
def test_tiny_affine_coefficient_on_wide_range():
    m = Model("aff")
    x = m.continuous("x", lb=-1e15, ub=1e15)
    m.continuous("y", lb=-1, ub=1)
    y = m._variables[1]
    b = mr._scaled_affine_lower_bound(1e-13 * x + y, 1.0, m, *_box(m))
    assert _sound(b, -101.0), b  # was -1.0: the -100 from 1e-13 * x was dropped


@pytest.mark.smoke
def test_tiny_cos_argument_coefficient_is_kept():
    m = Model("cos")
    n = m.integer("n", lb=1e9, ub=1e9)  # fixed, so the enumeration is one point
    expr = dm.cos(1e-12 * n + 0.5)
    b = mr._integer_affine_cos_lower_bound(expr, 1.0, m, *_box(m))
    # The true value is cos(0.501); dropping the term gave cos(0.5), which is larger.
    assert b is not None and b <= math.cos(1e-12 * 1e9 + 0.5) + 1e-15


@pytest.mark.smoke
def test_positive_product_keeps_tiny_shift():
    m = Model("pp")
    x = m.continuous("x", lb=2e-9, ub=1)
    y = m.continuous("y", lb=1, ub=2)
    lo, hi = _box(m)
    # (x + 1e-12)**-2 is not x**-2 (1e-3 relative at x = 2e-9): not a monomial.
    assert mr._extract_positive_product((x + 1e-12) ** -2 * y, m, 2, lo, hi) is None
    assert mr._extract_positive_product((1e-13 + x) * y, m, 2, lo, hi) is None
    # Positive control: an exact monomial and an exact-zero shift still match.
    assert mr._extract_positive_product(x * y, m, 2, lo, hi) == (1.0, {0: 1.0, 1: 1.0})
    assert mr._extract_positive_product((0.0 + x) * y, m, 2, lo, hi) == (1.0, {0: 1.0, 1: 1.0})


@pytest.mark.smoke
def test_even_power_exponent_must_be_exact():
    m = Model("ep")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    base = x[0] - x[1]
    assert mr._match_scaled_even_power(base ** (2 + 1e-13), 1.0) is None
    assert mr._match_scaled_even_power(base**2, 1.0) is not None
