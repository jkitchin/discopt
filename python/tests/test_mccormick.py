"""Tests for McCormick relaxation primitives.

The soundness invariant is NON-NEGOTIABLE:
  cv <= f(x) + tol   (convex underestimator)
  cc >= f(x) - tol   (concave overestimator)
  cv <= cc + tol      (ordering)
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

import jax
import jax.numpy as jnp
import pytest

from discopt._jax.mccormick import (
    relax_abs,
    relax_add,
    relax_bilinear,
    relax_cos,
    relax_div,
    relax_exp,
    relax_log,
    relax_log2,
    relax_log10,
    relax_max,
    relax_min,
    relax_neg,
    relax_pow,
    relax_sign,
    relax_sin,
    relax_sqrt,
    relax_square,
    relax_sub,
    relax_tan,
)

TOL = 1e-10
N_POINTS = 10_000


def _random_points(key, lb, ub, n=N_POINTS):
    """Generate n random points in [lb, ub]."""
    return lb + (ub - lb) * jax.random.uniform(key, shape=(n,), dtype=jnp.float64)


def _check_soundness(cv, cc, true_val, label=""):
    """Assert the non-negotiable soundness invariant."""
    msg = f" [{label}]" if label else ""
    assert jnp.all(cv <= true_val + TOL), (
        f"cv > f(x){msg}: max violation = {jnp.max(cv - true_val)}"
    )
    assert jnp.all(cc >= true_val - TOL), (
        f"cc < f(x){msg}: max violation = {jnp.max(true_val - cc)}"
    )
    assert jnp.all(cv <= cc + TOL), (
        f"cv > cc{msg}: max violation = {jnp.max(cv - cc)}"
    )


# ===================================================================
# Soundness tests (10,000 random points each)
# ===================================================================

class TestBilinearSoundness:
    def test_positive_bounds(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 2.0, 7.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x * y, "bilinear pos")

    def test_mixed_sign_bounds(self):
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -3.0, 4.0
        y_lb, y_ub = -2.0, 5.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x * y, "bilinear mixed")

    def test_negative_bounds(self):
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -5.0, -1.0
        y_lb, y_ub = -7.0, -2.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x * y, "bilinear neg")


class TestExpSoundness:
    def test_positive_range(self):
        key = jax.random.PRNGKey(10)
        lb, ub = 0.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_exp(x, lb, ub)
        _check_soundness(cv, cc, jnp.exp(x), "exp [0,3]")

    def test_negative_range(self):
        key = jax.random.PRNGKey(11)
        lb, ub = -5.0, -1.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_exp(x, lb, ub)
        _check_soundness(cv, cc, jnp.exp(x), "exp [-5,-1]")

    def test_wide_range(self):
        key = jax.random.PRNGKey(12)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_exp(x, lb, ub)
        _check_soundness(cv, cc, jnp.exp(x), "exp [-3,3]")


class TestSquareSoundness:
    def test_positive_range(self):
        key = jax.random.PRNGKey(20)
        lb, ub = 1.0, 4.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_square(x, lb, ub)
        _check_soundness(cv, cc, x ** 2, "square [1,4]")

    def test_mixed_sign(self):
        key = jax.random.PRNGKey(21)
        lb, ub = -3.0, 2.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_square(x, lb, ub)
        _check_soundness(cv, cc, x ** 2, "square [-3,2]")


class TestSqrtSoundness:
    def test_standard(self):
        key = jax.random.PRNGKey(30)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sqrt(x, lb, ub)
        _check_soundness(cv, cc, jnp.sqrt(x), "sqrt")


class TestLogSoundness:
    def test_log(self):
        key = jax.random.PRNGKey(40)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_log(x, lb, ub)
        _check_soundness(cv, cc, jnp.log(x), "log")

    def test_log2(self):
        key = jax.random.PRNGKey(41)
        lb, ub = 0.5, 8.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_log2(x, lb, ub)
        _check_soundness(cv, cc, jnp.log2(x), "log2")

    def test_log10(self):
        key = jax.random.PRNGKey(42)
        lb, ub = 0.1, 100.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_log10(x, lb, ub)
        _check_soundness(cv, cc, jnp.log10(x), "log10")


class TestSinSoundness:
    def test_narrow_positive(self):
        key = jax.random.PRNGKey(50)
        lb, ub = 0.1, 1.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sin(x, lb, ub)
        _check_soundness(cv, cc, jnp.sin(x), "sin [0.1,1]")

    def test_narrow_negative(self):
        key = jax.random.PRNGKey(51)
        lb, ub = -3.0, -2.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sin(x, lb, ub)
        _check_soundness(cv, cc, jnp.sin(x), "sin [-3,-2]")

    def test_mixed(self):
        key = jax.random.PRNGKey(52)
        lb, ub = -1.0, 2.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sin(x, lb, ub)
        _check_soundness(cv, cc, jnp.sin(x), "sin [-1,2]")

    def test_wide(self):
        key = jax.random.PRNGKey(53)
        lb, ub = -4.0, 4.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sin(x, lb, ub)
        _check_soundness(cv, cc, jnp.sin(x), "sin [-4,4]")

    def test_full_period(self):
        key = jax.random.PRNGKey(54)
        lb, ub = 0.0, 2 * jnp.pi + 1.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sin(x, lb, ub)
        _check_soundness(cv, cc, jnp.sin(x), "sin [0,2pi+1]")


class TestCosSoundness:
    def test_narrow(self):
        key = jax.random.PRNGKey(60)
        lb, ub = 0.5, 1.5
        x = _random_points(key, lb, ub)
        cv, cc = relax_cos(x, lb, ub)
        _check_soundness(cv, cc, jnp.cos(x), "cos [0.5,1.5]")

    def test_mixed(self):
        key = jax.random.PRNGKey(61)
        lb, ub = -2.0, 2.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_cos(x, lb, ub)
        _check_soundness(cv, cc, jnp.cos(x), "cos [-2,2]")

    def test_wide(self):
        key = jax.random.PRNGKey(62)
        lb, ub = -5.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_cos(x, lb, ub)
        _check_soundness(cv, cc, jnp.cos(x), "cos [-5,5]")


class TestTanSoundness:
    def test_standard(self):
        key = jax.random.PRNGKey(70)
        lb, ub = -1.0, 1.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_tan(x, lb, ub)
        _check_soundness(cv, cc, jnp.tan(x), "tan [-1,1]")

    def test_positive(self):
        key = jax.random.PRNGKey(71)
        lb, ub = 0.1, 1.2
        x = _random_points(key, lb, ub)
        cv, cc = relax_tan(x, lb, ub)
        _check_soundness(cv, cc, jnp.tan(x), "tan [0.1,1.2]")


class TestAbsSoundness:
    def test_positive(self):
        key = jax.random.PRNGKey(80)
        lb, ub = 1.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_abs(x, lb, ub)
        _check_soundness(cv, cc, jnp.abs(x), "abs [1,5]")

    def test_mixed(self):
        key = jax.random.PRNGKey(81)
        lb, ub = -3.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_abs(x, lb, ub)
        _check_soundness(cv, cc, jnp.abs(x), "abs [-3,5]")

    def test_negative(self):
        key = jax.random.PRNGKey(82)
        lb, ub = -5.0, -1.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_abs(x, lb, ub)
        _check_soundness(cv, cc, jnp.abs(x), "abs [-5,-1]")


class TestPowSoundness:
    def test_even_power_positive(self):
        key = jax.random.PRNGKey(90)
        lb, ub = 1.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 2)
        _check_soundness(cv, cc, x ** 2, "x^2 [1,3]")

    def test_even_power_mixed(self):
        key = jax.random.PRNGKey(91)
        lb, ub = -2.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 2)
        _check_soundness(cv, cc, x ** 2, "x^2 [-2,3]")

    def test_even_power_4(self):
        key = jax.random.PRNGKey(92)
        lb, ub = -2.0, 2.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 4)
        _check_soundness(cv, cc, x ** 4, "x^4 [-2,2]")

    def test_odd_power_positive(self):
        key = jax.random.PRNGKey(93)
        lb, ub = 0.5, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 3)
        _check_soundness(cv, cc, x ** 3, "x^3 [0.5,3]")

    def test_odd_power_negative(self):
        key = jax.random.PRNGKey(94)
        lb, ub = -3.0, -0.5
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 3)
        _check_soundness(cv, cc, x ** 3, "x^3 [-3,-0.5]")

    def test_odd_power_mixed(self):
        key = jax.random.PRNGKey(95)
        lb, ub = -2.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 3)
        _check_soundness(cv, cc, x ** 3, "x^3 [-2,3]")

    def test_linear(self):
        key = jax.random.PRNGKey(96)
        lb, ub = -5.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, 1)
        _check_soundness(cv, cc, x, "x^1")


class TestDivSoundness:
    def test_positive_divisor(self):
        key = jax.random.PRNGKey(100)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 1.0, 3.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_div(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x / y, "div pos/pos")

    def test_negative_divisor(self):
        key = jax.random.PRNGKey(101)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = -3.0, -1.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_div(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x / y, "div pos/neg")


class TestSignSoundness:
    def test_positive(self):
        key = jax.random.PRNGKey(110)
        lb, ub = 0.5, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sign(x, lb, ub)
        _check_soundness(cv, cc, jnp.sign(x), "sign pos")

    def test_negative(self):
        key = jax.random.PRNGKey(111)
        lb, ub = -5.0, -0.5
        x = _random_points(key, lb, ub)
        cv, cc = relax_sign(x, lb, ub)
        _check_soundness(cv, cc, jnp.sign(x), "sign neg")

    def test_mixed(self):
        key = jax.random.PRNGKey(112)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = relax_sign(x, lb, ub)
        _check_soundness(cv, cc, jnp.sign(x), "sign mixed")


class TestAddSubNeg:
    def test_add_soundness(self):
        key = jax.random.PRNGKey(120)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, -5.0, 5.0)
        y = _random_points(k2, -3.0, 7.0)
        # Use exp relaxations as inputs to add
        cv_x, cc_x = relax_exp(x, -5.0, 5.0)
        cv_y, cc_y = relax_exp(y, -3.0, 7.0)
        cv, cc = relax_add(cv_x, cc_x, cv_y, cc_y)
        true_val = jnp.exp(x) + jnp.exp(y)
        _check_soundness(cv, cc, true_val, "add")

    def test_sub_soundness(self):
        key = jax.random.PRNGKey(121)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, -5.0, 5.0)
        y = _random_points(k2, -3.0, 7.0)
        cv_x, cc_x = relax_exp(x, -5.0, 5.0)
        cv_y, cc_y = relax_exp(y, -3.0, 7.0)
        cv, cc = relax_sub(cv_x, cc_x, cv_y, cc_y)
        true_val = jnp.exp(x) - jnp.exp(y)
        _check_soundness(cv, cc, true_val, "sub")

    def test_neg_soundness(self):
        key = jax.random.PRNGKey(122)
        x = _random_points(key, -5.0, 5.0)
        cv_x, cc_x = relax_exp(x, -5.0, 5.0)
        cv, cc = relax_neg(cv_x, cc_x)
        true_val = -jnp.exp(x)
        _check_soundness(cv, cc, true_val, "neg")


class TestMinMaxSoundness:
    def test_min_soundness(self):
        key = jax.random.PRNGKey(130)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, 0.1, 5.0)
        y = _random_points(k2, 0.1, 5.0)
        cv_x, cc_x = relax_sqrt(x, 0.1, 5.0)
        cv_y, cc_y = relax_sqrt(y, 0.1, 5.0)
        cv, cc = relax_min(x, y, cv_x, cc_x, cv_y, cc_y)
        true_val = jnp.minimum(jnp.sqrt(x), jnp.sqrt(y))
        _check_soundness(cv, cc, true_val, "min")

    def test_max_soundness(self):
        key = jax.random.PRNGKey(131)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, 0.1, 5.0)
        y = _random_points(k2, 0.1, 5.0)
        cv_x, cc_x = relax_sqrt(x, 0.1, 5.0)
        cv_y, cc_y = relax_sqrt(y, 0.1, 5.0)
        cv, cc = relax_max(x, y, cv_x, cc_x, cv_y, cc_y)
        true_val = jnp.maximum(jnp.sqrt(x), jnp.sqrt(y))
        _check_soundness(cv, cc, true_val, "max")


# ===================================================================
# Tightness at bounds
# ===================================================================

class TestTightnessAtBounds:
    """When x = lb or x = ub, relaxations should equal the true value."""

    TIGHT_TOL = 1e-10

    def test_exp_tight(self):
        lb, ub = -2.0, 3.0
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_exp(x, lb, ub)
            fval = jnp.exp(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"exp cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"exp cc not tight at x={x}"

    def test_square_tight(self):
        lb, ub = -2.0, 3.0
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_square(x, lb, ub)
            fval = x ** 2
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"square cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"square cc not tight at x={x}"

    def test_sqrt_tight(self):
        lb, ub = 0.5, 4.0
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_sqrt(x, lb, ub)
            fval = jnp.sqrt(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"sqrt cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"sqrt cc not tight at x={x}"

    def test_log_tight(self):
        lb, ub = 0.5, 4.0
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_log(x, lb, ub)
            fval = jnp.log(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"log cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"log cc not tight at x={x}"

    def test_sin_tight(self):
        lb, ub = 0.5, 1.5
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_sin(x, lb, ub)
            fval = jnp.sin(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"sin cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"sin cc not tight at x={x}"

    def test_cos_tight(self):
        lb, ub = 0.5, 1.5
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_cos(x, lb, ub)
            fval = jnp.cos(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"cos cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"cos cc not tight at x={x}"

    def test_bilinear_tight(self):
        x_lb, x_ub = 1.0, 3.0
        y_lb, y_ub = 2.0, 5.0
        for x, y in [(x_lb, y_lb), (x_lb, y_ub), (x_ub, y_lb), (x_ub, y_ub)]:
            x, y = jnp.float64(x), jnp.float64(y)
            cv, cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
            fval = x * y
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, (
                f"bilinear cv not tight at x={x}, y={y}"
            )
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, (
                f"bilinear cc not tight at x={x}, y={y}"
            )

    def test_abs_tight(self):
        lb, ub = -3.0, 5.0
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_abs(x, lb, ub)
            fval = jnp.abs(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"abs cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"abs cc not tight at x={x}"

    def test_tan_tight(self):
        lb, ub = -1.0, 1.0
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_tan(x, lb, ub)
            fval = jnp.tan(x)
            assert jnp.abs(cv - fval) < self.TIGHT_TOL, f"tan cv not tight at x={x}"
            assert jnp.abs(cc - fval) < self.TIGHT_TOL, f"tan cc not tight at x={x}"


# ===================================================================
# Degenerate bounds (lb ≈ ub)
# ===================================================================

class TestDegenerateBounds:
    """When lb ≈ ub, both relaxations should approximate f(x)."""

    DEGEN_TOL = 1e-8

    def test_exp_degenerate(self):
        x = jnp.float64(1.5)
        lb = ub = x
        cv, cc = relax_exp(x, lb, ub)
        fval = jnp.exp(x)
        assert jnp.abs(cv - fval) < self.DEGEN_TOL
        assert jnp.abs(cc - fval) < self.DEGEN_TOL

    def test_log_degenerate(self):
        x = jnp.float64(2.0)
        lb = ub = x
        cv, cc = relax_log(x, lb, ub)
        fval = jnp.log(x)
        assert jnp.abs(cv - fval) < self.DEGEN_TOL
        assert jnp.abs(cc - fval) < self.DEGEN_TOL

    def test_sqrt_degenerate(self):
        x = jnp.float64(4.0)
        lb = ub = x
        cv, cc = relax_sqrt(x, lb, ub)
        fval = jnp.sqrt(x)
        assert jnp.abs(cv - fval) < self.DEGEN_TOL
        assert jnp.abs(cc - fval) < self.DEGEN_TOL

    def test_sin_degenerate(self):
        x = jnp.float64(1.0)
        lb = ub = x
        cv, cc = relax_sin(x, lb, ub)
        fval = jnp.sin(x)
        assert jnp.abs(cv - fval) < self.DEGEN_TOL
        assert jnp.abs(cc - fval) < self.DEGEN_TOL

    def test_square_degenerate(self):
        x = jnp.float64(3.0)
        lb = ub = x
        cv, cc = relax_square(x, lb, ub)
        fval = x ** 2
        assert jnp.abs(cv - fval) < self.DEGEN_TOL
        assert jnp.abs(cc - fval) < self.DEGEN_TOL


# ===================================================================
# Gap monotonicity
# ===================================================================

class TestGapMonotonicity:
    """As bounds tighten, relaxation gap should decrease."""

    def _gap_at_midpoint(self, relax_fn, f, center, half_widths):
        """Compute relaxation gap at center for progressively tighter bounds."""
        gaps = []
        x = jnp.float64(center)
        for hw in half_widths:
            lb = center - hw
            ub = center + hw
            cv, cc = relax_fn(x, lb, ub)
            fval = f(x)
            # Verify soundness
            assert cv <= fval + 1e-10
            assert cc >= fval - 1e-10
            gaps.append(float(cc - cv))
        return gaps

    def test_exp_gap_decreases(self):
        half_widths = [4.0, 2.0, 1.0, 0.5, 0.1]
        gaps = self._gap_at_midpoint(relax_exp, jnp.exp, 1.0, half_widths)
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-10, (
                f"exp gap not monotone: {gaps}"
            )

    def test_log_gap_decreases(self):
        half_widths = [3.0, 1.5, 0.75, 0.3, 0.1]
        gaps = self._gap_at_midpoint(relax_log, jnp.log, 4.0, half_widths)
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-10, (
                f"log gap not monotone: {gaps}"
            )

    def test_sqrt_gap_decreases(self):
        half_widths = [3.0, 1.5, 0.75, 0.3, 0.1]
        gaps = self._gap_at_midpoint(relax_sqrt, jnp.sqrt, 4.0, half_widths)
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-10, (
                f"sqrt gap not monotone: {gaps}"
            )

    def test_square_gap_decreases(self):
        half_widths = [4.0, 2.0, 1.0, 0.5, 0.1]
        gaps = self._gap_at_midpoint(relax_square, lambda x: x ** 2, 2.0, half_widths)
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-10, (
                f"square gap not monotone: {gaps}"
            )


# ===================================================================
# JIT compatibility
# ===================================================================

class TestJITCompatibility:
    """All relaxation functions should work under jax.jit."""

    def test_exp_jit(self):
        f = jax.jit(relax_exp)
        x = jnp.float64(1.0)
        cv, cc = f(x, 0.0, 2.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_square_jit(self):
        f = jax.jit(relax_square)
        x = jnp.float64(1.0)
        cv, cc = f(x, -1.0, 2.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_sqrt_jit(self):
        f = jax.jit(relax_sqrt)
        x = jnp.float64(1.0)
        cv, cc = f(x, 0.1, 2.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_log_jit(self):
        f = jax.jit(relax_log)
        x = jnp.float64(1.0)
        cv, cc = f(x, 0.1, 2.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_sin_jit(self):
        f = jax.jit(relax_sin)
        x = jnp.float64(1.0)
        cv, cc = f(x, 0.0, 2.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_cos_jit(self):
        f = jax.jit(relax_cos)
        x = jnp.float64(1.0)
        cv, cc = f(x, 0.0, 2.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_tan_jit(self):
        f = jax.jit(relax_tan)
        x = jnp.float64(0.5)
        cv, cc = f(x, -1.0, 1.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_bilinear_jit(self):
        f = jax.jit(relax_bilinear)
        cv, cc = f(jnp.float64(1.0), jnp.float64(2.0), 0.0, 3.0, 1.0, 4.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_abs_jit(self):
        f = jax.jit(relax_abs)
        cv, cc = f(jnp.float64(1.0), -2.0, 3.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_pow_jit(self):
        f = jax.jit(lambda x, lb, ub: relax_pow(x, lb, ub, 3))
        cv, cc = f(jnp.float64(1.0), -2.0, 3.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_sign_jit(self):
        f = jax.jit(relax_sign)
        cv, cc = f(jnp.float64(1.0), -2.0, 3.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_div_jit(self):
        f = jax.jit(relax_div)
        cv, cc = f(jnp.float64(2.0), jnp.float64(3.0), 1.0, 5.0, 1.0, 4.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


# ===================================================================
# vmap compatibility
# ===================================================================

class TestVmapCompatibility:
    """Relaxation functions should work under jax.vmap over batched inputs/bounds."""

    def test_exp_vmap(self):
        key = jax.random.PRNGKey(200)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-2.0, maxval=2.0)
        lbs = jnp.full(batch, -2.0)
        ubs = jnp.full(batch, 2.0)
        cv, cc = jax.vmap(relax_exp)(x, lbs, ubs)
        assert cv.shape == (batch,)
        assert cc.shape == (batch,)
        _check_soundness(cv, cc, jnp.exp(x), "vmap exp")

    def test_bilinear_vmap(self):
        key = jax.random.PRNGKey(201)
        k1, k2 = jax.random.split(key)
        batch = 64
        x = jax.random.uniform(k1, (batch,), dtype=jnp.float64, minval=1.0, maxval=5.0)
        y = jax.random.uniform(k2, (batch,), dtype=jnp.float64, minval=2.0, maxval=7.0)
        x_lbs = jnp.full(batch, 1.0)
        x_ubs = jnp.full(batch, 5.0)
        y_lbs = jnp.full(batch, 2.0)
        y_ubs = jnp.full(batch, 7.0)
        cv, cc = jax.vmap(relax_bilinear)(x, y, x_lbs, x_ubs, y_lbs, y_ubs)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x * y, "vmap bilinear")

    def test_sin_vmap(self):
        key = jax.random.PRNGKey(202)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-1.0, maxval=2.0)
        lbs = jnp.full(batch, -1.0)
        ubs = jnp.full(batch, 2.0)
        cv, cc = jax.vmap(relax_sin)(x, lbs, ubs)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.sin(x), "vmap sin")

    def test_log_vmap(self):
        key = jax.random.PRNGKey(203)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=0.1, maxval=10.0)
        lbs = jnp.full(batch, 0.1)
        ubs = jnp.full(batch, 10.0)
        cv, cc = jax.vmap(relax_log)(x, lbs, ubs)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.log(x), "vmap log")

    def test_pow_vmap(self):
        key = jax.random.PRNGKey(204)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-2.0, maxval=3.0)
        lbs = jnp.full(batch, -2.0)
        ubs = jnp.full(batch, 3.0)
        cv, cc = jax.vmap(lambda xi, lbi, ubi: relax_pow(xi, lbi, ubi, 3))(x, lbs, ubs)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x ** 3, "vmap pow")

    def test_vmap_different_bounds(self):
        """Test that vmap works with varying bounds per element."""
        key = jax.random.PRNGKey(205)
        batch = 64
        lbs = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-3.0, maxval=-0.5)
        ubs = lbs + jax.random.uniform(
            jax.random.PRNGKey(206), (batch,), dtype=jnp.float64, minval=0.5, maxval=3.0
        )
        x = lbs + (ubs - lbs) * jax.random.uniform(
            jax.random.PRNGKey(207), (batch,), dtype=jnp.float64
        )
        cv, cc = jax.vmap(relax_exp)(x, lbs, ubs)
        _check_soundness(cv, cc, jnp.exp(x), "vmap exp varying bounds")


# ===================================================================
# Gradient through relaxations
# ===================================================================

class TestGradients:
    """jax.grad of relaxations should return finite values."""

    def test_exp_grad(self):
        def loss(x):
            cv, cc = relax_exp(x, -2.0, 2.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(0.5))
        assert jnp.isfinite(g), f"exp grad is not finite: {g}"

    def test_square_grad(self):
        def loss(x):
            cv, cc = relax_square(x, -2.0, 2.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(0.5))
        assert jnp.isfinite(g), f"square grad is not finite: {g}"

    def test_sqrt_grad(self):
        def loss(x):
            cv, cc = relax_sqrt(x, 0.1, 5.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"sqrt grad is not finite: {g}"

    def test_log_grad(self):
        def loss(x):
            cv, cc = relax_log(x, 0.1, 5.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"log grad is not finite: {g}"

    def test_sin_grad(self):
        def loss(x):
            cv, cc = relax_sin(x, -1.0, 2.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(0.5))
        assert jnp.isfinite(g), f"sin grad is not finite: {g}"

    def test_cos_grad(self):
        def loss(x):
            cv, cc = relax_cos(x, -1.0, 2.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(0.5))
        assert jnp.isfinite(g), f"cos grad is not finite: {g}"

    def test_tan_grad(self):
        def loss(x):
            cv, cc = relax_tan(x, -1.0, 1.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(0.3))
        assert jnp.isfinite(g), f"tan grad is not finite: {g}"

    def test_bilinear_grad(self):
        def loss(x):
            cv, cc = relax_bilinear(x, x + 1, 0.0, 3.0, 1.0, 4.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"bilinear grad is not finite: {g}"

    def test_abs_grad(self):
        def loss(x):
            cv, cc = relax_abs(x, -3.0, 3.0)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"abs grad is not finite: {g}"

    def test_pow_grad(self):
        def loss(x):
            cv, cc = relax_pow(x, -2.0, 3.0, 3)
            return cv + cc
        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"pow grad is not finite: {g}"
