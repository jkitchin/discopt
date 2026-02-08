"""Tests for piecewise McCormick relaxation primitives.

Validates:
  1. Soundness: cv <= f(x) <= cc for 10,000 random points
  2. Tightness: piecewise gap <= standard gap for all test points
  3. k=1 matches standard McCormick exactly
  4. jit/vmap compatibility
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

import jax
import jax.numpy as jnp
from discopt._jax.mccormick import (
    relax_bilinear,
    relax_exp,
    relax_log,
    relax_sin,
    relax_sqrt,
)
from discopt._jax.piecewise_mccormick import (
    piecewise_mccormick_bilinear,
    piecewise_relax_cos,
    piecewise_relax_exp,
    piecewise_relax_log,
    piecewise_relax_sin,
    piecewise_relax_sqrt,
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
    assert jnp.all(cv <= cc + TOL), f"cv > cc{msg}: max violation = {jnp.max(cv - cc)}"


# ===================================================================
# Soundness tests (10,000 random points each)
# ===================================================================


class TestPiecewiseBilinearSoundness:
    def test_positive_bounds(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 2.0, 7.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=8)
        )(x, y)
        _check_soundness(cv, cc, x * y, "pw bilinear pos")

    def test_mixed_sign_bounds(self):
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -3.0, 4.0
        y_lb, y_ub = -2.0, 5.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=8)
        )(x, y)
        _check_soundness(cv, cc, x * y, "pw bilinear mixed")

    def test_negative_bounds(self):
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -5.0, -1.0
        y_lb, y_ub = -7.0, -2.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=8)
        )(x, y)
        _check_soundness(cv, cc, x * y, "pw bilinear neg")

    def test_different_k_values(self):
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 2.0, 7.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        for k_val in [2, 4, 8, 16]:
            cv, cc = jax.vmap(
                lambda xi, yi, _k=k_val: piecewise_mccormick_bilinear(
                    xi, yi, x_lb, x_ub, y_lb, y_ub, k=_k
                )
            )(x, y)
            _check_soundness(cv, cc, x * y, f"pw bilinear k={k_val}")


class TestPiecewiseExpSoundness:
    def test_positive_range(self):
        key = jax.random.PRNGKey(10)
        lb, ub = 0.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.exp(x), "pw exp [0,3]")

    def test_negative_range(self):
        key = jax.random.PRNGKey(11)
        lb, ub = -5.0, -1.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.exp(x), "pw exp [-5,-1]")

    def test_wide_range(self):
        key = jax.random.PRNGKey(12)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.exp(x), "pw exp [-3,3]")


class TestPiecewiseLogSoundness:
    def test_standard(self):
        key = jax.random.PRNGKey(20)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.log(x), "pw log [0.1,10]")


class TestPiecewiseSqrtSoundness:
    def test_standard(self):
        key = jax.random.PRNGKey(30)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.sqrt(x), "pw sqrt [0.1,10]")


class TestPiecewiseSinSoundness:
    def test_narrow_positive(self):
        key = jax.random.PRNGKey(40)
        lb, ub = 0.1, 1.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.sin(x), "pw sin [0.1,1]")

    def test_mixed(self):
        key = jax.random.PRNGKey(41)
        lb, ub = -1.0, 2.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.sin(x), "pw sin [-1,2]")

    def test_wide(self):
        key = jax.random.PRNGKey(42)
        lb, ub = -4.0, 4.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.sin(x), "pw sin [-4,4]")


class TestPiecewiseCosSoundness:
    def test_narrow(self):
        key = jax.random.PRNGKey(50)
        lb, ub = 0.5, 1.5
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_cos(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.cos(x), "pw cos [0.5,1.5]")

    def test_mixed(self):
        key = jax.random.PRNGKey(51)
        lb, ub = -2.0, 2.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_cos(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.cos(x), "pw cos [-2,2]")


# ===================================================================
# Tightness: piecewise gap <= standard gap
# ===================================================================


class TestTightnessVsStandard:
    """Piecewise relaxations should have tighter or equal gaps."""

    TIGHT_TOL = 1e-10

    def test_exp_tighter(self):
        key = jax.random.PRNGKey(100)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_exp(xi, lb, ub))(x)
        std_gap = std_cc - std_cv

        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv

        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL), (
            f"pw exp gap not tighter: max excess = {jnp.max(pw_gap - std_gap)}"
        )

    def test_log_tighter(self):
        key = jax.random.PRNGKey(101)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_log(xi, lb, ub))(x)
        std_gap = std_cc - std_cv

        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv

        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL), (
            f"pw log gap not tighter: max excess = {jnp.max(pw_gap - std_gap)}"
        )

    def test_sqrt_tighter(self):
        key = jax.random.PRNGKey(102)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_sqrt(xi, lb, ub))(x)
        std_gap = std_cc - std_cv

        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv

        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL), (
            f"pw sqrt gap not tighter: max excess = {jnp.max(pw_gap - std_gap)}"
        )

    def test_bilinear_tighter(self):
        key = jax.random.PRNGKey(103)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 2.0, 7.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)

        std_cv, std_cc = jax.vmap(lambda xi, yi: relax_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub))(
            x, y
        )
        std_gap = std_cc - std_cv

        pw_cv, pw_cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=8)
        )(x, y)
        pw_gap = pw_cc - pw_cv

        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL), (
            f"pw bilinear gap not tighter: max excess = {jnp.max(pw_gap - std_gap)}"
        )

    def test_sin_tighter(self):
        key = jax.random.PRNGKey(104)
        lb, ub = -1.0, 2.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_sin(xi, lb, ub))(x)
        std_gap = std_cc - std_cv

        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv

        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL), (
            f"pw sin gap not tighter: max excess = {jnp.max(pw_gap - std_gap)}"
        )


# ===================================================================
# k=1 matches standard McCormick exactly
# ===================================================================


class TestK1MatchesStandard:
    """With k=1, piecewise should match standard McCormick exactly."""

    MATCH_TOL = 1e-12

    def test_exp_k1(self):
        key = jax.random.PRNGKey(200)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_exp(xi, lb, ub))(x)
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=1))(x)

        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL), (
            f"exp k=1 cv mismatch: max diff = {jnp.max(jnp.abs(std_cv - pw_cv))}"
        )
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL), (
            f"exp k=1 cc mismatch: max diff = {jnp.max(jnp.abs(std_cc - pw_cc))}"
        )

    def test_log_k1(self):
        key = jax.random.PRNGKey(201)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_log(xi, lb, ub))(x)
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=1))(x)

        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL), (
            f"log k=1 cv mismatch: max diff = {jnp.max(jnp.abs(std_cv - pw_cv))}"
        )
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL), (
            f"log k=1 cc mismatch: max diff = {jnp.max(jnp.abs(std_cc - pw_cc))}"
        )

    def test_sqrt_k1(self):
        key = jax.random.PRNGKey(202)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_sqrt(xi, lb, ub))(x)
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=1))(x)

        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL), (
            f"sqrt k=1 cv mismatch: max diff = {jnp.max(jnp.abs(std_cv - pw_cv))}"
        )
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL), (
            f"sqrt k=1 cc mismatch: max diff = {jnp.max(jnp.abs(std_cc - pw_cc))}"
        )

    def test_bilinear_k1(self):
        key = jax.random.PRNGKey(203)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 2.0, 7.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)

        std_cv, std_cc = jax.vmap(lambda xi, yi: relax_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub))(
            x, y
        )
        pw_cv, pw_cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=1)
        )(x, y)

        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL), (
            f"bilinear k=1 cv mismatch: max diff = {jnp.max(jnp.abs(std_cv - pw_cv))}"
        )
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL), (
            f"bilinear k=1 cc mismatch: max diff = {jnp.max(jnp.abs(std_cc - pw_cc))}"
        )


# ===================================================================
# JIT compatibility
# ===================================================================


class TestPiecewiseJIT:
    def test_exp_jit(self):
        f = jax.jit(lambda x: piecewise_relax_exp(x, 0.0, 2.0, k=4))
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_log_jit(self):
        f = jax.jit(lambda x: piecewise_relax_log(x, 0.1, 5.0, k=4))
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_sqrt_jit(self):
        f = jax.jit(lambda x: piecewise_relax_sqrt(x, 0.1, 5.0, k=4))
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_sin_jit(self):
        f = jax.jit(lambda x: piecewise_relax_sin(x, -1.0, 2.0, k=4))
        cv, cc = f(jnp.float64(0.5))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_cos_jit(self):
        f = jax.jit(lambda x: piecewise_relax_cos(x, -1.0, 2.0, k=4))
        cv, cc = f(jnp.float64(0.5))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_bilinear_jit(self):
        f = jax.jit(lambda x, y: piecewise_mccormick_bilinear(x, y, 0.0, 3.0, 1.0, 4.0, k=4))
        cv, cc = f(jnp.float64(1.0), jnp.float64(2.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


# ===================================================================
# vmap compatibility
# ===================================================================


class TestPiecewiseVmap:
    def test_exp_vmap(self):
        key = jax.random.PRNGKey(300)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-2.0, maxval=2.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, -2.0, 2.0, k=4))(x)
        assert cv.shape == (batch,)
        assert cc.shape == (batch,)
        _check_soundness(cv, cc, jnp.exp(x), "vmap pw exp")

    def test_bilinear_vmap(self):
        key = jax.random.PRNGKey(301)
        k1, k2 = jax.random.split(key)
        batch = 64
        x = jax.random.uniform(k1, (batch,), dtype=jnp.float64, minval=1.0, maxval=5.0)
        y = jax.random.uniform(k2, (batch,), dtype=jnp.float64, minval=2.0, maxval=7.0)
        cv, cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, 1.0, 5.0, 2.0, 7.0, k=4)
        )(x, y)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x * y, "vmap pw bilinear")

    def test_log_vmap(self):
        key = jax.random.PRNGKey(302)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=0.1, maxval=10.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_log(xi, 0.1, 10.0, k=4))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.log(x), "vmap pw log")

    def test_sin_vmap(self):
        key = jax.random.PRNGKey(303)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-1.0, maxval=2.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, -1.0, 2.0, k=4))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.sin(x), "vmap pw sin")

    def test_vmap_varying_bounds(self):
        """Test vmap with per-element bounds."""
        key = jax.random.PRNGKey(304)
        batch = 64
        lbs = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-3.0, maxval=-0.5)
        ubs = lbs + jax.random.uniform(
            jax.random.PRNGKey(305), (batch,), dtype=jnp.float64, minval=0.5, maxval=3.0
        )
        x = lbs + (ubs - lbs) * jax.random.uniform(
            jax.random.PRNGKey(306), (batch,), dtype=jnp.float64
        )
        cv, cc = jax.vmap(lambda xi, lbi, ubi: piecewise_relax_exp(xi, lbi, ubi, k=4))(x, lbs, ubs)
        _check_soundness(cv, cc, jnp.exp(x), "vmap pw exp varying bounds")


# ===================================================================
# Gradient compatibility
# ===================================================================


class TestPiecewiseGradients:
    def test_exp_grad(self):
        def loss(x):
            cv, cc = piecewise_relax_exp(x, -2.0, 2.0, k=4)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(0.5))
        assert jnp.isfinite(g), f"pw exp grad not finite: {g}"

    def test_log_grad(self):
        def loss(x):
            cv, cc = piecewise_relax_log(x, 0.1, 5.0, k=4)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"pw log grad not finite: {g}"

    def test_bilinear_grad(self):
        def loss(x):
            cv, cc = piecewise_mccormick_bilinear(x, x + 1, 0.0, 3.0, 1.0, 4.0, k=4)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"pw bilinear grad not finite: {g}"

    def test_sin_grad(self):
        def loss(x):
            cv, cc = piecewise_relax_sin(x, -1.0, 2.0, k=4)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(0.5))
        assert jnp.isfinite(g), f"pw sin grad not finite: {g}"


# ===================================================================
# Root gap reduction measurement
# ===================================================================


class TestRootGapReduction:
    """Measure the gap reduction achieved by piecewise vs standard."""

    def test_exp_gap_reduction(self):
        key = jax.random.PRNGKey(400)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)

        std_cv, std_cc = jax.vmap(lambda xi: relax_exp(xi, lb, ub))(x)
        std_gap = jnp.mean(std_cc - std_cv)

        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        pw_gap = jnp.mean(pw_cc - pw_cv)

        reduction = 1.0 - pw_gap / std_gap
        # With k=8, we expect significant gap reduction for exp
        assert reduction > 0.5, f"exp gap reduction only {reduction:.1%}, expected > 50%"

    def test_bilinear_gap_reduction(self):
        key = jax.random.PRNGKey(401)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -3.0, 4.0
        y_lb, y_ub = -2.0, 5.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)

        std_cv, std_cc = jax.vmap(lambda xi, yi: relax_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub))(
            x, y
        )
        std_gap = jnp.mean(std_cc - std_cv)

        pw_cv, pw_cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=8)
        )(x, y)
        pw_gap = jnp.mean(pw_cc - pw_cv)

        reduction = 1.0 - pw_gap / std_gap
        assert reduction > 0.3, f"bilinear gap reduction only {reduction:.1%}, expected > 30%"
