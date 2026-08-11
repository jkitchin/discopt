"""Tests for piecewise McCormick relaxation primitives and convex envelopes.

Validates:
  1. Soundness: cv <= f(x) <= cc for 1,000 random points
  2. Tightness: piecewise gap <= standard gap for all test points
  3. k=1 matches standard McCormick exactly
  4. Adaptive partitioning produces non-uniform breakpoints
  5. Adaptive piecewise is at least as tight as uniform
  6. Monotone refinement: k=16 is tighter than k=4
  7. Trilinear, fractional, signomial envelope soundness
  8. Gap reduction >= 60% vs standard McCormick
  9. jit/vmap compatibility
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from discopt._relax.envelopes import (
    relax_fractional,
    relax_signomial,
    relax_trilinear,
)
from discopt._relax.mccormick import (
    relax_bilinear,
    relax_exp,
    relax_log,
    relax_sqrt,
    relax_square,
)
from discopt._relax.piecewise_mccormick import (
    _adaptive_partition_bounds,
    _partition_bounds,
    piecewise_mccormick_bilinear,
    piecewise_relax_cos,
    piecewise_relax_exp,
    piecewise_relax_log,
    piecewise_relax_sin,
    piecewise_relax_sqrt,
    piecewise_relax_square,
    piecewise_relax_tan,
)

TOL = 1e-10
N_POINTS = 1_000


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
# Soundness tests (1,000 random points each)
# ===================================================================


class TestPiecewiseBilinearSoundness:
    @pytest.mark.parametrize(
        "seed, x_lb, x_ub, y_lb, y_ub, label",
        [
            (0, 1.0, 5.0, 2.0, 7.0, "pos"),
            (1, -3.0, 4.0, -2.0, 5.0, "mixed"),
        ],
    )
    def test_bounds(self, seed, x_lb, x_ub, y_lb, y_ub, label):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=8)
        )(x, y)
        _check_soundness(cv, cc, x * y, f"pw bilinear {label}")

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


class TestPiecewiseUnivSoundness:
    @pytest.mark.parametrize(
        "seed, lb, ub, pw_fn, true_fn, label",
        [
            (12, -3.0, 3.0, piecewise_relax_exp, jnp.exp, "exp"),
            (20, 0.1, 10.0, piecewise_relax_log, jnp.log, "log"),
            (30, 0.1, 10.0, piecewise_relax_sqrt, jnp.sqrt, "sqrt"),
            (32, -3.0, 4.0, piecewise_relax_square, None, "square"),
            (41, -1.0, 2.0, piecewise_relax_sin, jnp.sin, "sin"),
            (51, -2.0, 2.0, piecewise_relax_cos, jnp.cos, "cos"),
            (55, -1.0, 1.0, piecewise_relax_tan, jnp.tan, "tan"),
        ],
    )
    def test_soundness(self, seed, lb, ub, pw_fn, true_fn, label):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: pw_fn(xi, lb, ub, k=8))(x)
        expected = x**2 if true_fn is None else true_fn(x)
        _check_soundness(cv, cc, expected, f"pw {label}")


# ===================================================================
# Adaptive partitioning
# ===================================================================


class TestAdaptivePartitioning:
    def test_produces_non_uniform_breakpoints(self):
        """Adaptive partitioning on exp should NOT be equispaced."""
        lbs_u, ubs_u = _partition_bounds(0.0, 5.0, 8)
        lbs_a, ubs_a = _adaptive_partition_bounds(jnp.exp, 0.0, 5.0, 8)
        widths_u = ubs_u - lbs_u
        widths_a = ubs_a - lbs_a
        assert jnp.std(widths_u) < 1e-10
        assert jnp.std(widths_a) > 0.01, "adaptive should produce non-uniform widths"

    def test_endpoints_exact(self):
        """Adaptive partition endpoints must match lb and ub exactly."""
        lbs, ubs = _adaptive_partition_bounds(jnp.exp, -2.0, 3.0, 8)
        assert jnp.abs(lbs[0] - (-2.0)) < 1e-14
        assert jnp.abs(ubs[-1] - 3.0) < 1e-14

    def test_covers_domain(self):
        """Sub-intervals must be contiguous and cover [lb, ub]."""
        lbs, ubs = _adaptive_partition_bounds(jnp.exp, 1.0, 10.0, 12)
        for i in range(len(lbs) - 1):
            assert jnp.abs(ubs[i] - lbs[i + 1]) < 1e-12

    def test_more_partitions_near_high_curvature(self):
        """For exp, higher curvature at larger x => smaller parts."""
        lbs, ubs = _adaptive_partition_bounds(jnp.exp, 0.0, 5.0, 8)
        widths = ubs - lbs
        assert widths[-1] < widths[0]

    def test_correct_count(self):
        """Number of sub-intervals equals k."""
        for k in [4, 8, 16]:
            lbs, ubs = _adaptive_partition_bounds(jnp.exp, 0.0, 5.0, k)
            assert len(lbs) == k
            assert len(ubs) == k


# ===================================================================
# Adaptive piecewise tighter than uniform
# ===================================================================


class TestAdaptiveTighterThanUniform:
    def test_exp_adaptive_reduces_max_gap(self):
        key = jax.random.PRNGKey(300)
        lb, ub = 0.0, 5.0
        x = _random_points(key, 3.0, 5.0)
        cv_u, cc_u = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        cv_a, cc_a = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8, adaptive=True))(x)
        gap_u = jnp.mean(cc_u - cv_u)
        gap_a = jnp.mean(cc_a - cv_a)
        assert gap_a <= gap_u + 1e-8, f"adaptive gap {gap_a} > uniform gap {gap_u}"

    def test_adaptive_exp_soundness(self):
        key = jax.random.PRNGKey(302)
        lb, ub = 0.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8, adaptive=True))(x)
        _check_soundness(cv, cc, jnp.exp(x), "adaptive exp")


# ===================================================================
# Monotone refinement: k=16 tighter than k=4
# ===================================================================


class TestMonotoneRefinement:
    @pytest.mark.parametrize(
        "seed, lb, ub, pw_fn, label",
        [
            (350, -2.0, 3.0, piecewise_relax_exp, "exp"),
            (353, -3.0, 4.0, piecewise_relax_square, "square"),
        ],
    )
    def test_univariate_k16_tighter_than_k4(self, seed, lb, ub, pw_fn, label):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        cv4, cc4 = jax.vmap(lambda xi: pw_fn(xi, lb, ub, k=4))(x)
        cv16, cc16 = jax.vmap(lambda xi: pw_fn(xi, lb, ub, k=16))(x)
        gap4 = jnp.mean(cc4 - cv4)
        gap16 = jnp.mean(cc16 - cv16)
        assert gap16 <= gap4 + 1e-8

    def test_bilinear_k16_tighter_than_k4(self):
        key = jax.random.PRNGKey(354)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -2.0, 3.0
        y_lb, y_ub = 1.0, 5.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv4, cc4 = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=4)
        )(x, y)
        cv16, cc16 = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, x_lb, x_ub, y_lb, y_ub, k=16)
        )(x, y)
        gap4 = jnp.mean(cc4 - cv4)
        gap16 = jnp.mean(cc16 - cv16)
        assert gap16 <= gap4 + 1e-8


# ===================================================================
# Tightness: piecewise gap <= standard gap
# ===================================================================


class TestTightnessVsStandard:
    """Piecewise relaxations should have tighter or equal gaps."""

    TIGHT_TOL = 1e-10

    @pytest.mark.parametrize(
        "seed, lb, ub, std_fn, pw_fn",
        [
            (100, -3.0, 3.0, relax_exp, piecewise_relax_exp),
            (101, 0.1, 10.0, relax_log, piecewise_relax_log),
            (102, 0.1, 10.0, relax_sqrt, piecewise_relax_sqrt),
        ],
    )
    def test_univariate_tighter(self, seed, lb, ub, std_fn, pw_fn):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: std_fn(xi, lb, ub))(x)
        std_gap = std_cc - std_cv
        pw_cv, pw_cc = jax.vmap(lambda xi: pw_fn(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv
        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL)

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
        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL)


# ===================================================================
# k=1 matches standard McCormick exactly
# ===================================================================


class TestK1MatchesStandard:
    """With k=1, piecewise should match standard McCormick exactly."""

    MATCH_TOL = 1e-12

    @pytest.mark.parametrize(
        "seed, lb, ub, std_fn, pw_fn",
        [
            (200, -3.0, 3.0, relax_exp, piecewise_relax_exp),
            (201, 0.1, 10.0, relax_log, piecewise_relax_log),
        ],
    )
    def test_univariate_k1(self, seed, lb, ub, std_fn, pw_fn):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: std_fn(xi, lb, ub))(x)
        pw_cv, pw_cc = jax.vmap(lambda xi: pw_fn(xi, lb, ub, k=1))(x)
        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL)
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL)

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
        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL)
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL)


# ===================================================================
# Trilinear envelope soundness
# ===================================================================


class TestTrilinearSoundness:
    @pytest.mark.parametrize(
        "seed, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub, label",
        [
            (700, 1.0, 3.0, 2.0, 5.0, 0.5, 4.0, "pos"),
            (701, -2.0, 3.0, -1.0, 4.0, -3.0, 2.0, "mixed"),
        ],
    )
    def test_soundness(self, seed, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub, label):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        z = _random_points(k3, z_lb, z_ub)
        cv, cc = jax.vmap(
            lambda xi, yi, zi: relax_trilinear(
                xi,
                yi,
                zi,
                x_lb,
                x_ub,
                y_lb,
                y_ub,
                z_lb,
                z_ub,
            )
        )(x, y, z)
        _check_soundness(cv, cc, x * y * z, f"trilinear {label}")


# ===================================================================
# Fractional envelope soundness
# ===================================================================


class TestFractionalSoundness:
    @pytest.mark.parametrize(
        "seed, x_lb, x_ub, y_lb, y_ub, label",
        [
            (800, 1.0, 5.0, 1.0, 3.0, "pos y"),
            (802, -2.0, 4.0, 0.5, 3.0, "mixed x pos y"),
        ],
    )
    def test_soundness(self, seed, x_lb, x_ub, y_lb, y_ub, label):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(lambda xi, yi: relax_fractional(xi, yi, x_lb, x_ub, y_lb, y_ub))(x, y)
        _check_soundness(cv, cc, x / y, f"fractional {label}")


# ===================================================================
# Signomial envelope soundness
# ===================================================================


class TestSignomialSoundness:
    @pytest.mark.parametrize(
        "seed, lb, ub, a",
        [
            (900, 0.1, 10.0, 0.5),
            (901, 0.1, 5.0, 1.5),
            (903, 0.5, 5.0, -1.0),
        ],
    )
    def test_soundness(self, seed, lb, ub, a):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, lb, ub, a))(x)
        _check_soundness(cv, cc, x**a, f"signomial a={a}")


# ===================================================================
# Gap reduction: piecewise k=8 achieves >= 60% reduction vs k=1
# ===================================================================


class TestGapReduction:
    @pytest.mark.parametrize(
        "seed, lb, ub, std_fn, pw_fn, label",
        [
            (600, -2.0, 3.0, relax_exp, piecewise_relax_exp, "exp"),
            (
                603,
                -3.0,
                4.0,
                relax_square,
                piecewise_relax_square,
                "square",
            ),
        ],
    )
    def test_gap_reduction(self, seed, lb, ub, std_fn, pw_fn, label):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        cv1, cc1 = std_fn(x, lb, ub)
        cv8, cc8 = jax.vmap(lambda xj: pw_fn(xj, lb, ub, k=8))(x)
        gap1 = jnp.mean(cc1 - cv1)
        gap8 = jnp.mean(cc8 - cv8)
        reduction = 1.0 - gap8 / jnp.maximum(gap1, 1e-15)
        assert reduction >= 0.60, (
            f"{label}: gap reduction = {float(reduction):.1%} < 60% "
            f"(gap_k1={float(gap1):.6f}, gap_k8={float(gap8):.6f})"
        )


# ===================================================================
# JIT compatibility
# ===================================================================


class TestPiecewiseJIT:
    @pytest.mark.parametrize(
        "fn_factory",
        [
            lambda: jax.jit(lambda x: piecewise_relax_exp(x, 0.0, 2.0, k=4)),
            lambda: jax.jit(lambda x: piecewise_relax_log(x, 0.1, 5.0, k=4)),
            lambda: jax.jit(lambda x: piecewise_relax_sin(x, -1.0, 2.0, k=4)),
            lambda: jax.jit(lambda x: piecewise_relax_exp(x, 0.0, 5.0, k=8, adaptive=True)),
        ],
        ids=["exp", "log", "sin", "adaptive_exp"],
    )
    def test_univariate_jit(self, fn_factory):
        f = fn_factory()
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_bilinear_jit(self):
        f = jax.jit(lambda x, y: piecewise_mccormick_bilinear(x, y, 0.0, 3.0, 1.0, 4.0, k=4))
        cv, cc = f(jnp.float64(1.0), jnp.float64(2.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_trilinear_jit(self):
        f = jax.jit(lambda x, y, z: relax_trilinear(x, y, z, 0.0, 3.0, 1.0, 4.0, 0.5, 2.0))
        cv, cc = f(jnp.float64(1.0), jnp.float64(2.0), jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_fractional_jit(self):
        f = jax.jit(lambda x, y: relax_fractional(x, y, 1.0, 5.0, 1.0, 3.0))
        cv, cc = f(jnp.float64(2.0), jnp.float64(2.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_signomial_jit(self):
        f = jax.jit(lambda x: relax_signomial(x, 0.5, 5.0, 1.5))
        cv, cc = f(jnp.float64(2.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


# ===================================================================
# vmap compatibility
# ===================================================================


class TestPiecewiseVmap:
    @pytest.mark.parametrize(
        "seed, lb, ub, pw_fn, true_fn, label",
        [
            (1000, -2.0, 2.0, piecewise_relax_exp, jnp.exp, "exp"),
            (1002, 0.1, 10.0, piecewise_relax_log, jnp.log, "log"),
            (1004, -3.0, 4.0, piecewise_relax_square, None, "square"),
        ],
    )
    def test_univariate_vmap(self, seed, lb, ub, pw_fn, true_fn, label):
        key = jax.random.PRNGKey(seed)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=lb, maxval=ub)
        cv, cc = jax.vmap(lambda xi: pw_fn(xi, lb, ub, k=4))(x)
        assert cv.shape == (batch,)
        assert cc.shape == (batch,)
        expected = x**2 if true_fn is None else true_fn(x)
        _check_soundness(cv, cc, expected, f"vmap pw {label}")

    def test_bilinear_vmap(self):
        key = jax.random.PRNGKey(1001)
        k1, k2 = jax.random.split(key)
        batch = 64
        x = jax.random.uniform(k1, (batch,), dtype=jnp.float64, minval=1.0, maxval=5.0)
        y = jax.random.uniform(k2, (batch,), dtype=jnp.float64, minval=2.0, maxval=7.0)
        cv, cc = jax.vmap(
            lambda xi, yi: piecewise_mccormick_bilinear(xi, yi, 1.0, 5.0, 2.0, 7.0, k=4)
        )(x, y)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x * y, "vmap pw bilinear")

    def test_trilinear_vmap(self):
        key = jax.random.PRNGKey(1006)
        k1, k2, k3 = jax.random.split(key, 3)
        batch = 64
        x = jax.random.uniform(k1, (batch,), dtype=jnp.float64, minval=1.0, maxval=3.0)
        y = jax.random.uniform(k2, (batch,), dtype=jnp.float64, minval=2.0, maxval=5.0)
        z = jax.random.uniform(k3, (batch,), dtype=jnp.float64, minval=0.5, maxval=4.0)
        cv, cc = jax.vmap(
            lambda xi, yi, zi: relax_trilinear(xi, yi, zi, 1.0, 3.0, 2.0, 5.0, 0.5, 4.0)
        )(x, y, z)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x * y * z, "vmap trilinear")

    def test_signomial_vmap(self):
        key = jax.random.PRNGKey(1008)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=0.5, maxval=5.0)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, 0.5, 5.0, 1.5))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x**1.5, "vmap signomial")

    def test_vmap_varying_bounds(self):
        """Test vmap with per-element bounds."""
        key = jax.random.PRNGKey(1009)
        batch = 64
        lbs = jax.random.uniform(
            key,
            (batch,),
            dtype=jnp.float64,
            minval=-3.0,
            maxval=-0.5,
        )
        ubs = lbs + jax.random.uniform(
            jax.random.PRNGKey(1010),
            (batch,),
            dtype=jnp.float64,
            minval=0.5,
            maxval=3.0,
        )
        x = lbs + (ubs - lbs) * jax.random.uniform(
            jax.random.PRNGKey(1011), (batch,), dtype=jnp.float64
        )
        cv, cc = jax.vmap(lambda xi, lbi, ubi: piecewise_relax_exp(xi, lbi, ubi, k=4))(x, lbs, ubs)
        _check_soundness(cv, cc, jnp.exp(x), "vmap pw exp varying bounds")


# ===================================================================
# Gradient compatibility
# ===================================================================


class TestPiecewiseGradients:
    @pytest.mark.parametrize(
        "loss_fn, x_val, label",
        [
            (
                lambda x: sum(piecewise_relax_exp(x, -2.0, 2.0, k=4)),
                0.5,
                "exp",
            ),
            (
                lambda x: sum(piecewise_mccormick_bilinear(x, x + 1, 0.0, 3.0, 1.0, 4.0, k=4)),
                1.0,
                "bilinear",
            ),
            (
                lambda x: sum(piecewise_relax_square(x, -3.0, 4.0, k=4)),
                1.0,
                "square",
            ),
            (
                lambda x: sum(relax_signomial(x, 0.5, 5.0, 1.5)),
                2.0,
                "signomial",
            ),
        ],
    )
    def test_grad_finite(self, loss_fn, x_val, label):
        g = jax.grad(loss_fn)(jnp.float64(x_val))
        assert jnp.isfinite(g), f"pw {label} grad not finite: {g}"


# ===================================================================
# Root gap reduction measurement
# ===================================================================


class TestRootGapReduction:
    def test_exp_gap_reduction(self):
        key = jax.random.PRNGKey(400)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_exp(xi, lb, ub))(x)
        std_gap = jnp.mean(std_cc - std_cv)
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        pw_gap = jnp.mean(pw_cc - pw_cv)
        reduction = 1.0 - pw_gap / std_gap
        assert reduction > 0.5, f"exp gap reduction only {float(reduction):.1%}, expected > 50%"


# ===================================================================
# Phase D: Specialized envelope tests
# ===================================================================


class TestPowerIntEnvelope:
    """Tests for relax_power_int from envelopes.py."""

    @pytest.mark.parametrize(
        "lb, ub, n, tol",
        [
            (-2.0, 2.0, 2, 1e-10),
            (-2.0, 3.0, 3, 1e-8),
            (-3.0, 3.0, 4, 1e-8),
        ],
        ids=["x2", "x3_mixed", "x4"],
    )
    def test_soundness(self, lb, ub, n, tol):
        from discopt._relax.envelopes import relax_power_int

        x = jnp.linspace(lb, ub, 50)
        cv, cc = jax.vmap(lambda xi: relax_power_int(xi, lb, ub, n))(x)
        true_val = x**n
        assert jnp.all(cv <= true_val + tol)
        assert jnp.all(cc >= true_val - tol)

    def test_jit_compatible(self):
        from discopt._relax.envelopes import relax_power_int

        f = jax.jit(lambda x: relax_power_int(x, -2.0, 2.0, 3))
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestExpBilinearEnvelope:
    """Tests for relax_exp_bilinear."""

    def test_soundness(self):
        from discopt._relax.envelopes import relax_exp_bilinear

        x = jnp.linspace(-1.0, 2.0, 20)
        y = jnp.linspace(0.5, 3.0, 20)
        cv, cc = jax.vmap(lambda xi, yi: relax_exp_bilinear(xi, yi, -1.0, 2.0, 0.5, 3.0))(x, y)
        true_val = jnp.exp(x) * y
        assert jnp.all(cv <= true_val + 1e-8)
        assert jnp.all(cc >= true_val - 1e-8)


class TestLogSumEnvelope:
    """Tests for relax_log_sum."""

    def test_soundness(self):
        from discopt._relax.envelopes import relax_log_sum

        x = jnp.linspace(1.0, 5.0, 20)
        y = jnp.linspace(1.0, 5.0, 20)
        cv, cc = jax.vmap(lambda xi, yi: relax_log_sum(xi, yi, 1.0, 5.0, 1.0, 5.0))(x, y)
        true_val = jnp.log(x + y)
        assert jnp.all(cv <= true_val + 1e-8)
        assert jnp.all(cc >= true_val - 1e-8)
