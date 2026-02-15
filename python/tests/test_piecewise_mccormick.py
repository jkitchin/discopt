"""Tests for piecewise McCormick relaxation primitives and convex envelopes.

Validates:
  1. Soundness: cv <= f(x) <= cc for 10,000 random points
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
from discopt._jax.envelopes import (
    relax_fractional,
    relax_signomial,
    relax_trilinear,
)
from discopt._jax.mccormick import (
    relax_bilinear,
    relax_exp,
    relax_log,
    relax_sin,
    relax_sqrt,
    relax_square,
)
from discopt._jax.piecewise_mccormick import (
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


class TestPiecewiseSquareSoundness:
    def test_positive_range(self):
        key = jax.random.PRNGKey(31)
        lb, ub = 1.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_square(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, x**2, "pw square [1,5]")

    def test_mixed_sign(self):
        key = jax.random.PRNGKey(32)
        lb, ub = -3.0, 4.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_square(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, x**2, "pw square [-3,4]")


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


class TestPiecewiseTanSoundness:
    def test_standard(self):
        key = jax.random.PRNGKey(55)
        lb, ub = -1.0, 1.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_tan(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.tan(x), "pw tan [-1,1]")

    def test_positive(self):
        key = jax.random.PRNGKey(56)
        lb, ub = 0.1, 1.2
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_tan(xi, lb, ub, k=8))(x)
        _check_soundness(cv, cc, jnp.tan(x), "pw tan [0.1,1.2]")


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
        """For exp, higher curvature at larger x => smaller partitions there."""
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
        """Adaptive should reduce gap in the high-curvature region."""
        key = jax.random.PRNGKey(300)
        lb, ub = 0.0, 5.0
        # Sample from the high-curvature region (right side for exp)
        x = _random_points(key, 3.0, 5.0)
        cv_u, cc_u = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        cv_a, cc_a = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8, adaptive=True))(x)
        gap_u = jnp.mean(cc_u - cv_u)
        gap_a = jnp.mean(cc_a - cv_a)
        assert gap_a <= gap_u + 1e-8, (
            f"adaptive gap {gap_a} > uniform gap {gap_u} in high-curvature"
        )

    def test_log_adaptive_reduces_max_gap(self):
        """Adaptive should reduce gap in the high-curvature region."""
        key = jax.random.PRNGKey(301)
        lb, ub = 0.1, 10.0
        # High-curvature region for log is near lb
        x = _random_points(key, 0.1, 1.0)
        cv_u, cc_u = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=8))(x)
        cv_a, cc_a = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=8, adaptive=True))(x)
        gap_u = jnp.mean(cc_u - cv_u)
        gap_a = jnp.mean(cc_a - cv_a)
        assert gap_a <= gap_u + 1e-8

    def test_adaptive_exp_soundness(self):
        key = jax.random.PRNGKey(302)
        lb, ub = 0.0, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8, adaptive=True))(x)
        _check_soundness(cv, cc, jnp.exp(x), "adaptive exp")

    def test_adaptive_square_soundness(self):
        key = jax.random.PRNGKey(303)
        lb, ub = -3.0, 4.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_square(xi, lb, ub, k=8, adaptive=True))(x)
        _check_soundness(cv, cc, x**2, "adaptive square")


# ===================================================================
# Monotone refinement: k=16 tighter than k=4
# ===================================================================


class TestMonotoneRefinement:
    def test_exp_k16_tighter_than_k4(self):
        key = jax.random.PRNGKey(350)
        lb, ub = -2.0, 3.0
        x = _random_points(key, lb, ub)
        cv4, cc4 = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=4))(x)
        cv16, cc16 = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=16))(x)
        gap4 = jnp.mean(cc4 - cv4)
        gap16 = jnp.mean(cc16 - cv16)
        assert gap16 <= gap4 + 1e-8

    def test_log_k16_tighter_than_k4(self):
        key = jax.random.PRNGKey(351)
        lb, ub = 0.5, 8.0
        x = _random_points(key, lb, ub)
        cv4, cc4 = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=4))(x)
        cv16, cc16 = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=16))(x)
        gap4 = jnp.mean(cc4 - cv4)
        gap16 = jnp.mean(cc16 - cv16)
        assert gap16 <= gap4 + 1e-8

    def test_sqrt_k16_tighter_than_k4(self):
        key = jax.random.PRNGKey(352)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        cv4, cc4 = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=4))(x)
        cv16, cc16 = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=16))(x)
        gap4 = jnp.mean(cc4 - cv4)
        gap16 = jnp.mean(cc16 - cv16)
        assert gap16 <= gap4 + 1e-8

    def test_square_k16_tighter_than_k4(self):
        key = jax.random.PRNGKey(353)
        lb, ub = -3.0, 4.0
        x = _random_points(key, lb, ub)
        cv4, cc4 = jax.vmap(lambda xi: piecewise_relax_square(xi, lb, ub, k=4))(x)
        cv16, cc16 = jax.vmap(lambda xi: piecewise_relax_square(xi, lb, ub, k=16))(x)
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

    def test_exp_tighter(self):
        key = jax.random.PRNGKey(100)
        lb, ub = -3.0, 3.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_exp(xi, lb, ub))(x)
        std_gap = std_cc - std_cv
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv
        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL)

    def test_log_tighter(self):
        key = jax.random.PRNGKey(101)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_log(xi, lb, ub))(x)
        std_gap = std_cc - std_cv
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv
        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL)

    def test_sqrt_tighter(self):
        key = jax.random.PRNGKey(102)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_sqrt(xi, lb, ub))(x)
        std_gap = std_cc - std_cv
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=8))(x)
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

    def test_sin_tighter(self):
        key = jax.random.PRNGKey(104)
        lb, ub = -1.0, 2.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_sin(xi, lb, ub))(x)
        std_gap = std_cc - std_cv
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv
        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL)

    def test_square_tighter(self):
        key = jax.random.PRNGKey(105)
        lb, ub = -3.0, 4.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_square(xi, lb, ub))(x)
        std_gap = std_cc - std_cv
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_square(xi, lb, ub, k=8))(x)
        pw_gap = pw_cc - pw_cv
        assert jnp.all(pw_gap <= std_gap + self.TIGHT_TOL)


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
        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL)
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL)

    def test_log_k1(self):
        key = jax.random.PRNGKey(201)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_log(xi, lb, ub))(x)
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_log(xi, lb, ub, k=1))(x)
        assert jnp.allclose(std_cv, pw_cv, atol=self.MATCH_TOL)
        assert jnp.allclose(std_cc, pw_cc, atol=self.MATCH_TOL)

    def test_sqrt_k1(self):
        key = jax.random.PRNGKey(202)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        std_cv, std_cc = jax.vmap(lambda xi: relax_sqrt(xi, lb, ub))(x)
        pw_cv, pw_cc = jax.vmap(lambda xi: piecewise_relax_sqrt(xi, lb, ub, k=1))(x)
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
    def test_positive_bounds(self):
        key = jax.random.PRNGKey(700)
        k1, k2, k3 = jax.random.split(key, 3)
        x_lb, x_ub = 1.0, 3.0
        y_lb, y_ub = 2.0, 5.0
        z_lb, z_ub = 0.5, 4.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        z = _random_points(k3, z_lb, z_ub)
        cv, cc = jax.vmap(
            lambda xi, yi, zi: relax_trilinear(xi, yi, zi, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
        )(x, y, z)
        _check_soundness(cv, cc, x * y * z, "trilinear pos")

    def test_mixed_sign_bounds(self):
        key = jax.random.PRNGKey(701)
        k1, k2, k3 = jax.random.split(key, 3)
        x_lb, x_ub = -2.0, 3.0
        y_lb, y_ub = -1.0, 4.0
        z_lb, z_ub = -3.0, 2.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        z = _random_points(k3, z_lb, z_ub)
        cv, cc = jax.vmap(
            lambda xi, yi, zi: relax_trilinear(xi, yi, zi, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
        )(x, y, z)
        _check_soundness(cv, cc, x * y * z, "trilinear mixed")

    def test_negative_bounds(self):
        key = jax.random.PRNGKey(702)
        k1, k2, k3 = jax.random.split(key, 3)
        x_lb, x_ub = -5.0, -1.0
        y_lb, y_ub = -4.0, -0.5
        z_lb, z_ub = -3.0, -0.1
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        z = _random_points(k3, z_lb, z_ub)
        cv, cc = jax.vmap(
            lambda xi, yi, zi: relax_trilinear(xi, yi, zi, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
        )(x, y, z)
        _check_soundness(cv, cc, x * y * z, "trilinear neg")

    def test_unit_cube(self):
        key = jax.random.PRNGKey(703)
        k1, k2, k3 = jax.random.split(key, 3)
        lb, ub = 0.0, 1.0
        x = _random_points(k1, lb, ub)
        y = _random_points(k2, lb, ub)
        z = _random_points(k3, lb, ub)
        cv, cc = jax.vmap(lambda xi, yi, zi: relax_trilinear(xi, yi, zi, lb, ub, lb, ub, lb, ub))(
            x, y, z
        )
        _check_soundness(cv, cc, x * y * z, "trilinear unit cube")


# ===================================================================
# Fractional envelope soundness
# ===================================================================


class TestFractionalSoundness:
    def test_positive_y(self):
        key = jax.random.PRNGKey(800)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = 1.0, 3.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(lambda xi, yi: relax_fractional(xi, yi, x_lb, x_ub, y_lb, y_ub))(x, y)
        _check_soundness(cv, cc, x / y, "fractional pos y")

    def test_negative_y(self):
        key = jax.random.PRNGKey(801)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = 1.0, 5.0
        y_lb, y_ub = -3.0, -1.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(lambda xi, yi: relax_fractional(xi, yi, x_lb, x_ub, y_lb, y_ub))(x, y)
        _check_soundness(cv, cc, x / y, "fractional neg y")

    def test_mixed_x_positive_y(self):
        key = jax.random.PRNGKey(802)
        k1, k2 = jax.random.split(key)
        x_lb, x_ub = -2.0, 4.0
        y_lb, y_ub = 0.5, 3.0
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = jax.vmap(lambda xi, yi: relax_fractional(xi, yi, x_lb, x_ub, y_lb, y_ub))(x, y)
        _check_soundness(cv, cc, x / y, "fractional mixed x pos y")


# ===================================================================
# Signomial envelope soundness
# ===================================================================


class TestSignomialSoundness:
    def test_a_half(self):
        """x^0.5 is concave on (0, inf)."""
        key = jax.random.PRNGKey(900)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, lb, ub, 0.5))(x)
        _check_soundness(cv, cc, x**0.5, "signomial a=0.5")

    def test_a_1_5(self):
        """x^1.5 is convex on (0, inf)."""
        key = jax.random.PRNGKey(901)
        lb, ub = 0.1, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, lb, ub, 1.5))(x)
        _check_soundness(cv, cc, x**1.5, "signomial a=1.5")

    def test_a_2_5(self):
        """x^2.5 is convex on (0, inf)."""
        key = jax.random.PRNGKey(902)
        lb, ub = 0.5, 4.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, lb, ub, 2.5))(x)
        _check_soundness(cv, cc, x**2.5, "signomial a=2.5")

    def test_a_neg_1(self):
        """x^(-1) is convex on (0, inf)."""
        key = jax.random.PRNGKey(903)
        lb, ub = 0.5, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, lb, ub, -1.0))(x)
        _check_soundness(cv, cc, x ** (-1.0), "signomial a=-1")

    def test_a_neg_half(self):
        """x^(-0.5) is convex on (0, inf)."""
        key = jax.random.PRNGKey(904)
        lb, ub = 0.5, 5.0
        x = _random_points(key, lb, ub)
        cv, cc = jax.vmap(lambda xi: relax_signomial(xi, lb, ub, -0.5))(x)
        _check_soundness(cv, cc, x ** (-0.5), "signomial a=-0.5")


# ===================================================================
# Gap reduction: piecewise k=8 achieves >= 60% reduction vs k=1
# ===================================================================


class TestGapReduction:
    def _gap_reduction(self, std_fn, pw_fn, x, label):
        """Compute gap reduction percentage."""
        cv1, cc1 = std_fn(x)
        cv8, cc8 = pw_fn(x)
        gap1 = jnp.mean(cc1 - cv1)
        gap8 = jnp.mean(cc8 - cv8)
        reduction = 1.0 - gap8 / jnp.maximum(gap1, 1e-15)
        assert reduction >= 0.60, (
            f"{label}: gap reduction = {float(reduction):.1%} < 60% "
            f"(gap_k1={float(gap1):.6f}, gap_k8={float(gap8):.6f})"
        )
        return reduction

    def test_exp_gap_reduction(self):
        key = jax.random.PRNGKey(600)
        lb, ub = -2.0, 3.0
        x = _random_points(key, lb, ub)
        self._gap_reduction(
            lambda xi: (relax_exp(xi, lb, ub)),
            lambda xi: jax.vmap(lambda xj: piecewise_relax_exp(xj, lb, ub, k=8))(xi),
            x,
            "exp",
        )

    def test_log_gap_reduction(self):
        key = jax.random.PRNGKey(601)
        lb, ub = 0.5, 8.0
        x = _random_points(key, lb, ub)
        self._gap_reduction(
            lambda xi: (relax_log(xi, lb, ub)),
            lambda xi: jax.vmap(lambda xj: piecewise_relax_log(xj, lb, ub, k=8))(xi),
            x,
            "log",
        )

    def test_sqrt_gap_reduction(self):
        key = jax.random.PRNGKey(602)
        lb, ub = 0.1, 10.0
        x = _random_points(key, lb, ub)
        self._gap_reduction(
            lambda xi: (relax_sqrt(xi, lb, ub)),
            lambda xi: jax.vmap(lambda xj: piecewise_relax_sqrt(xj, lb, ub, k=8))(xi),
            x,
            "sqrt",
        )

    def test_square_gap_reduction(self):
        key = jax.random.PRNGKey(603)
        lb, ub = -3.0, 4.0
        x = _random_points(key, lb, ub)
        self._gap_reduction(
            lambda xi: (relax_square(xi, lb, ub)),
            lambda xi: jax.vmap(lambda xj: piecewise_relax_square(xj, lb, ub, k=8))(xi),
            x,
            "square",
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

    def test_square_jit(self):
        f = jax.jit(lambda x: piecewise_relax_square(x, -3.0, 4.0, k=4))
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

    def test_tan_jit(self):
        f = jax.jit(lambda x: piecewise_relax_tan(x, -1.0, 1.0, k=4))
        cv, cc = f(jnp.float64(0.5))
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

    def test_adaptive_exp_jit(self):
        f = jax.jit(lambda x: piecewise_relax_exp(x, 0.0, 5.0, k=8, adaptive=True))
        cv, cc = f(jnp.float64(2.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


# ===================================================================
# vmap compatibility
# ===================================================================


class TestPiecewiseVmap:
    def test_exp_vmap(self):
        key = jax.random.PRNGKey(1000)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-2.0, maxval=2.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_exp(xi, -2.0, 2.0, k=4))(x)
        assert cv.shape == (batch,)
        assert cc.shape == (batch,)
        _check_soundness(cv, cc, jnp.exp(x), "vmap pw exp")

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

    def test_log_vmap(self):
        key = jax.random.PRNGKey(1002)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=0.1, maxval=10.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_log(xi, 0.1, 10.0, k=4))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.log(x), "vmap pw log")

    def test_sin_vmap(self):
        key = jax.random.PRNGKey(1003)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-1.0, maxval=2.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_sin(xi, -1.0, 2.0, k=4))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.sin(x), "vmap pw sin")

    def test_square_vmap(self):
        key = jax.random.PRNGKey(1004)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-3.0, maxval=4.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_square(xi, -3.0, 4.0, k=4))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x**2, "vmap pw square")

    def test_tan_vmap(self):
        key = jax.random.PRNGKey(1005)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-1.0, maxval=1.0)
        cv, cc = jax.vmap(lambda xi: piecewise_relax_tan(xi, -1.0, 1.0, k=4))(x)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, jnp.tan(x), "vmap pw tan")

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

    def test_fractional_vmap(self):
        key = jax.random.PRNGKey(1007)
        k1, k2 = jax.random.split(key)
        batch = 64
        x = jax.random.uniform(k1, (batch,), dtype=jnp.float64, minval=1.0, maxval=5.0)
        y = jax.random.uniform(k2, (batch,), dtype=jnp.float64, minval=1.0, maxval=3.0)
        cv, cc = jax.vmap(lambda xi, yi: relax_fractional(xi, yi, 1.0, 5.0, 1.0, 3.0))(x, y)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x / y, "vmap fractional")

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
        lbs = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-3.0, maxval=-0.5)
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

    def test_square_grad(self):
        def loss(x):
            cv, cc = piecewise_relax_square(x, -3.0, 4.0, k=4)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"pw square grad not finite: {g}"

    def test_trilinear_grad(self):
        def loss(x):
            cv, cc = relax_trilinear(x, x + 1, x + 2, 0.0, 3.0, 1.0, 4.0, 2.0, 5.0)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"trilinear grad not finite: {g}"

    def test_signomial_grad(self):
        def loss(x):
            cv, cc = relax_signomial(x, 0.5, 5.0, 1.5)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(2.0))
        assert jnp.isfinite(g), f"signomial grad not finite: {g}"


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
        assert reduction > 0.5, f"exp gap reduction only {float(reduction):.1%}, expected > 50%"

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
        assert reduction > 0.3, (
            f"bilinear gap reduction only {float(reduction):.1%}, expected > 30%"
        )


# ===================================================================
# Phase D: Specialized envelope tests
# ===================================================================


class TestPowerIntEnvelope:
    """Tests for relax_power_int from envelopes.py."""

    def test_even_power_convex(self):
        """x^2 is convex — cv = x^2, cc >= x^2."""
        from discopt._jax.envelopes import relax_power_int

        x = jnp.linspace(-2.0, 2.0, 20)
        cv, cc = jax.vmap(lambda xi: relax_power_int(xi, -2.0, 2.0, 2))(x)
        true_val = x**2
        assert jnp.all(cv <= true_val + 1e-10)
        assert jnp.all(cc >= true_val - 1e-10)

    def test_odd_power_positive(self):
        """x^3 on [1, 5] — convex region."""
        from discopt._jax.envelopes import relax_power_int

        x = jnp.linspace(1.0, 5.0, 20)
        cv, cc = jax.vmap(lambda xi: relax_power_int(xi, 1.0, 5.0, 3))(x)
        true_val = x**3
        assert jnp.all(cv <= true_val + 1e-10)
        assert jnp.all(cc >= true_val - 1e-10)

    def test_odd_power_negative(self):
        """x^3 on [-5, -1] — concave region."""
        from discopt._jax.envelopes import relax_power_int

        x = jnp.linspace(-5.0, -1.0, 20)
        cv, cc = jax.vmap(lambda xi: relax_power_int(xi, -5.0, -1.0, 3))(x)
        true_val = x**3
        assert jnp.all(cv <= true_val + 1e-10)
        assert jnp.all(cc >= true_val - 1e-10)

    def test_odd_power_mixed(self):
        """x^3 on [-2, 3] — mixed convex/concave."""
        from discopt._jax.envelopes import relax_power_int

        x = jnp.linspace(-2.0, 3.0, 50)
        cv, cc = jax.vmap(lambda xi: relax_power_int(xi, -2.0, 3.0, 3))(x)
        true_val = x**3
        assert jnp.all(cv <= true_val + 1e-8)
        assert jnp.all(cc >= true_val - 1e-8)

    def test_x4_convex(self):
        """x^4 on [-3, 3] — even, always convex."""
        from discopt._jax.envelopes import relax_power_int

        x = jnp.linspace(-3.0, 3.0, 30)
        cv, cc = jax.vmap(lambda xi: relax_power_int(xi, -3.0, 3.0, 4))(x)
        true_val = x**4
        assert jnp.all(cv <= true_val + 1e-8)
        assert jnp.all(cc >= true_val - 1e-8)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_power_int

        f = jax.jit(lambda x: relax_power_int(x, -2.0, 2.0, 3))
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestExpBilinearEnvelope:
    """Tests for relax_exp_bilinear."""

    def test_soundness(self):
        from discopt._jax.envelopes import relax_exp_bilinear

        x = jnp.linspace(-1.0, 2.0, 20)
        y = jnp.linspace(0.5, 3.0, 20)
        cv, cc = jax.vmap(lambda xi, yi: relax_exp_bilinear(xi, yi, -1.0, 2.0, 0.5, 3.0))(x, y)
        true_val = jnp.exp(x) * y
        assert jnp.all(cv <= true_val + 1e-8)
        assert jnp.all(cc >= true_val - 1e-8)


class TestLogSumEnvelope:
    """Tests for relax_log_sum."""

    def test_soundness(self):
        from discopt._jax.envelopes import relax_log_sum

        x = jnp.linspace(1.0, 5.0, 20)
        y = jnp.linspace(1.0, 5.0, 20)
        cv, cc = jax.vmap(lambda xi, yi: relax_log_sum(xi, yi, 1.0, 5.0, 1.0, 5.0))(x, y)
        true_val = jnp.log(x + y)
        assert jnp.all(cv <= true_val + 1e-8)
        assert jnp.all(cc >= true_val - 1e-8)
