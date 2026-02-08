"""Tests for alphaBB convex underestimators.

Validates:
  1. Convexity: Hessian of underestimator is PSD at random points
  2. Soundness: underestimator(x) <= f(x) for all x in [lb, ub]
  3. Tightness: underestimator touches f at boundary points
  4. Overestimator soundness: overestimator(x) >= f(x)
  5. JIT/vmap compatibility
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

import jax
import jax.numpy as jnp
from discopt._jax.alphabb import (
    alphabb_overestimator,
    alphabb_underestimator,
    estimate_alpha,
    make_alphabb_relaxation,
)

TOL = 1e-6
N_POINTS = 1_000


def _random_box_points(key, lb, ub, n=N_POINTS):
    """Generate n random points in the box [lb, ub]."""
    ndim = lb.shape[0]
    return lb + (ub - lb) * jax.random.uniform(key, shape=(n, ndim), dtype=jnp.float64)


# ===================================================================
# Test functions (known nonconvex functions)
# ===================================================================


def _rosenbrock(x):
    """Rosenbrock: f(x) = (1-x0)^2 + 100*(x1 - x0^2)^2. Nonconvex."""
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def _rastrigin_2d(x):
    """2D Rastrigin: highly nonconvex with many local minima."""
    A = 10.0
    return (
        A * 2
        + (x[0] ** 2 - A * jnp.cos(2 * jnp.pi * x[0]))
        + (x[1] ** 2 - A * jnp.cos(2 * jnp.pi * x[1]))
    )


def _simple_nonconvex(x):
    """Simple nonconvex: f(x) = sin(x0) * cos(x1) + x0^2."""
    return jnp.sin(x[0]) * jnp.cos(x[1]) + x[0] ** 2


def _bilinear(x):
    """Bilinear: f(x) = x0 * x1. Indefinite Hessian."""
    return x[0] * x[1]


# ===================================================================
# Alpha estimation tests
# ===================================================================


class TestAlphaEstimation:
    def test_convex_function_zero_alpha(self):
        """For a convex function, alpha should be ~0."""

        def convex_f(x):
            return x[0] ** 2 + x[1] ** 2

        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(convex_f, lb, ub, n_samples=50)
        # Alpha should be very small (just the safety margin)
        assert jnp.all(alpha < 0.01), f"alpha for convex fn should be ~0, got {alpha}"

    def test_nonconvex_function_positive_alpha(self):
        """For a nonconvex function, alpha should be positive."""
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_bilinear, lb, ub, n_samples=50)
        assert jnp.all(alpha > 0), f"alpha for bilinear should be > 0, got {alpha}"

    def test_alpha_shape(self):
        """Alpha should have the same dimension as the input."""
        lb = jnp.array([-1.0, -1.0, -1.0])
        ub = jnp.array([1.0, 1.0, 1.0])

        def f(x):
            return x[0] * x[1] + x[1] * x[2]

        alpha = estimate_alpha(f, lb, ub)
        assert alpha.shape == (3,), f"alpha shape mismatch: {alpha.shape}"


# ===================================================================
# Soundness tests (underestimator <= f)
# ===================================================================


class TestUnderestimatorSoundness:
    def test_rosenbrock(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(0)
        points = _random_box_points(key, lb, ub)

        for i in range(points.shape[0]):
            x = points[i]
            under = alphabb_underestimator(_rosenbrock, x, lb, ub, alpha)
            true_val = _rosenbrock(x)
            assert under <= true_val + TOL, f"underestimator > f at {x}: {under} > {true_val}"

    def test_rastrigin(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rastrigin_2d, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(1)
        points = _random_box_points(key, lb, ub)

        for i in range(points.shape[0]):
            x = points[i]
            under = alphabb_underestimator(_rastrigin_2d, x, lb, ub, alpha)
            true_val = _rastrigin_2d(x)
            assert under <= true_val + TOL, f"underestimator > f at {x}: {under} > {true_val}"

    def test_simple_nonconvex(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        alpha = estimate_alpha(_simple_nonconvex, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(2)
        points = _random_box_points(key, lb, ub)

        for i in range(points.shape[0]):
            x = points[i]
            under = alphabb_underestimator(_simple_nonconvex, x, lb, ub, alpha)
            true_val = _simple_nonconvex(x)
            assert under <= true_val + TOL, f"underestimator > f at {x}: {under} > {true_val}"

    def test_bilinear(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        alpha = estimate_alpha(_bilinear, lb, ub, n_samples=50)

        key = jax.random.PRNGKey(3)
        points = _random_box_points(key, lb, ub)

        for i in range(points.shape[0]):
            x = points[i]
            under = alphabb_underestimator(_bilinear, x, lb, ub, alpha)
            true_val = _bilinear(x)
            assert under <= true_val + TOL, f"underestimator > f at {x}: {under} > {true_val}"


# ===================================================================
# Overestimator soundness (overestimator >= f)
# ===================================================================


class TestOverestimatorSoundness:
    def test_rosenbrock(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        def neg_f(x):
            return -_rosenbrock(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(10)
        points = _random_box_points(key, lb, ub)

        for i in range(points.shape[0]):
            x = points[i]
            over = alphabb_overestimator(_rosenbrock, x, lb, ub, alpha_neg)
            true_val = _rosenbrock(x)
            assert over >= true_val - TOL, f"overestimator < f at {x}: {over} < {true_val}"

    def test_simple_nonconvex(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])

        def neg_f(x):
            return -_simple_nonconvex(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(11)
        points = _random_box_points(key, lb, ub)

        for i in range(points.shape[0]):
            x = points[i]
            over = alphabb_overestimator(_simple_nonconvex, x, lb, ub, alpha_neg)
            true_val = _simple_nonconvex(x)
            assert over >= true_val - TOL, f"overestimator < f at {x}: {over} < {true_val}"


# ===================================================================
# Convexity: Hessian of underestimator should be PSD
# ===================================================================


class TestConvexity:
    def _check_psd(self, f, lb, ub, label=""):
        """Check that the underestimator's Hessian is PSD."""
        alpha = estimate_alpha(f, lb, ub, n_samples=100)

        def under(x):
            return alphabb_underestimator(f, x, lb, ub, alpha)

        hess_fn = jax.hessian(under)

        key = jax.random.PRNGKey(20)
        points = _random_box_points(key, lb, ub, n=200)

        for i in range(points.shape[0]):
            x = points[i]
            H = hess_fn(x)
            eigvals = jnp.linalg.eigvalsh(H)
            min_eig = jnp.min(eigvals)
            assert min_eig >= -TOL, (
                f"Non-PSD Hessian for {label} at {x}: min eigenvalue = {min_eig}"
            )

    def test_rosenbrock_convex(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        self._check_psd(_rosenbrock, lb, ub, "rosenbrock")

    def test_bilinear_convex(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        self._check_psd(_bilinear, lb, ub, "bilinear")

    def test_simple_nonconvex_convex(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        self._check_psd(_simple_nonconvex, lb, ub, "simple_nonconvex")

    def test_rastrigin_convex(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        self._check_psd(_rastrigin_2d, lb, ub, "rastrigin")


# ===================================================================
# Tightness: underestimator touches f at boundary corners
# ===================================================================


class TestTightnessAtBoundary:
    """The alphaBB perturbation is zero at box corners, so L(corner) = f(corner)."""

    TIGHT_TOL = 1e-10

    def test_rosenbrock_boundary(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub)

        # Test all 4 corners
        corners = [
            jnp.array([lb[0], lb[1]]),
            jnp.array([lb[0], ub[1]]),
            jnp.array([ub[0], lb[1]]),
            jnp.array([ub[0], ub[1]]),
        ]
        for corner in corners:
            under = alphabb_underestimator(_rosenbrock, corner, lb, ub, alpha)
            true_val = _rosenbrock(corner)
            assert jnp.abs(under - true_val) < self.TIGHT_TOL, (
                f"Not tight at corner {corner}: under={under}, f={true_val}"
            )

    def test_bilinear_boundary(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        alpha = estimate_alpha(_bilinear, lb, ub)

        corners = [
            jnp.array([lb[0], lb[1]]),
            jnp.array([lb[0], ub[1]]),
            jnp.array([ub[0], lb[1]]),
            jnp.array([ub[0], ub[1]]),
        ]
        for corner in corners:
            under = alphabb_underestimator(_bilinear, corner, lb, ub, alpha)
            true_val = _bilinear(corner)
            assert jnp.abs(under - true_val) < self.TIGHT_TOL, (
                f"Not tight at corner {corner}: under={under}, f={true_val}"
            )

    def test_overestimator_boundary(self):
        """Overestimator should also touch f at corners."""
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        def neg_f(x):
            return -_rosenbrock(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        corners = [
            jnp.array([lb[0], lb[1]]),
            jnp.array([lb[0], ub[1]]),
            jnp.array([ub[0], lb[1]]),
            jnp.array([ub[0], ub[1]]),
        ]
        for corner in corners:
            over = alphabb_overestimator(_rosenbrock, corner, lb, ub, alpha_neg)
            true_val = _rosenbrock(corner)
            assert jnp.abs(over - true_val) < self.TIGHT_TOL, (
                f"Overestimator not tight at corner {corner}: over={over}, f={true_val}"
            )


# ===================================================================
# JIT compatibility
# ===================================================================


class TestJITCompatibility:
    def test_underestimator_jit(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub)

        @jax.jit
        def under(x):
            return alphabb_underestimator(_rosenbrock, x, lb, ub, alpha)

        x = jnp.array([0.5, 0.5])
        val = under(x)
        assert jnp.isfinite(val), f"JIT underestimator not finite: {val}"

    def test_overestimator_jit(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        def neg_f(x):
            return -_rosenbrock(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        @jax.jit
        def over(x):
            return alphabb_overestimator(_rosenbrock, x, lb, ub, alpha_neg)

        x = jnp.array([0.5, 0.5])
        val = over(x)
        assert jnp.isfinite(val), f"JIT overestimator not finite: {val}"

    def test_make_relaxation_jit(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        under_fn, over_fn, _, _ = make_alphabb_relaxation(_simple_nonconvex, lb, ub)

        under_jit = jax.jit(under_fn)
        over_jit = jax.jit(over_fn)

        x = jnp.array([0.5, 0.5])
        u = under_jit(x)
        o = over_jit(x)
        assert jnp.isfinite(u) and jnp.isfinite(o)
        assert u <= _simple_nonconvex(x) + TOL
        assert o >= _simple_nonconvex(x) - TOL


# ===================================================================
# vmap compatibility
# ===================================================================


class TestVmapCompatibility:
    def test_underestimator_vmap(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub)

        key = jax.random.PRNGKey(30)
        points = _random_box_points(key, lb, ub, n=64)

        def under(x):
            return alphabb_underestimator(_rosenbrock, x, lb, ub, alpha)

        vals = jax.vmap(under)(points)
        true_vals = jax.vmap(_rosenbrock)(points)

        assert vals.shape == (64,)
        assert jnp.all(vals <= true_vals + TOL), (
            f"vmap underestimator violation: max = {jnp.max(vals - true_vals)}"
        )

    def test_overestimator_vmap(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        def neg_f(x):
            return -_rosenbrock(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        key = jax.random.PRNGKey(31)
        points = _random_box_points(key, lb, ub, n=64)

        def over(x):
            return alphabb_overestimator(_rosenbrock, x, lb, ub, alpha_neg)

        vals = jax.vmap(over)(points)
        true_vals = jax.vmap(_rosenbrock)(points)

        assert vals.shape == (64,)
        assert jnp.all(vals >= true_vals - TOL), (
            f"vmap overestimator violation: max = {jnp.max(true_vals - vals)}"
        )


# ===================================================================
# Gradient compatibility
# ===================================================================


class TestGradientCompatibility:
    def test_underestimator_grad(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub)

        def under(x):
            return alphabb_underestimator(_rosenbrock, x, lb, ub, alpha)

        x = jnp.array([0.5, 0.5])
        g = jax.grad(under)(x)
        assert jnp.all(jnp.isfinite(g)), f"underestimator grad not finite: {g}"

    def test_overestimator_grad(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        def neg_f(x):
            return -_rosenbrock(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        def over(x):
            return alphabb_overestimator(_rosenbrock, x, lb, ub, alpha_neg)

        x = jnp.array([0.5, 0.5])
        g = jax.grad(over)(x)
        assert jnp.all(jnp.isfinite(g)), f"overestimator grad not finite: {g}"


# ===================================================================
# make_alphabb_relaxation convenience function
# ===================================================================


class TestMakeRelaxation:
    def test_soundness(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        under_fn, over_fn, alpha, alpha_neg = make_alphabb_relaxation(_simple_nonconvex, lb, ub)

        key = jax.random.PRNGKey(40)
        points = _random_box_points(key, lb, ub, n=500)

        for i in range(points.shape[0]):
            x = points[i]
            u = under_fn(x)
            o = over_fn(x)
            fval = _simple_nonconvex(x)
            assert u <= fval + TOL, f"under > f at {x}: {u} > {fval}"
            assert o >= fval - TOL, f"over < f at {x}: {o} < {fval}"
            assert u <= o + TOL, f"under > over at {x}: {u} > {o}"

    def test_alpha_returned(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        _, _, alpha, alpha_neg = make_alphabb_relaxation(_bilinear, lb, ub)
        assert alpha.shape == (2,)
        assert alpha_neg.shape == (2,)
        assert jnp.all(alpha >= 0)
        assert jnp.all(alpha_neg >= 0)
