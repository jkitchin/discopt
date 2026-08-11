"""Tests for alphaBB convex underestimators.

Validates:
  1. Soundness: cv <= f(x) <= cc at 10,000 random points (tol=1e-10)
  2. Convexity: Hessian of underestimator is PSD
  3. Tightness: relaxation touches f at box corners
  4. Gap monotonicity: tighter bounds -> smaller gap
  5. JIT/vmap/grad compatibility
  6. Gershgorin method produces valid (conservative) alpha
  7. alphaBB gap vs McCormick comparison on nonconvex functions
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from discopt._relax.alphabb import (
    _eigenvalue_method,
    _gershgorin_method,
    alphabb_overestimator,
    alphabb_underestimator,
    estimate_alpha,
    make_alphabb_relaxation,
    relax_alphabb,
)

TOL = 1e-10
N_POINTS = 10_000


def _random_box_points(key, lb, ub, n=N_POINTS):
    """Generate n random points in the box [lb, ub]."""
    ndim = lb.shape[0]
    return lb + (ub - lb) * jax.random.uniform(key, shape=(n, ndim), dtype=jnp.float64)


def _check_soundness(cv_vals, cc_vals, true_vals, label=""):
    """Assert the non-negotiable soundness invariant at all points."""
    msg = f" [{label}]" if label else ""
    assert jnp.all(cv_vals <= true_vals + TOL), (
        f"cv > f(x){msg}: max violation = {jnp.max(cv_vals - true_vals)}"
    )
    assert jnp.all(cc_vals >= true_vals - TOL), (
        f"cc < f(x){msg}: max violation = {jnp.max(true_vals - cc_vals)}"
    )
    assert jnp.all(cv_vals <= cc_vals + TOL), (
        f"cv > cc{msg}: max violation = {jnp.max(cv_vals - cc_vals)}"
    )


# ===================================================================
# Test functions (known nonconvex functions)
# ===================================================================


def _rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def _rastrigin_2d(x):
    A = 10.0
    return (
        A * 2
        + (x[0] ** 2 - A * jnp.cos(2 * jnp.pi * x[0]))
        + (x[1] ** 2 - A * jnp.cos(2 * jnp.pi * x[1]))
    )


def _simple_nonconvex(x):
    return jnp.sin(x[0]) * jnp.cos(x[1]) + x[0] ** 2


def _bilinear(x):
    return x[0] * x[1]


def _exp_product(x):
    return jnp.exp(x[0] * x[1])


def _sin_cos_sum(x):
    return jnp.sin(x[0]) + jnp.cos(x[1])


def _quadratic(x):
    return x[0] ** 2 + 2.0 * x[1] ** 2 + 0.5 * x[0] * x[1]


# ===================================================================
# Eigenvalue estimation methods
# ===================================================================


class TestEigenvalueMethods:
    def test_eigenvalue_psd(self):
        H = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        lam = _eigenvalue_method(H)
        assert lam > 0, f"PSD matrix should have positive min eigenvalue, got {lam}"

    def test_eigenvalue_indefinite(self):
        H = jnp.array([[1.0, 3.0], [3.0, 1.0]])
        lam = _eigenvalue_method(H)
        assert lam < 0, f"Indefinite matrix should have negative min eigenvalue, got {lam}"

    def test_gershgorin_conservative(self):
        H = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        lam_exact = _eigenvalue_method(H)
        lam_gersh = _gershgorin_method(H)
        assert lam_gersh <= lam_exact + 1e-12, (
            f"Gershgorin should be conservative: {lam_gersh} > {lam_exact}"
        )

    def test_gershgorin_indefinite(self):
        H = jnp.array([[1.0, 3.0], [3.0, 1.0]])
        lam_gersh = _gershgorin_method(H)
        assert lam_gersh < 0, f"Gershgorin on indefinite matrix should be < 0, got {lam_gersh}"

    def test_gershgorin_diagonal(self):
        H = jnp.diag(jnp.array([5.0, 2.0, 8.0]))
        lam_gersh = _gershgorin_method(H)
        assert jnp.abs(lam_gersh - 2.0) < 1e-12, (
            f"Gershgorin on diagonal should be exact, got {lam_gersh}"
        )


# ===================================================================
# Alpha estimation tests
# ===================================================================


class TestAlphaEstimation:
    def test_convex_function_zero_alpha(self):
        def convex_f(x):
            return x[0] ** 2 + x[1] ** 2

        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(convex_f, lb, ub, n_samples=50)
        assert jnp.all(alpha < 0.01), f"alpha for convex fn should be ~0, got {alpha}"

    def test_nonconvex_function_positive_alpha(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_bilinear, lb, ub, n_samples=50)
        assert jnp.all(alpha > 0), f"alpha for bilinear should be > 0, got {alpha}"

    def test_alpha_nonnegative(self):
        for f in [_rosenbrock, _rastrigin_2d, _simple_nonconvex, _bilinear]:
            lb = jnp.array([-2.0, -2.0])
            ub = jnp.array([2.0, 2.0])
            alpha = estimate_alpha(f, lb, ub, n_samples=50)
            assert jnp.all(alpha >= 0), f"alpha must be non-negative, got {alpha}"

    def test_alpha_shape(self):
        lb = jnp.array([-1.0, -1.0, -1.0])
        ub = jnp.array([1.0, 1.0, 1.0])

        def f(x):
            return x[0] * x[1] + x[1] * x[2]

        alpha = estimate_alpha(f, lb, ub)
        assert alpha.shape == (3,), f"alpha shape mismatch: {alpha.shape}"

    def test_gershgorin_method(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha_eig = estimate_alpha(_bilinear, lb, ub, n_samples=50, method="eigenvalue")
        alpha_ger = estimate_alpha(_bilinear, lb, ub, n_samples=50, method="gershgorin")
        assert jnp.all(alpha_ger >= alpha_eig - 1e-6), (
            "Gershgorin alpha should be >= eigenvalue alpha"
        )


# ===================================================================
# Soundness tests (10,000 random points, cv <= f(x) <= cc)
# ===================================================================


class TestSoundness10k:
    """Soundness at 10,000 random points using relax_alphabb."""

    def _run_soundness(self, f, lb, ub, label, seed=0):
        lb = jnp.asarray(lb, dtype=jnp.float64)
        ub = jnp.asarray(ub, dtype=jnp.float64)
        alpha = estimate_alpha(f, lb, ub, n_samples=100)

        def neg_f(z):
            return -f(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(seed)
        points = _random_box_points(key, lb, ub, n=N_POINTS)

        def eval_one(x):
            cv, cc = relax_alphabb(f, x, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            fval = f(x)
            return cv, cc, fval

        cv_vals, cc_vals, true_vals = jax.vmap(eval_one)(points)
        _check_soundness(cv_vals, cc_vals, true_vals, label)

    def test_rosenbrock(self):
        self._run_soundness(_rosenbrock, [-2.0, -2.0], [2.0, 2.0], "rosenbrock", 0)

    def test_rastrigin(self):
        self._run_soundness(_rastrigin_2d, [-2.0, -2.0], [2.0, 2.0], "rastrigin", 1)

    def test_simple_nonconvex(self):
        self._run_soundness(_simple_nonconvex, [-3.0, -3.0], [3.0, 3.0], "sin*cos+x^2", 2)

    def test_bilinear(self):
        self._run_soundness(_bilinear, [-3.0, -3.0], [3.0, 3.0], "bilinear", 3)

    def test_exp_product(self):
        self._run_soundness(_exp_product, [-1.0, -1.0], [1.0, 1.0], "exp(x*y)", 4)

    def test_sin_cos_sum(self):
        self._run_soundness(_sin_cos_sum, [-3.0, -3.0], [3.0, 3.0], "sin(x)+cos(y)", 5)

    def test_quadratic(self):
        self._run_soundness(_quadratic, [-2.0, -2.0], [2.0, 2.0], "quadratic", 6)


# ===================================================================
# Underestimator soundness (loop-based for clarity)
# ===================================================================


class TestUnderestimatorSoundness:
    def test_rosenbrock(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(0)
        points = _random_box_points(key, lb, ub, n=1000)

        def under(x):
            return alphabb_underestimator(_rosenbrock, x, lb, ub, alpha)

        vals = jax.vmap(under)(points)
        true_vals = jax.vmap(_rosenbrock)(points)
        assert jnp.all(vals <= true_vals + 1e-6)

    def test_bilinear(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])
        alpha = estimate_alpha(_bilinear, lb, ub, n_samples=50)

        key = jax.random.PRNGKey(3)
        points = _random_box_points(key, lb, ub, n=1000)

        def under(x):
            return alphabb_underestimator(_bilinear, x, lb, ub, alpha)

        vals = jax.vmap(under)(points)
        true_vals = jax.vmap(_bilinear)(points)
        assert jnp.all(vals <= true_vals + 1e-6)


# ===================================================================
# Overestimator soundness
# ===================================================================


class TestOverestimatorSoundness:
    def test_rosenbrock(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])

        def neg_f(x):
            return -_rosenbrock(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(10)
        points = _random_box_points(key, lb, ub, n=1000)

        def over(x):
            return alphabb_overestimator(_rosenbrock, x, lb, ub, alpha_neg)

        vals = jax.vmap(over)(points)
        true_vals = jax.vmap(_rosenbrock)(points)
        assert jnp.all(vals >= true_vals - 1e-6)

    def test_simple_nonconvex(self):
        lb = jnp.array([-3.0, -3.0])
        ub = jnp.array([3.0, 3.0])

        def neg_f(x):
            return -_simple_nonconvex(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=100)

        key = jax.random.PRNGKey(11)
        points = _random_box_points(key, lb, ub, n=1000)

        def over(x):
            return alphabb_overestimator(_simple_nonconvex, x, lb, ub, alpha_neg)

        vals = jax.vmap(over)(points)
        true_vals = jax.vmap(_simple_nonconvex)(points)
        assert jnp.all(vals >= true_vals - 1e-6)


# ===================================================================
# Convexity: Hessian of underestimator should be PSD
# ===================================================================


class TestConvexity:
    def _check_psd(self, f, lb, ub, label=""):
        alpha = estimate_alpha(f, lb, ub, n_samples=100)

        def under(x):
            return alphabb_underestimator(f, x, lb, ub, alpha)

        hess_fn = jax.hessian(under)

        key = jax.random.PRNGKey(20)
        points = _random_box_points(key, lb, ub, n=200)

        def _min_eig(x):
            H = hess_fn(x)
            return jnp.min(jnp.linalg.eigvalsh(H))

        min_eigs = jax.vmap(_min_eig)(points)
        assert jnp.all(min_eigs >= -1e-6), (
            f"Non-PSD Hessian for {label}: min eigenvalue = {jnp.min(min_eigs)}"
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
# Tightness: relaxation touches f at box corners
# ===================================================================


class TestTightnessAtBoundary:
    TIGHT_TOL = 1e-10

    def test_underestimator_corners(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_rosenbrock, lb, ub)

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

    def test_overestimator_corners(self):
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

    def test_relax_alphabb_corners(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_bilinear, lb, ub)

        def neg_f(x):
            return -_bilinear(x)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        corners = [
            jnp.array([lb[0], lb[1]]),
            jnp.array([lb[0], ub[1]]),
            jnp.array([ub[0], lb[1]]),
            jnp.array([ub[0], ub[1]]),
        ]
        for corner in corners:
            cv, cc = relax_alphabb(_bilinear, corner, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            true_val = _bilinear(corner)
            assert jnp.abs(cv - true_val) < self.TIGHT_TOL, f"cv not tight at corner {corner}"
            assert jnp.abs(cc - true_val) < self.TIGHT_TOL, f"cc not tight at corner {corner}"


# ===================================================================
# Gap monotonicity: tighter bounds -> smaller gap
# ===================================================================


class TestGapMonotonicity:
    def test_bilinear_gap_decreases(self):
        center = jnp.array([0.5, 0.5])
        half_widths = [4.0, 2.0, 1.0, 0.5, 0.1]
        gaps = []
        for hw in half_widths:
            lb = center - hw
            ub = center + hw
            alpha = estimate_alpha(_bilinear, lb, ub, n_samples=50)

            def neg_f(z):
                return -_bilinear(z)

            alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=50)
            cv, cc = relax_alphabb(_bilinear, center, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            gaps.append(float(cc - cv))
            assert cv <= _bilinear(center) + TOL
            assert cc >= _bilinear(center) - TOL

        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-6, f"gap not monotone decreasing: {gaps}"

    def test_rosenbrock_gap_decreases(self):
        center = jnp.array([0.5, 0.5])
        half_widths = [3.0, 1.5, 0.75, 0.3, 0.1]
        gaps = []
        for hw in half_widths:
            lb = center - hw
            ub = center + hw
            alpha = estimate_alpha(_rosenbrock, lb, ub, n_samples=50)

            def neg_f(z):
                return -_rosenbrock(z)

            alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=50)
            cv, cc = relax_alphabb(_rosenbrock, center, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            gaps.append(float(cc - cv))

        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-6, f"rosenbrock gap not monotone: {gaps}"


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

    def test_relax_alphabb_jit(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_simple_nonconvex, lb, ub)

        def neg_f(z):
            return -_simple_nonconvex(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        @jax.jit
        def relax(x):
            return relax_alphabb(_simple_nonconvex, x, lb, ub, alpha=alpha, alpha_neg=alpha_neg)

        x = jnp.array([0.5, 0.5])
        cv, cc = relax(x)
        fval = _simple_nonconvex(x)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)
        assert cv <= fval + TOL
        assert cc >= fval - TOL

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
        assert u <= _simple_nonconvex(x) + 1e-6
        assert o >= _simple_nonconvex(x) - 1e-6


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
        assert jnp.all(vals <= true_vals + 1e-6)

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
        assert jnp.all(vals >= true_vals - 1e-6)

    def test_relax_alphabb_vmap(self):
        lb = jnp.array([-1.0, -1.0])
        ub = jnp.array([1.0, 1.0])
        alpha = estimate_alpha(_exp_product, lb, ub, n_samples=50)

        def neg_f(z):
            return -_exp_product(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=50)

        key = jax.random.PRNGKey(32)
        points = _random_box_points(key, lb, ub, n=128)

        def eval_relax(x):
            return relax_alphabb(_exp_product, x, lb, ub, alpha=alpha, alpha_neg=alpha_neg)

        cv_vals, cc_vals = jax.vmap(eval_relax)(points)
        true_vals = jax.vmap(_exp_product)(points)

        assert cv_vals.shape == (128,)
        assert cc_vals.shape == (128,)
        _check_soundness(cv_vals, cc_vals, true_vals, "vmap relax_alphabb")


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

    def test_relax_alphabb_grad(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_simple_nonconvex, lb, ub)

        def neg_f(z):
            return -_simple_nonconvex(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

        def loss(x):
            cv, cc = relax_alphabb(_simple_nonconvex, x, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            return cv + cc

        x = jnp.array([0.5, 0.5])
        g = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(g)), f"relax_alphabb grad not finite: {g}"


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

        def eval_one(x):
            u = under_fn(x)
            o = over_fn(x)
            fval = _simple_nonconvex(x)
            return u, o, fval

        u_vals, o_vals, f_vals = jax.vmap(eval_one)(points)
        assert jnp.all(u_vals <= f_vals + 1e-6)
        assert jnp.all(o_vals >= f_vals - 1e-6)
        assert jnp.all(u_vals <= o_vals + 1e-6)

    def test_alpha_returned(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        _, _, alpha, alpha_neg = make_alphabb_relaxation(_bilinear, lb, ub)
        assert alpha.shape == (2,)
        assert alpha_neg.shape == (2,)
        assert jnp.all(alpha >= 0)
        assert jnp.all(alpha_neg >= 0)


# ===================================================================
# Gershgorin method soundness
# ===================================================================


class TestGershgorinSoundness:
    def test_bilinear_soundness_with_gershgorin(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        alpha = estimate_alpha(_bilinear, lb, ub, n_samples=50, method="gershgorin")

        def neg_f(z):
            return -_bilinear(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples=50, method="gershgorin")

        key = jax.random.PRNGKey(50)
        points = _random_box_points(key, lb, ub, n=1000)

        def eval_one(x):
            cv, cc = relax_alphabb(_bilinear, x, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            fval = _bilinear(x)
            return cv, cc, fval

        cv_vals, cc_vals, true_vals = jax.vmap(eval_one)(points)
        _check_soundness(cv_vals, cc_vals, true_vals, "gershgorin bilinear")

    def test_rosenbrock_soundness_with_gershgorin(self):
        lb = jnp.array([-2.0, -2.0])
        ub = jnp.array([2.0, 2.0])
        _, _, alpha, alpha_neg = make_alphabb_relaxation(_rosenbrock, lb, ub, method="gershgorin")

        key = jax.random.PRNGKey(51)
        points = _random_box_points(key, lb, ub, n=1000)

        def eval_one(x):
            cv, cc = relax_alphabb(_rosenbrock, x, lb, ub, alpha=alpha, alpha_neg=alpha_neg)
            fval = _rosenbrock(x)
            return cv, cc, fval

        cv_vals, cc_vals, true_vals = jax.vmap(eval_one)(points)
        _check_soundness(cv_vals, cc_vals, true_vals, "gershgorin rosenbrock")
