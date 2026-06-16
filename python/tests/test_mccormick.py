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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest
from discopt._jax.mccormick import (
    relax_abs,
    relax_add,
    relax_bilinear,
    relax_cos,
    relax_div,
    relax_entropy,
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


class TestBilinearSoundness:
    @pytest.mark.parametrize(
        "seed, x_lb, x_ub, y_lb, y_ub, label",
        [
            (0, 1.0, 5.0, 2.0, 7.0, "pos"),
            (1, -3.0, 4.0, -2.0, 5.0, "mixed"),
            (2, -5.0, -1.0, -7.0, -2.0, "neg"),
        ],
    )
    def test_soundness(self, seed, x_lb, x_ub, y_lb, y_ub, label):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x * y, f"bilinear {label}")


class TestUnarySoundness:
    @pytest.mark.parametrize(
        "relax_fn, true_fn, seed, lb, ub, label",
        [
            (relax_exp, jnp.exp, 10, 0.0, 3.0, "exp [0,3]"),
            (relax_exp, jnp.exp, 12, -3.0, 3.0, "exp [-3,3]"),
            (relax_square, lambda x: x**2, 21, -3.0, 2.0, "sq"),
            (relax_sqrt, jnp.sqrt, 30, 0.1, 10.0, "sqrt"),
            (relax_log, jnp.log, 40, 0.1, 10.0, "log"),
            (relax_log2, jnp.log2, 41, 0.5, 8.0, "log2"),
            (relax_log10, jnp.log10, 42, 0.1, 100.0, "log10"),
            (relax_sin, jnp.sin, 52, -1.0, 2.0, "sin mixed"),
            (relax_sin, jnp.sin, 53, -4.0, 4.0, "sin wide"),
            (
                relax_sin,
                jnp.sin,
                54,
                0.0,
                2 * 3.141592653589793 + 1.0,
                "sin full",
            ),
            (relax_cos, jnp.cos, 60, 0.5, 1.5, "cos narrow"),
            (relax_cos, jnp.cos, 62, -5.0, 5.0, "cos wide"),
            (relax_tan, jnp.tan, 70, -1.0, 1.0, "tan"),
            (relax_abs, jnp.abs, 81, -3.0, 5.0, "abs mixed"),
            (relax_abs, jnp.abs, 82, -5.0, -1.0, "abs neg"),
            (relax_sign, jnp.sign, 112, -3.0, 3.0, "sign"),
        ],
        ids=lambda val: val if isinstance(val, str) else "",
    )
    def test_soundness(self, relax_fn, true_fn, seed, lb, ub, label):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        cv, cc = relax_fn(x, lb, ub)
        _check_soundness(cv, cc, true_fn(x), label)


class TestPowSoundness:
    @pytest.mark.parametrize(
        "seed, lb, ub, power, label",
        [
            (91, -2.0, 3.0, 2, "x^2 [-2,3]"),
            (92, -2.0, 2.0, 4, "x^4 [-2,2]"),
            (93, 0.5, 3.0, 3, "x^3 [0.5,3]"),
            (95, -2.0, 3.0, 3, "x^3 [-2,3]"),
            (96, -5.0, 5.0, 1, "x^1"),
        ],
    )
    def test_soundness(self, seed, lb, ub, power, label):
        key = jax.random.PRNGKey(seed)
        x = _random_points(key, lb, ub)
        cv, cc = relax_pow(x, lb, ub, power)
        _check_soundness(cv, cc, x**power, label)


class TestDivSoundness:
    @pytest.mark.parametrize(
        "seed, x_lb, x_ub, y_lb, y_ub, label",
        [
            (100, 1.0, 5.0, 1.0, 3.0, "div pos/pos"),
            (101, 1.0, 5.0, -3.0, -1.0, "div pos/neg"),
        ],
    )
    def test_soundness(self, seed, x_lb, x_ub, y_lb, y_ub, label):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, x_lb, x_ub)
        y = _random_points(k2, y_lb, y_ub)
        cv, cc = relax_div(x, y, x_lb, x_ub, y_lb, y_ub)
        _check_soundness(cv, cc, x / y, label)


class TestAddSubNeg:
    def test_add_soundness(self):
        key = jax.random.PRNGKey(120)
        k1, k2 = jax.random.split(key)
        x = _random_points(k1, -5.0, 5.0)
        y = _random_points(k2, -3.0, 7.0)
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


TIGHT_TOL = 1e-10


class TestTightnessAtBounds:
    """When x = lb or x = ub, relaxations should equal the true value."""

    @pytest.mark.parametrize(
        "relax_fn, true_fn, lb, ub, label",
        [
            (relax_exp, jnp.exp, -2.0, 3.0, "exp"),
            (relax_square, lambda x: x**2, -2.0, 3.0, "square"),
            (relax_sqrt, jnp.sqrt, 0.5, 4.0, "sqrt"),
            (relax_log, jnp.log, 0.5, 4.0, "log"),
            (relax_sin, jnp.sin, 0.5, 1.5, "sin"),
            (relax_cos, jnp.cos, 0.5, 1.5, "cos"),
        ],
    )
    def test_tight(self, relax_fn, true_fn, lb, ub, label):
        for x in [lb, ub]:
            x = jnp.float64(x)
            cv, cc = relax_fn(x, lb, ub)
            fval = true_fn(x)
            assert jnp.abs(cv - fval) < TIGHT_TOL, f"{label} cv not tight at x={x}"
            assert jnp.abs(cc - fval) < TIGHT_TOL, f"{label} cc not tight at x={x}"

    def test_bilinear_tight(self):
        x_lb, x_ub = 1.0, 3.0
        y_lb, y_ub = 2.0, 5.0
        corners = [
            (x_lb, y_lb),
            (x_lb, y_ub),
            (x_ub, y_lb),
            (x_ub, y_ub),
        ]
        for xv, yv in corners:
            xv, yv = jnp.float64(xv), jnp.float64(yv)
            cv, cc = relax_bilinear(xv, yv, x_lb, x_ub, y_lb, y_ub)
            fval = xv * yv
            assert jnp.abs(cv - fval) < TIGHT_TOL, f"bilinear cv not tight at x={xv}, y={yv}"
            assert jnp.abs(cc - fval) < TIGHT_TOL, f"bilinear cc not tight at x={xv}, y={yv}"


# ===================================================================
# Degenerate bounds (lb == ub)
# ===================================================================


DEGEN_TOL = 1e-8


class TestDegenerateBounds:
    """When lb == ub, both relaxations should approximate f(x)."""

    @pytest.mark.parametrize(
        "relax_fn, true_fn, x_val, label",
        [
            (relax_exp, jnp.exp, 1.5, "exp"),
            (relax_log, jnp.log, 2.0, "log"),
            (relax_sin, jnp.sin, 1.0, "sin"),
        ],
    )
    def test_degenerate(self, relax_fn, true_fn, x_val, label):
        x = jnp.float64(x_val)
        lb = ub = x
        cv, cc = relax_fn(x, lb, ub)
        fval = true_fn(x)
        assert jnp.abs(cv - fval) < DEGEN_TOL, f"{label} cv degenerate fail"
        assert jnp.abs(cc - fval) < DEGEN_TOL, f"{label} cc degenerate fail"


# ===================================================================
# Gap monotonicity
# ===================================================================


class TestGapMonotonicity:
    """As bounds tighten, relaxation gap should decrease."""

    def _gap_at_midpoint(self, relax_fn, f, center, half_widths):
        gaps = []
        x = jnp.float64(center)
        for hw in half_widths:
            lb = center - hw
            ub = center + hw
            cv, cc = relax_fn(x, lb, ub)
            fval = f(x)
            assert cv <= fval + 1e-10
            assert cc >= fval - 1e-10
            gaps.append(float(cc - cv))
        return gaps

    @pytest.mark.parametrize(
        "relax_fn, true_fn, center, half_widths, label",
        [
            (
                relax_exp,
                jnp.exp,
                1.0,
                [4.0, 2.0, 1.0, 0.5, 0.1],
                "exp",
            ),
            (
                relax_log,
                jnp.log,
                4.0,
                [3.0, 1.5, 0.75, 0.3, 0.1],
                "log",
            ),
            (
                relax_sqrt,
                jnp.sqrt,
                4.0,
                [3.0, 1.5, 0.75, 0.3, 0.1],
                "sqrt",
            ),
            (
                relax_square,
                lambda x: x**2,
                2.0,
                [4.0, 2.0, 1.0, 0.5, 0.1],
                "square",
            ),
        ],
    )
    def test_gap_decreases(self, relax_fn, true_fn, center, half_widths, label):
        gaps = self._gap_at_midpoint(relax_fn, true_fn, center, half_widths)
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] <= gaps[i] + 1e-10, f"{label} gap not monotone: {gaps}"


# ===================================================================
# JIT compatibility
# ===================================================================


class TestJITCompatibility:
    """All relaxation functions should work under jax.jit."""

    @pytest.mark.parametrize(
        "relax_fn, args",
        [
            (relax_exp, (jnp.float64(1.0), 0.0, 2.0)),
            (relax_sqrt, (jnp.float64(1.0), 0.1, 2.0)),
            (relax_sin, (jnp.float64(1.0), 0.0, 2.0)),
            (relax_abs, (jnp.float64(1.0), -2.0, 3.0)),
            (relax_sign, (jnp.float64(1.0), -2.0, 3.0)),
        ],
    )
    def test_unary_jit(self, relax_fn, args):
        f = jax.jit(relax_fn)
        cv, cc = f(*args)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_bilinear_jit(self):
        f = jax.jit(relax_bilinear)
        cv, cc = f(jnp.float64(1.0), jnp.float64(2.0), 0.0, 3.0, 1.0, 4.0)
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_pow_jit(self):
        f = jax.jit(lambda x, lb, ub: relax_pow(x, lb, ub, 3))
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
    """Relaxation functions should work under jax.vmap."""

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

    def test_pow_vmap(self):
        key = jax.random.PRNGKey(204)
        batch = 64
        x = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-2.0, maxval=3.0)
        lbs = jnp.full(batch, -2.0)
        ubs = jnp.full(batch, 3.0)
        cv, cc = jax.vmap(lambda xi, lbi, ubi: relax_pow(xi, lbi, ubi, 3))(x, lbs, ubs)
        assert cv.shape == (batch,)
        _check_soundness(cv, cc, x**3, "vmap pow")

    def test_vmap_different_bounds(self):
        """Test that vmap works with varying bounds per element."""
        key = jax.random.PRNGKey(205)
        batch = 64
        lbs = jax.random.uniform(key, (batch,), dtype=jnp.float64, minval=-3.0, maxval=-0.5)
        ubs = lbs + jax.random.uniform(
            jax.random.PRNGKey(206),
            (batch,),
            dtype=jnp.float64,
            minval=0.5,
            maxval=3.0,
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

    @pytest.mark.parametrize(
        "relax_fn, lb, ub, x_val, label",
        [
            (relax_exp, -2.0, 2.0, 0.5, "exp"),
            (relax_sqrt, 0.1, 5.0, 1.0, "sqrt"),
            (relax_sin, -1.0, 2.0, 0.5, "sin"),
            (relax_tan, -1.0, 1.0, 0.3, "tan"),
            (relax_abs, -3.0, 3.0, 1.0, "abs"),
        ],
    )
    def test_unary_grad(self, relax_fn, lb, ub, x_val, label):
        def loss(x):
            cv, cc = relax_fn(x, lb, ub)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(x_val))
        assert jnp.isfinite(g), f"{label} grad is not finite: {g}"

    def test_bilinear_grad(self):
        def loss(x):
            cv, cc = relax_bilinear(x, x + 1, 0.0, 3.0, 1.0, 4.0)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"bilinear grad is not finite: {g}"

    def test_pow_grad(self):
        def loss(x):
            cv, cc = relax_pow(x, -2.0, 3.0, 3)
            return cv + cc

        g = jax.grad(loss)(jnp.float64(1.0))
        assert jnp.isfinite(g), f"pow grad is not finite: {g}"


class TestEntropySoundness:
    """relax_entropy: entropy(x) = x*log(x), convex on x > 0."""

    @pytest.mark.parametrize(
        "seed, lb, ub, label",
        [
            (70, 0.01, 1.0, "entropy (0,1]"),
            (71, 0.5, 5.0, "entropy [0.5,5]"),
            (72, 1.0, 100.0, "entropy [1,100]"),
        ],
    )
    def test_soundness(self, seed, lb, ub, label):
        x = _random_points(jax.random.PRNGKey(seed), lb, ub)
        cv, cc = relax_entropy(x, lb, ub)
        _check_soundness(cv, cc, x * jnp.log(x), label)

    def test_convex_underestimator_is_tight(self):
        # For a convex function the underestimator equals f itself.
        x = _random_points(jax.random.PRNGKey(73), 0.1, 4.0)
        cv, _ = relax_entropy(x, 0.1, 4.0)
        assert jnp.allclose(cv, x * jnp.log(x), atol=1e-10)
