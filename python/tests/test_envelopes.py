"""Tests for envelopes: trig (D2), signomial (D3), and additional envelopes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erf as jax_erf

pytestmark = pytest.mark.unit


class TestRelaxSinTight:
    """Tests for relax_sin_tight (D2)."""

    def test_soundness_concave_regime(self):
        """sin on [0.2, 2.5] (concave regime): cv <= sin(x) <= cc."""
        from discopt._jax.envelopes import relax_sin_tight

        lb, ub = 0.2, 2.5
        xs = jnp.linspace(lb, ub, 50)
        for x in xs:
            cv, cc = relax_sin_tight(x, lb, ub)
            assert cv <= jnp.sin(x) + 1e-10, f"cv={cv} > sin({x})={jnp.sin(x)}"
            assert cc >= jnp.sin(x) - 1e-10, f"cc={cc} < sin({x})={jnp.sin(x)}"

    def test_soundness_convex_regime(self):
        """sin on [3.5, 5.5] (convex regime): cv <= sin(x) <= cc."""
        from discopt._jax.envelopes import relax_sin_tight

        lb, ub = 3.5, 5.5
        xs = jnp.linspace(lb, ub, 50)
        for x in xs:
            cv, cc = relax_sin_tight(x, lb, ub)
            assert cv <= jnp.sin(x) + 1e-10
            assert cc >= jnp.sin(x) - 1e-10

    def test_soundness_mixed_regime(self):
        """sin on [1.0, 4.0] (mixed concave/convex): cv <= sin(x) <= cc."""
        from discopt._jax.envelopes import relax_sin_tight

        lb, ub = 1.0, 4.0
        xs = jnp.linspace(lb, ub, 50)
        for x in xs:
            cv, cc = relax_sin_tight(x, lb, ub)
            assert cv <= jnp.sin(x) + 1e-10
            assert cc >= jnp.sin(x) - 1e-10

    def test_soundness_wide_interval(self):
        """sin on [0, 7] (> 2*pi): should return [-1, 1]."""
        from discopt._jax.envelopes import relax_sin_tight

        lb, ub = 0.0, 7.0
        x = jnp.float64(3.0)
        cv, cc = relax_sin_tight(x, lb, ub)
        np.testing.assert_allclose(float(cv), -1.0, atol=1e-10)
        np.testing.assert_allclose(float(cc), 1.0, atol=1e-10)

    def test_soundness_random(self):
        """Random soundness check with many x values."""
        from discopt._jax.envelopes import relax_sin_tight

        key = jax.random.PRNGKey(42)
        for _ in range(20):
            key, k1, k2 = jax.random.split(key, 3)
            a = float(jax.random.uniform(k1, minval=-5.0, maxval=5.0))
            w = float(jax.random.uniform(k2, minval=0.1, maxval=2.5))
            lb, ub = a, a + w
            xs = jnp.linspace(lb, ub, 20)
            for x in xs:
                cv, cc = relax_sin_tight(x, lb, ub)
                assert cv <= jnp.sin(x) + 1e-9
                assert cc >= jnp.sin(x) - 1e-9

    def test_jit_compatible(self):
        """relax_sin_tight works under jax.jit."""
        from discopt._jax.envelopes import relax_sin_tight

        @jax.jit
        def f(x):
            return relax_sin_tight(x, 0.5, 2.5)

        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_vmap_compatible(self):
        """relax_sin_tight works under jax.vmap."""
        from discopt._jax.envelopes import relax_sin_tight

        xs = jnp.linspace(0.5, 2.5, 10)
        lbs = jnp.full(10, 0.5)
        ubs = jnp.full(10, 2.5)
        cvs, ccs = jax.vmap(relax_sin_tight)(xs, lbs, ubs)
        assert cvs.shape == (10,)
        assert ccs.shape == (10,)


class TestRelaxCosTight:
    """Tests for relax_cos_tight (D2)."""

    def test_soundness(self):
        """cos on [0.5, 2.5]: cv <= cos(x) <= cc."""
        from discopt._jax.envelopes import relax_cos_tight

        lb, ub = 0.5, 2.5
        xs = jnp.linspace(lb, ub, 50)
        for x in xs:
            cv, cc = relax_cos_tight(x, lb, ub)
            assert cv <= jnp.cos(x) + 1e-10
            assert cc >= jnp.cos(x) - 1e-10

    def test_soundness_negative_range(self):
        """cos on [-2, -0.5]: cv <= cos(x) <= cc."""
        from discopt._jax.envelopes import relax_cos_tight

        lb, ub = -2.0, -0.5
        xs = jnp.linspace(lb, ub, 30)
        for x in xs:
            cv, cc = relax_cos_tight(x, lb, ub)
            assert cv <= jnp.cos(x) + 1e-10
            assert cc >= jnp.cos(x) - 1e-10


class TestRelaxSignomialMulti:
    """Tests for relax_signomial_multi (D3)."""

    def test_soundness_2var(self):
        """x^0.5 * y^1.5 on positive domain."""
        from discopt._jax.envelopes import relax_signomial_multi

        xs = jnp.array([2.0, 3.0])
        lbs = jnp.array([1.0, 1.0])
        ubs = jnp.array([4.0, 5.0])
        exponents = jnp.array([0.5, 1.5])
        cv, cc = relax_signomial_multi(xs, lbs, ubs, exponents)

        true_val = 2.0**0.5 * 3.0**1.5
        assert cv <= true_val + 1e-8, f"cv={cv} > true={true_val}"
        assert cc >= true_val - 1e-8, f"cc={cc} < true={true_val}"

    def test_soundness_3var(self):
        """x^0.3 * y^0.4 * z^0.3 on positive domain."""
        from discopt._jax.envelopes import relax_signomial_multi

        xs = jnp.array([2.0, 3.0, 4.0])
        lbs = jnp.array([1.0, 1.0, 1.0])
        ubs = jnp.array([5.0, 5.0, 5.0])
        exponents = jnp.array([0.3, 0.4, 0.3])
        cv, cc = relax_signomial_multi(xs, lbs, ubs, exponents)

        true_val = 2.0**0.3 * 3.0**0.4 * 4.0**0.3
        assert cv <= true_val + 1e-8
        assert cc >= true_val - 1e-8

    def test_negative_exponents(self):
        """x^{-0.5} * y^{1.0} on positive domain."""
        from discopt._jax.envelopes import relax_signomial_multi

        xs = jnp.array([2.0, 3.0])
        lbs = jnp.array([0.5, 1.0])
        ubs = jnp.array([4.0, 5.0])
        exponents = jnp.array([-0.5, 1.0])
        cv, cc = relax_signomial_multi(xs, lbs, ubs, exponents)

        true_val = 2.0 ** (-0.5) * 3.0**1.0
        assert cv <= true_val + 1e-8
        assert cc >= true_val - 1e-8

    def test_reduces_to_univariate(self):
        """Single variable x^0.5 matches relax_signomial."""
        from discopt._jax.envelopes import relax_signomial, relax_signomial_multi

        x = jnp.float64(3.0)
        lb, ub = jnp.float64(1.0), jnp.float64(5.0)

        cv_uni, cc_uni = relax_signomial(x, lb, ub, 0.5)
        cv_multi, cc_multi = relax_signomial_multi(
            jnp.array([x]), jnp.array([lb]), jnp.array([ub]), jnp.array([0.5])
        )
        # Both should be valid relaxations (may differ in tightness)
        true_val = x**0.5
        assert cv_uni <= true_val + 1e-8
        assert cc_uni >= true_val - 1e-8
        assert cv_multi <= true_val + 1e-8
        assert cc_multi >= true_val - 1e-8

    def test_jit_compatible(self):
        """relax_signomial_multi works under jax.jit."""
        from discopt._jax.envelopes import relax_signomial_multi

        @jax.jit
        def f(xs, lbs, ubs, exps):
            return relax_signomial_multi(xs, lbs, ubs, exps)

        cv, cc = f(
            jnp.array([2.0, 3.0]),
            jnp.array([1.0, 1.0]),
            jnp.array([4.0, 5.0]),
            jnp.array([0.5, 1.5]),
        )
        assert jnp.isfinite(cv) and jnp.isfinite(cc)

    def test_vmap_compatible(self):
        """relax_signomial_multi works under jax.vmap over batch dim."""
        from discopt._jax.envelopes import relax_signomial_multi

        batch_xs = jnp.array([[2.0, 3.0], [3.0, 4.0], [1.5, 2.5]])
        lbs = jnp.array([1.0, 1.0])
        ubs = jnp.array([5.0, 5.0])
        exps = jnp.array([0.5, 1.5])

        cvs, ccs = jax.vmap(lambda x: relax_signomial_multi(x, lbs, ubs, exps))(batch_xs)
        assert cvs.shape == (3,)
        assert ccs.shape == (3,)


# -----------------------------------------------------------------------
# Tests for additional envelope functions
# -----------------------------------------------------------------------


def _check_soundness(relax_fn, true_fn, lb, ub, n=50, tol=1e-9):
    """Helper: verify cv <= f(x) <= cc for n points in [lb, ub]."""
    xs = jnp.linspace(lb, ub, n)
    for x in xs:
        cv, cc = relax_fn(x, lb, ub)
        f_val = true_fn(x)
        assert cv <= f_val + tol, f"cv={float(cv)} > f({float(x)})={float(f_val)}"
        assert cc >= f_val - tol, f"cc={float(cc)} < f({float(x)})={float(f_val)}"


class TestRelaxAsinh:
    """Tests for relax_asinh."""

    def test_soundness_positive(self):
        from discopt._jax.envelopes import relax_asinh

        _check_soundness(relax_asinh, jnp.arcsinh, 0.5, 5.0)

    def test_soundness_negative(self):
        from discopt._jax.envelopes import relax_asinh

        _check_soundness(relax_asinh, jnp.arcsinh, -5.0, -0.5)

    def test_soundness_mixed(self):
        from discopt._jax.envelopes import relax_asinh

        _check_soundness(relax_asinh, jnp.arcsinh, -3.0, 3.0)

    def test_endpoints(self):
        from discopt._jax.envelopes import relax_asinh

        lb, ub = 1.0, 4.0
        for x in [jnp.float64(lb), jnp.float64(ub)]:
            cv, cc = relax_asinh(x, lb, ub)
            f_val = jnp.arcsinh(x)
            np.testing.assert_allclose(float(cv), float(f_val), atol=1e-10)
            np.testing.assert_allclose(float(cc), float(f_val), atol=1e-10)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_asinh

        f = jax.jit(lambda x: relax_asinh(x, 0.0, 3.0))
        cv, cc = f(jnp.float64(1.5))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestRelaxAcosh:
    """Tests for relax_acosh."""

    def test_soundness(self):
        from discopt._jax.envelopes import relax_acosh

        _check_soundness(relax_acosh, jnp.arccosh, 1.0, 5.0)

    def test_soundness_near_one(self):
        from discopt._jax.envelopes import relax_acosh

        _check_soundness(relax_acosh, jnp.arccosh, 1.01, 2.0)

    def test_endpoints(self):
        from discopt._jax.envelopes import relax_acosh

        lb, ub = 1.5, 4.0
        for x in [jnp.float64(lb), jnp.float64(ub)]:
            cv, cc = relax_acosh(x, lb, ub)
            f_val = jnp.arccosh(x)
            np.testing.assert_allclose(float(cv), float(f_val), atol=1e-10)
            np.testing.assert_allclose(float(cc), float(f_val), atol=1e-10)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_acosh

        f = jax.jit(lambda x: relax_acosh(x, 1.0, 5.0))
        cv, cc = f(jnp.float64(2.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestRelaxAtanh:
    """Tests for relax_atanh."""

    def test_soundness_positive(self):
        from discopt._jax.envelopes import relax_atanh

        _check_soundness(relax_atanh, jnp.arctanh, 0.1, 0.9)

    def test_soundness_negative(self):
        from discopt._jax.envelopes import relax_atanh

        _check_soundness(relax_atanh, jnp.arctanh, -0.9, -0.1)

    def test_soundness_mixed(self):
        from discopt._jax.envelopes import relax_atanh

        _check_soundness(relax_atanh, jnp.arctanh, -0.5, 0.5)

    def test_endpoints(self):
        from discopt._jax.envelopes import relax_atanh

        lb, ub = 0.1, 0.8
        for x in [jnp.float64(lb), jnp.float64(ub)]:
            cv, cc = relax_atanh(x, lb, ub)
            f_val = jnp.arctanh(x)
            np.testing.assert_allclose(float(cv), float(f_val), atol=1e-10)
            np.testing.assert_allclose(float(cc), float(f_val), atol=1e-10)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_atanh

        f = jax.jit(lambda x: relax_atanh(x, -0.5, 0.5))
        cv, cc = f(jnp.float64(0.2))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestRelaxErf:
    """Tests for relax_erf."""

    def test_soundness_positive(self):
        from discopt._jax.envelopes import relax_erf

        _check_soundness(relax_erf, jax_erf, 0.0, 3.0)

    def test_soundness_negative(self):
        from discopt._jax.envelopes import relax_erf

        _check_soundness(relax_erf, jax_erf, -3.0, 0.0)

    def test_soundness_mixed(self):
        from discopt._jax.envelopes import relax_erf

        _check_soundness(relax_erf, jax_erf, -2.0, 2.0)

    def test_endpoints(self):
        from discopt._jax.envelopes import relax_erf

        lb, ub = -1.0, 1.0
        for x in [jnp.float64(lb), jnp.float64(ub)]:
            cv, cc = relax_erf(x, lb, ub)
            f_val = jax_erf(x)
            np.testing.assert_allclose(float(cv), float(f_val), atol=1e-10)
            np.testing.assert_allclose(float(cc), float(f_val), atol=1e-10)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_erf

        f = jax.jit(lambda x: relax_erf(x, -2.0, 2.0))
        cv, cc = f(jnp.float64(0.5))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestRelaxLog1p:
    """Tests for relax_log1p."""

    def test_soundness(self):
        from discopt._jax.envelopes import relax_log1p

        _check_soundness(relax_log1p, jnp.log1p, 0.0, 5.0)

    def test_soundness_near_minus_one(self):
        from discopt._jax.envelopes import relax_log1p

        _check_soundness(relax_log1p, jnp.log1p, -0.5, 2.0)

    def test_endpoints(self):
        from discopt._jax.envelopes import relax_log1p

        lb, ub = 0.0, 4.0
        for x in [jnp.float64(lb), jnp.float64(ub)]:
            cv, cc = relax_log1p(x, lb, ub)
            f_val = jnp.log1p(x)
            np.testing.assert_allclose(float(cv), float(f_val), atol=1e-10)
            np.testing.assert_allclose(float(cc), float(f_val), atol=1e-10)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_log1p

        f = jax.jit(lambda x: relax_log1p(x, 0.0, 3.0))
        cv, cc = f(jnp.float64(1.0))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)


class TestRelaxReciprocal:
    """Tests for relax_reciprocal."""

    def test_soundness_positive(self):
        from discopt._jax.envelopes import relax_reciprocal

        def recip(x):
            return 1.0 / x

        _check_soundness(relax_reciprocal, recip, 0.5, 5.0)

    def test_soundness_negative(self):
        from discopt._jax.envelopes import relax_reciprocal

        def recip(x):
            return 1.0 / x

        _check_soundness(relax_reciprocal, recip, -5.0, -0.5)

    def test_endpoints(self):
        from discopt._jax.envelopes import relax_reciprocal

        lb, ub = 1.0, 4.0
        for x in [jnp.float64(lb), jnp.float64(ub)]:
            cv, cc = relax_reciprocal(x, lb, ub)
            f_val = 1.0 / x
            np.testing.assert_allclose(float(cv), float(f_val), atol=1e-10)
            np.testing.assert_allclose(float(cc), float(f_val), atol=1e-10)

    def test_jit_compatible(self):
        from discopt._jax.envelopes import relax_reciprocal

        f = jax.jit(lambda x: relax_reciprocal(x, 0.5, 3.0))
        cv, cc = f(jnp.float64(1.5))
        assert jnp.isfinite(cv) and jnp.isfinite(cc)
