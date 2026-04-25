"""Soundness tests for interval forward-mode AD (Hessian).

The critical property: at every sample point in the box, the pointwise
Hessian ``H(x)`` computed by JAX autodiff must lie inside the interval
Hessian enclosure returned by ``interval_hessian``. A violation
anywhere is a sound-certificate failure — the downstream PSD test
would then produce a verdict that isn't actually a proof.

References
----------
Griewank, Walther (2008), *Evaluating Derivatives*, §3.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.convexity.interval_ad import IntervalAD, interval_hessian
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import Model

SAMPLES = 12
TOL = 1e-8


def _flat_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
        ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
    return np.concatenate(lbs), np.concatenate(ubs)


def _sample(model: Model, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lb, ub = _flat_bounds(model)
    return rng.uniform(lb, ub, size=(n, lb.size))


def _assert_hess_contains_pointwise(expr, model, *, seed: int = 0):
    """Numerical Hessian at sample points must lie inside the interval."""
    ad: IntervalAD = interval_hessian(expr, model)
    f = compile_expression(expr, model)
    hess_fn = jax.jit(jax.hessian(f))
    lo = np.asarray(ad.hess.lo)
    hi = np.asarray(ad.hess.hi)
    xs = _sample(model, SAMPLES, seed=seed)
    for x in xs:
        H = np.asarray(hess_fn(jnp.asarray(x)))
        # Scalar output means Hessian is (n, n). Degenerate 1-var
        # case yields a 0-d array; expand.
        if H.ndim == 0:
            H = H.reshape(1, 1)
        assert np.all(lo <= H + TOL), (
            f"interval Hessian lo exceeds pointwise value at x={x}:\nlo=\n{lo}\npointwise=\n{H}"
        )
        assert np.all(H <= hi + TOL), (
            f"pointwise Hessian exceeds interval hi at x={x}:\nhi=\n{hi}\npointwise=\n{H}"
        )


def _assert_grad_contains_pointwise(expr, model, *, seed: int = 0):
    ad: IntervalAD = interval_hessian(expr, model)
    f = compile_expression(expr, model)
    grad_fn = jax.jit(jax.grad(f))
    lo = np.asarray(ad.grad.lo)
    hi = np.asarray(ad.grad.hi)
    xs = _sample(model, SAMPLES, seed=seed)
    for x in xs:
        g = np.asarray(grad_fn(jnp.asarray(x)))
        if g.ndim == 0:
            g = g.reshape(1)
        assert np.all(lo <= g + TOL), (
            f"interval gradient lo exceeds pointwise at x={x}:\nlo={lo}\npointwise={g}"
        )
        assert np.all(g <= hi + TOL), (
            f"pointwise gradient exceeds interval hi at x={x}:\nhi={hi}\npointwise={g}"
        )


# ──────────────────────────────────────────────────────────────────────
# Polynomials — core test coverage
# ──────────────────────────────────────────────────────────────────────


class TestPolynomialHessians:
    def test_x_squared(self):
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        _assert_grad_contains_pointwise(x**2, m)
        _assert_hess_contains_pointwise(x**2, m)

    def test_x_cubed(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.5, ub=2.0)
        _assert_hess_contains_pointwise(x**3, m)

    def test_sum_of_squares(self):
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        _assert_hess_contains_pointwise(x**2 + y**2, m)

    def test_bilinear(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=2.0)
        y = m.continuous("y", lb=-1.0, ub=2.0)
        _assert_hess_contains_pointwise(x * y, m)

    def test_cubic_polynomial(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.5)
        expr = x**3 - 2.0 * x**2 + 3.0 * x - 1.0
        _assert_hess_contains_pointwise(expr, m)


# ──────────────────────────────────────────────────────────────────────
# Exp / log / sqrt compositions
# ──────────────────────────────────────────────────────────────────────


class TestExpLogSqrtHessians:
    def test_exp_of_square(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.2, ub=1.2)
        _assert_hess_contains_pointwise(dm.exp(x**2), m)

    def test_log_of_sum(self):
        m = Model("t")
        x = m.continuous("x", lb=0.2, ub=2.0)
        y = m.continuous("y", lb=0.2, ub=2.0)
        _assert_hess_contains_pointwise(dm.log(x + y), m)

    def test_sqrt_of_positive(self):
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=5.0)
        _assert_hess_contains_pointwise(dm.sqrt(x), m)

    def test_exp_plus_log(self):
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=3.0)
        _assert_hess_contains_pointwise(dm.exp(x) + dm.log(x), m)

    def test_rational(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=3.0)
        _assert_hess_contains_pointwise(1.0 / x, m)


# ──────────────────────────────────────────────────────────────────────
# Multi-variable compositions
# ──────────────────────────────────────────────────────────────────────


class TestMultiVariableHessians:
    def test_log_of_sum_of_squares(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        _assert_hess_contains_pointwise(dm.log(x**2 + y**2 + 1.0), m)

    def test_exp_of_bilinear(self):
        m = Model("t")
        x = m.continuous("x", lb=-0.5, ub=1.0)
        y = m.continuous("y", lb=-0.5, ub=1.0)
        _assert_hess_contains_pointwise(dm.exp(x * y), m)

    def test_wide_exp_bilinear_abstention_is_quiet(self):
        """Non-finite interval Hessian entries should not emit RuntimeWarning."""
        m = Model("t")
        x = m.continuous("x", lb=-1000.0, ub=1000.0)
        y = m.continuous("y", lb=-1000.0, ub=1000.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ad = interval_hessian(dm.exp(x * y), m)

        assert np.any(~np.isfinite(np.asarray(ad.hess.lo))) or np.any(
            ~np.isfinite(np.asarray(ad.hess.hi))
        )

    def test_quadratic_form(self):
        """``x^2 + 4 x y + 3 y^2`` — classic indefinite quadratic."""
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        _assert_hess_contains_pointwise(x**2 + 4.0 * x * y + 3.0 * y**2, m)


# ──────────────────────────────────────────────────────────────────────
# Unsupported atoms produce unbounded enclosures (sound abstention)
# ──────────────────────────────────────────────────────────────────────


class TestUnsupportedFallsThrough:
    def test_abs_returns_unbounded(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        ad = interval_hessian(dm.abs(x), m)
        # At least one Hessian entry should be unbounded.
        assert np.any(~np.isfinite(np.asarray(ad.hess.lo))) or np.any(
            ~np.isfinite(np.asarray(ad.hess.hi))
        )

    def test_trig_returns_unbounded(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        ad = interval_hessian(dm.sin(x), m)
        assert np.any(~np.isfinite(np.asarray(ad.hess.lo))) or np.any(
            ~np.isfinite(np.asarray(ad.hess.hi))
        )


# ──────────────────────────────────────────────────────────────────────
# Array / indexed variables
# ──────────────────────────────────────────────────────────────────────


class TestIndexedVariables:
    def test_indexed_quadratic(self):
        m = Model("t")
        x = m.continuous("x", shape=(3,), lb=-1.0, ub=1.0)
        expr = x[0] ** 2 + x[1] * x[2]
        _assert_hess_contains_pointwise(expr, m)


# ──────────────────────────────────────────────────────────────────────
# Non-scalar variable should raise
# ──────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_array_variable_without_indexing_raises(self):
        m = Model("t")
        x = m.continuous("x", shape=(3,), lb=-1.0, ub=1.0)
        with pytest.raises(ValueError):
            # Using the full array x (without indexing) isn't supported
            # by the scalar-output Hessian walker.
            interval_hessian(dm.exp(x), m)
