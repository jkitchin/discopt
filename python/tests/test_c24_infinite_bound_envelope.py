"""C-24: McCormick secant envelopes must never return NaN on infinite bounds.

`_secant` computes ``slope = (f(ub) - f(lb)) / (ub - lb)``. When either bound is
non-finite this is ``inf/inf`` / ``inf - inf`` = ``NaN``, and the NaN propagates
into ``cc`` (for convex relaxations) or ``cv`` (for concave relaxations) and into
``relax_bilinear``. A NaN is not a valid over/under-estimator: it silently defeats
downstream soundness comparisons (``cv <= f``, ``f <= cc``) which are all False for
NaN, and any consumer that uses the value unguarded gets an unsafe bound.

The sound behavior on a half-/fully-infinite box is a *no-information* envelope:
the concave overestimator is ``+inf`` and the convex underestimator is ``-inf``.
These tests pin that: on infinite bounds the primitives must return an envelope
that (a) is never NaN and (b) still brackets the function ``cv <= f <= cc``.

All calls hit the relaxation primitives directly, so this is a sub-second smoke
test suitable for every-PR CI.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax import mccormick as mc


def _finite_and_brackets(cv, cc, fx):
    """cv, cc must be non-NaN and bracket fx (allowing +/-inf as no-info)."""
    cv = float(cv)
    cc = float(cc)
    fx = float(fx)
    assert not np.isnan(cv), f"cv is NaN (fx={fx})"
    assert not np.isnan(cc), f"cc is NaN (fx={fx})"
    # Bracketing: cv <= f <= cc, with +/-inf permitted as no-information.
    assert cv <= fx + 1e-9, f"cv={cv} > f={fx}"
    assert cc >= fx - 1e-9, f"cc={cc} < f={fx}"


@pytest.mark.smoke
def test_secant_infinite_bounds_no_nan():
    """`_secant` on an infinite bound returns the requested no-info fallback,
    never NaN."""

    def f(t):
        return t**2

    # Overestimator role -> fallback +inf; underestimator role -> fallback -inf.
    for lb, ub in [(-2.0, np.inf), (-np.inf, 2.0), (-np.inf, np.inf), (1.0, np.inf)]:
        cc = mc._secant(f, jnp.array(0.0), jnp.array(lb), jnp.array(ub), fallback=jnp.inf)
        cv = mc._secant(f, jnp.array(0.0), jnp.array(lb), jnp.array(ub), fallback=-jnp.inf)
        assert not np.isnan(float(cc)), f"cc NaN on [{lb},{ub}]"
        assert not np.isnan(float(cv)), f"cv NaN on [{lb},{ub}]"
        assert float(cc) == np.inf
        assert float(cv) == -np.inf

    # Finite bounds must be unaffected (regression guard on the sound path).
    cc = mc._secant(f, jnp.array(0.0), jnp.array(-2.0), jnp.array(2.0), fallback=jnp.inf)
    assert abs(float(cc) - 4.0) < 1e-9  # secant of x^2 over [-2,2] at 0 is 4


@pytest.mark.smoke
def test_convex_relaxations_infinite_bounds_no_nan():
    """Convex univariate relaxations (cc = secant) must not emit a NaN cc when a
    bound is infinite; cc must be a valid overestimator (+inf ok)."""
    cases = [
        (mc.relax_square, lambda x: x**2, 0.0),
        (mc.relax_exp, lambda x: np.exp(x), 0.0),
        (mc.relax_cosh, lambda x: np.cosh(x), 0.0),
    ]
    for relax_fn, f, x in cases:
        for lb, ub in [(-2.0, np.inf), (-np.inf, 2.0), (-np.inf, np.inf)]:
            cv, cc = relax_fn(jnp.array(x), jnp.array(lb), jnp.array(ub))
            _finite_and_brackets(cv, cc, f(x))


@pytest.mark.smoke
def test_concave_relaxations_infinite_bounds_no_nan():
    """Concave univariate relaxations (cv = secant) must not emit a NaN cv when
    the upper bound is infinite; cv must be a valid underestimator (-inf ok)."""
    # sqrt/log require lb finite & > 0; ub can be +inf.
    cases = [
        (mc.relax_sqrt, lambda x: np.sqrt(x), 1.0, 0.0),
        (mc.relax_log, lambda x: np.log(x), 1.0, 0.5),
    ]
    for relax_fn, f, x, lb in cases:
        cv, cc = relax_fn(jnp.array(x), jnp.array(lb), jnp.array(np.inf))
        _finite_and_brackets(cv, cc, f(x))


@pytest.mark.smoke
def test_bilinear_infinite_bounds_no_nan():
    """`relax_bilinear` must not emit NaN cv/cc when a factor bound is infinite.
    The envelopes must still bracket the product at an interior point."""
    x, y = 1.0, 1.0
    for x_lb, x_ub, y_lb, y_ub in [
        (-2.0, np.inf, -2.0, 2.0),
        (-np.inf, 2.0, -2.0, 2.0),
        (-2.0, 2.0, -np.inf, np.inf),
        (0.0, np.inf, 0.0, np.inf),
    ]:
        cv, cc = mc.relax_bilinear(
            jnp.array(x),
            jnp.array(y),
            jnp.array(x_lb),
            jnp.array(x_ub),
            jnp.array(y_lb),
            jnp.array(y_ub),
        )
        _finite_and_brackets(cv, cc, x * y)


@pytest.mark.smoke
def test_pow_odd_infinite_bounds_no_nan():
    """Odd-power relaxation (piecewise secants) must not emit NaN on inf bounds."""
    n = 3
    f = lambda t: t**n  # noqa: E731
    for lb, ub, x in [(-2.0, np.inf, 0.5), (-np.inf, 2.0, -0.5), (-np.inf, np.inf, 0.0)]:
        cv, cc = mc.relax_pow(jnp.array(x), jnp.array(lb), jnp.array(ub), n)
        _finite_and_brackets(cv, cc, f(x))
