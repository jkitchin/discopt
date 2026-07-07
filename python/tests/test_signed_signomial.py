"""Soundness and curvature tests for the signed-signomial DC relaxation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.symbolic.signed_signomial import signed_signomial_dc_envelope

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.relaxation


def _signomial_value(u, terms):
    """Direct evaluation of s(u) = sum_k sigma_k * exp(log_c_k + a_k . u)."""
    u = jnp.asarray(u)
    s = jnp.asarray(0.0)
    for sigma, log_c, exps in terms:
        s = s + sigma * jnp.exp(log_c + jnp.dot(jnp.asarray(exps), u))
    return s


def _random_terms(rng, n, n_terms):
    """Random signed signomial: c_k > 0, a in [-2, 2], mixed +/- signs."""
    terms = []
    for k in range(n_terms):
        # Force at least one of each sign for the first two terms.
        if k == 0:
            sigma = 1.0
        elif k == 1:
            sigma = -1.0
        else:
            sigma = float(rng.choice([-1.0, 1.0]))
        c = float(rng.uniform(0.2, 3.0))
        exps = rng.uniform(-2.0, 2.0, size=n)
        terms.append((sigma, float(np.log(c)), exps))
    return terms


def _box(n):
    u_lb = np.full(n, np.log(0.3))
    u_ub = np.full(n, np.log(4.0))
    return jnp.asarray(u_lb), jnp.asarray(u_ub)


@pytest.mark.slow
def test_soundness_cv_below_cc_above():
    """cv(u) <= s(u) <= cc(u) for many random signed signomials and samples."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 3):
        for _ in range(10):
            n_terms = int(rng.integers(2, 6))
            terms = _random_terms(rng, n, n_terms)
            u_lb, u_ub = _box(n)
            for _ in range(50):
                t = rng.uniform(size=n)
                u = jnp.asarray(np.asarray(u_lb) + t * np.asarray(u_ub - u_lb))
                cv, cc = signed_signomial_dc_envelope(u, terms, u_lb, u_ub)
                s = _signomial_value(u, terms)
                assert float(cv) <= float(s) + 1e-9
                assert float(cc) >= float(s) - 1e-9


@pytest.mark.slow
def test_curvature_cv_convex_cc_concave():
    """Midpoint Jensen check: cv convex, cc concave along random chords."""
    rng = np.random.default_rng(1)
    for n in (1, 2, 3):
        for _ in range(10):
            n_terms = int(rng.integers(2, 6))
            terms = _random_terms(rng, n, n_terms)
            u_lb, u_ub = _box(n)
            lb = np.asarray(u_lb)
            span = np.asarray(u_ub - u_lb)
            for _ in range(20):
                ua = jnp.asarray(lb + rng.uniform(size=n) * span)
                ub = jnp.asarray(lb + rng.uniform(size=n) * span)
                umid = 0.5 * (ua + ub)
                cv_a, cc_a = signed_signomial_dc_envelope(ua, terms, u_lb, u_ub)
                cv_b, cc_b = signed_signomial_dc_envelope(ub, terms, u_lb, u_ub)
                cv_m, cc_m = signed_signomial_dc_envelope(umid, terms, u_lb, u_ub)
                # Convex: f(mid) <= 0.5 (f(a) + f(b)).
                assert float(cv_m) <= 0.5 * (float(cv_a) + float(cv_b)) + 1e-6
                # Concave: f(mid) >= 0.5 (f(a) + f(b)).
                assert float(cc_m) >= 0.5 * (float(cc_a) + float(cc_b)) - 1e-6


def test_purely_positive_reduces():
    """With Pminus = 0: cv = Pplus = s, and cc = max_corner Pplus - 0."""
    rng = np.random.default_rng(2)
    n = 2
    terms = [
        (1.0, float(np.log(1.5)), rng.uniform(-2.0, 2.0, size=n)),
        (1.0, float(np.log(0.7)), rng.uniform(-2.0, 2.0, size=n)),
        (1.0, float(np.log(2.2)), rng.uniform(-2.0, 2.0, size=n)),
    ]
    u_lb, u_ub = _box(n)

    # Corner-max of the positive posynomial == cc constant (Pminus = 0).
    corners = [
        jnp.asarray([a, b])
        for a in (float(u_lb[0]), float(u_ub[0]))
        for b in (float(u_lb[1]), float(u_ub[1]))
    ]
    sec_plus = max(float(_signomial_value(c, terms)) for c in corners)

    lb = np.asarray(u_lb)
    span = np.asarray(u_ub - u_lb)
    for _ in range(40):
        u = jnp.asarray(lb + rng.uniform(size=n) * span)
        cv, cc = signed_signomial_dc_envelope(u, terms, u_lb, u_ub)
        s = _signomial_value(u, terms)
        # Pminus = 0 => cv is exactly the function s.
        assert float(cv) == pytest.approx(float(s), abs=1e-9)
        # cc = constant corner-max of Pplus.
        assert float(cc) == pytest.approx(sec_plus, abs=1e-9)
