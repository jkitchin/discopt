# ruff: noqa: E402, I001
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from discopt._jax.symbolic.gp_hull import monomial_log_envelope

pytestmark = pytest.mark.relaxation


def _random_monomial(rng, n):
    c = rng.uniform(0.1, 10.0)
    a = rng.uniform(-2.0, 2.0, size=n)
    x_lb = rng.uniform(0.2, 1.0, size=n)
    x_ub = x_lb + rng.uniform(0.1, 4.0, size=n)  # in [0.2, 5]
    return c, a, x_lb, x_ub


def test_soundness():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = rng.integers(1, 5)
        c, a, x_lb, x_ub = _random_monomial(rng, n)
        log_c = np.log(c)
        u_lb = np.log(x_lb)
        u_ub = np.log(x_ub)
        for _ in range(50):
            u = rng.uniform(u_lb, u_ub)
            s = log_c + float(np.dot(a, u))
            t = np.exp(s)
            cv, cc = monomial_log_envelope(u, log_c, a, u_lb, u_ub)
            assert float(cv) <= t + 1e-9
            assert float(cc) >= t - 1e-9


def test_curvature():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = rng.integers(1, 5)
        c, a, x_lb, x_ub = _random_monomial(rng, n)
        log_c = np.log(c)
        u_lb = np.log(x_lb)
        u_ub = np.log(x_ub)
        for _ in range(30):
            ua = rng.uniform(u_lb, u_ub)
            ub = rng.uniform(u_lb, u_ub)
            umid = 0.5 * (ua + ub)
            cv_a, cc_a = monomial_log_envelope(ua, log_c, a, u_lb, u_ub)
            cv_b, cc_b = monomial_log_envelope(ub, log_c, a, u_lb, u_ub)
            cv_m, cc_m = monomial_log_envelope(umid, log_c, a, u_lb, u_ub)
            # cv convex: midpoint <= average
            assert float(cv_m) <= 0.5 * (float(cv_a) + float(cv_b)) + 1e-6
            # cc concave: midpoint >= average
            assert float(cc_m) >= 0.5 * (float(cc_a) + float(cc_b)) - 1e-6


def test_cv_tightness():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = rng.integers(1, 5)
        c, a, x_lb, x_ub = _random_monomial(rng, n)
        log_c = np.log(c)
        u_lb = np.log(x_lb)
        u_ub = np.log(x_ub)
        for _ in range(30):
            u = rng.uniform(u_lb, u_ub)
            s = log_c + float(np.dot(a, u))
            t = np.exp(s)
            cv, _ = monomial_log_envelope(u, log_c, a, u_lb, u_ub)
            assert abs(float(cv) - t) <= 1e-9


def test_value_at_log_x():
    rng = np.random.default_rng(3)
    for _ in range(40):
        n = rng.integers(1, 5)
        c, a, x_lb, x_ub = _random_monomial(rng, n)
        x = rng.uniform(x_lb, x_ub)
        u = np.log(x)
        log_c = np.log(c)
        lifted = float(jnp.exp(log_c + jnp.sum(jnp.asarray(a) * jnp.asarray(u))))
        direct = c * float(np.prod(x**a))
        assert abs(lifted - direct) <= 1e-9 * abs(direct)
