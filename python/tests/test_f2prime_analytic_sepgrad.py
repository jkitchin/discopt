"""F2′ — analytic IR gradient for convex separation (issue #632).

The supporting-hyperplane cut ``d ≥ g(x₀) + ∇g(x₀)·(x−x₀)`` needs ``∇g(x₀)``.
``g`` is a known factorable expression, so its gradient is the exact derivative
of the engine's own IR, computed by forward-mode interval AD at a *point* box
(``interval_hessian`` with ``lb == ub``) — no JAX. These tests pin the two
properties the separation soundness/quality relies on:

  1. the analytic point-gradient equals ``jax.grad`` of the same expression to
     floating-point (same function — a wrong gradient would emit an invalid or
     non-separating cut); and
  2. it is deterministic (byte-identical across repeats) — the reproducibility
     property the jitted path lacked (see ``docs/dev/sota-proof-plan.md`` §5 F2′).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discopt import Model  # noqa: E402
from discopt._jax.convexity.interval import Interval  # noqa: E402
from discopt._jax.convexity.interval_ad import interval_hessian  # noqa: E402
from discopt._jax.dag_compiler import compile_expression  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _cases():
    m = Model()
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    z = m.continuous("z", lb=0.5, ub=3.0)
    exprs = {
        "exp(2x-y)": dm.exp(2 * x - y),
        "log(x+2y)": dm.log(x + 2 * y),
        "quad": x * x + x * y + y * y,
        "linfrac": x / y,
        "sqrt": dm.sqrt(x + y),
        "exp*log+sq": dm.exp(x) * dm.log(y) + z * z,
        "quad_over_lin": (x + y) * (x + y) / z,
    }
    return m, [x, y, z], exprs


def _analytic_value_grad(expr, model, varlist, x0):
    box = {v: Interval(np.float64(x0[i]), np.float64(x0[i])) for i, v in enumerate(varlist)}
    ad = interval_hessian(expr, model, box)
    val = float(np.asarray(ad.value.lo).ravel()[0])
    g = np.asarray(ad.grad.lo, dtype=np.float64).ravel()
    return val, g


@pytest.mark.parametrize("name", list(_cases()[2].keys()))
def test_analytic_matches_autodiff(name):
    m, varlist, exprs = _cases()
    expr = exprs[name]
    f = compile_expression(expr, m)
    gf = jax.grad(lambda xv: jnp.reshape(f(xv), ()))
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    for _ in range(12):
        x0 = rng.uniform(0.6, 2.8, size=len(varlist))
        xv = jnp.asarray(x0)
        jval = float(jnp.reshape(f(xv), ()))
        jg = np.asarray(gf(xv), dtype=np.float64).ravel()
        aval, ag = _analytic_value_grad(expr, m, varlist, x0)
        assert np.isfinite(aval) and np.all(np.isfinite(ag)), f"{name}: uncovered atom"
        assert abs(jval - aval) < 1e-8, f"{name}: value {jval} vs {aval}"
        assert np.max(np.abs(jg - ag)) < 1e-8, f"{name}: grad {jg} vs {ag}"


@pytest.mark.parametrize("name", list(_cases()[2].keys()))
def test_analytic_is_deterministic(name):
    m, varlist, exprs = _cases()
    expr = exprs[name]
    x0 = np.array([1.3, 0.9, 2.1][: len(varlist)], dtype=np.float64)
    v1, g1 = _analytic_value_grad(expr, m, varlist, x0)
    v2, g2 = _analytic_value_grad(expr, m, varlist, x0)
    assert v1 == v2, f"{name}: value not byte-identical across repeats"
    assert np.array_equal(g1, g2), f"{name}: grad not byte-identical across repeats"
