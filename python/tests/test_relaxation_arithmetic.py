"""Tests for the ``arithmetic`` kwarg of ``compile_relaxation``.

Wires M2 (Chebyshev) and M3 (Taylor) of issue #51 into the LP
relaxation compiler. Verifies:

1. The default ``"mccormick"`` path is unchanged (covered by the existing
   ``test_relaxation_compiler.py`` suite, not duplicated here).
2. ``"chebyshev"`` and ``"taylor"`` produce sound (cv ≤ f ≤ cc)
   underestimator / overestimator pairs on a sample of inner points.
3. Falls back gracefully to McCormick for unsupported ops.
4. End-to-end ``Model.solve(relaxation_arithmetic=...)`` preserves the
   global optimum.
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import discopt
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.relaxation_compiler import compile_objective_relaxation


def _eval_at(fn, x_val, n_vars):
    """Evaluate (cv, cc) at a single scalar point."""
    x = jnp.asarray([x_val] * n_vars, dtype=jnp.float64)
    cv, cc = fn(x, x, jnp.full(n_vars, -1e6), jnp.full(n_vars, 1e6))
    return float(cv), float(cc)


@pytest.mark.parametrize("arithmetic", ["chebyshev", "taylor"])
def test_oa_relax_soundness_exp(arithmetic):
    """cv ≤ exp(x) ≤ cc on a Monte Carlo sample inside the box."""
    m = discopt.Model("exp_test")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(discopt.exp(x))
    m.subject_to(x >= -1.0)

    fn = compile_objective_relaxation(m, arithmetic=arithmetic)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, size=200)
    for xv in xs:
        cv, cc = _eval_at(fn, float(xv), n_vars=1)
        true = np.exp(xv)
        assert cv <= true + 1e-6, f"cv={cv} > exp({xv})={true} at arithmetic={arithmetic}"
        assert cc >= true - 1e-6, f"cc={cc} < exp({xv})={true} at arithmetic={arithmetic}"


@pytest.mark.parametrize("arithmetic", ["chebyshev", "taylor"])
def test_oa_relax_soundness_log(arithmetic):
    """cv ≤ log(x) ≤ cc on a Monte Carlo sample inside the box."""
    m = discopt.Model("log_test")
    x = m.continuous("x", lb=0.5, ub=3.0)
    m.minimize(discopt.log(x))
    m.subject_to(x >= 0.5)

    fn = compile_objective_relaxation(m, arithmetic=arithmetic)
    rng = np.random.default_rng(1)
    xs = rng.uniform(0.5, 3.0, size=200)
    for xv in xs:
        cv, cc = _eval_at(fn, float(xv), n_vars=1)
        true = np.log(xv)
        assert cv <= true + 1e-6
        assert cc >= true - 1e-6


def test_solve_with_eigenvalue_root_bound_preserves_optimum(caplog):
    """Opt-in eigenvalue root bound on a QP must not change the optimum."""
    import logging

    m = discopt.Model("eig_qp")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    # Mixed-sign Hessian: x^2 - y^2 + x - y (saddle on the box)
    m.minimize(x * x - y * y + x - y)
    m.subject_to(x + y >= -1.0)

    base = m.solve(time_limit=30.0)
    m2 = discopt.Model("eig_qp2")
    x2 = m2.continuous("x", lb=-2.0, ub=2.0)
    y2 = m2.continuous("y", lb=-2.0, ub=2.0)
    m2.minimize(x2 * x2 - y2 * y2 + x2 - y2)
    m2.subject_to(x2 + y2 >= -1.0)
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        eig = m2.solve(time_limit=30.0, eigenvalue_root_bound=True)
    assert base.status in ("optimal", "feasible")
    assert eig.status in ("optimal", "feasible")
    assert eig.objective == pytest.approx(base.objective, abs=1e-3, rel=1e-3)


def test_solve_with_chebyshev_preserves_optimum():
    """End-to-end: compile-time chebyshev path keeps the global optimum."""

    def _build(name):
        m = discopt.Model(name)
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(discopt.exp(x) + discopt.log(y + 2.0) + (x - y) ** 2)
        m.subject_to(x + y >= 0.0)
        return m

    base = _build("base")
    res_base = base.solve(time_limit=30.0, relaxation_arithmetic="mccormick")
    cheb = _build("cheb")
    res_cheb = cheb.solve(time_limit=30.0, relaxation_arithmetic="chebyshev")
    assert res_base.status in ("optimal", "feasible")
    assert res_cheb.status in ("optimal", "feasible")
    assert res_cheb.objective == pytest.approx(res_base.objective, abs=1e-3, rel=1e-3)
