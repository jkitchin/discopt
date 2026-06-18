"""Differentiable MILP/MIQP via fix-and-differentiate.

``differentiable_solve`` handles integer/binary models by solving the integer
problem, fixing the integers at their optimal values, and differentiating the
resulting continuous restriction at the incumbent (the envelope theorem at the
fixed-integer optimum). The optimal integer assignment is locally
piecewise-constant in the parameters, so wherever that optimum is stable the
gradient ``d(obj*)/dp`` is exact — which these tests pin against central finite
differences.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.differentiable import differentiable_solve


def _fd_obj_gradient(make_model, p0, eps=1e-5):
    """Central finite-difference d(obj*)/dp at p0 using full re-solves."""
    m_plus, _ = make_model(p0 + eps)
    m_minus, _ = make_model(p0 - eps)
    return (differentiable_solve(m_plus).objective - differentiable_solve(m_minus).objective) / (
        2 * eps
    )


# ───────────────────────────── MILP ─────────────────────────────


def _milp_fixed_charge(p0):
    """min p*y + x  s.t.  x >= 3 - y,  y∈{0,1}, x∈[0,10].

    For small p the optimum is y=1, x=2 (obj = p + 2), so d(obj)/dp = 1.
    """
    m = dm.Model("milp_fc")
    p = m.parameter("p", value=p0)
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x >= 3 - y)
    m.minimize(p * y + x)
    return m, p


def test_milp_solution_is_integer_optimum():
    m, _ = _milp_fixed_charge(0.5)
    r = differentiable_solve(m)
    assert r.status == "optimal"
    assert float(r.objective) == pytest.approx(2.5, abs=1e-4)
    assert float(np.asarray(r.x["y"])) == pytest.approx(1.0, abs=1e-6)
    assert float(np.asarray(r.x["x"])) == pytest.approx(2.0, abs=1e-4)


def test_milp_gradient_matches_finite_difference():
    m, p = _milp_fixed_charge(0.5)
    r = differentiable_solve(m)
    analytic = float(r.gradient(p))
    fd = _fd_obj_gradient(_milp_fixed_charge, 0.5)
    assert analytic == pytest.approx(fd, abs=1e-4)
    assert analytic == pytest.approx(1.0, abs=1e-4)


def _milp_param_rhs(b0):
    """min x + 2*z*  s.t.  x >= b,  z binary unused; tests RHS sensitivity."""
    m = dm.Model("milp_rhs")
    b = m.parameter("b", value=b0)
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x >= b)
    m.minimize(x + 2 * z)
    return m, b


def test_milp_rhs_gradient_matches_finite_difference():
    # Optimum: z=0, x=b -> obj=b -> d(obj)/db = 1.
    m, b = _milp_param_rhs(2.0)
    r = differentiable_solve(m)
    analytic = float(r.gradient(b))
    fd = _fd_obj_gradient(_milp_param_rhs, 2.0)
    assert analytic == pytest.approx(fd, abs=1e-4)
    assert analytic == pytest.approx(1.0, abs=1e-4)


# ───────────────────────────── MIQP ─────────────────────────────


def _miqp_binding(p0):
    """min (x - p)^2 + 0.5*y  s.t.  x <= 0.5,  y∈{0,1}, x∈[-5,5].

    For p>0.5 the box/constraint binds at x=0.5, y=0, obj=(0.5-p)^2; the
    gradient flows through the binding constraint: d(obj)/dp = 2(p-0.5).
    """
    m = dm.Model("miqp_bind")
    p = m.parameter("p", value=p0)
    y = m.binary("y")
    x = m.continuous("x", lb=-5, ub=5)
    m.subject_to(x <= 0.5)
    m.minimize((x - p) ** 2 + 0.5 * y)
    return m, p


def test_miqp_gradient_matches_finite_difference():
    m, p = _miqp_binding(2.0)
    r = differentiable_solve(m)
    assert r.status == "optimal"
    analytic = float(r.gradient(p))
    fd = _fd_obj_gradient(_miqp_binding, 2.0)
    assert analytic == pytest.approx(fd, abs=1e-3)
    # Analytic envelope value: 2*(p - 0.5) = 3.0 at p=2.
    assert analytic == pytest.approx(3.0, abs=1e-2)


def _miqp_interior(p0):
    """min (x - p)^2 + 0.5*y  s.t.  x >= y,  y∈{0,1}, x∈[-5,5].

    For 0<p<5 the unconstrained continuous optimum x=p is feasible with y=0,
    so obj=0 and d(obj)/dp = 0 (a flat-objective stability check).
    """
    m = dm.Model("miqp_int")
    p = m.parameter("p", value=p0)
    y = m.binary("y")
    x = m.continuous("x", lb=-5, ub=5)
    m.subject_to(x >= y)
    m.minimize((x - p) ** 2 + 0.5 * y)
    return m, p


def test_miqp_interior_gradient_is_zero():
    m, p = _miqp_interior(0.7)
    r = differentiable_solve(m)
    assert float(r.objective) == pytest.approx(0.0, abs=1e-4)
    assert float(r.gradient(p)) == pytest.approx(0.0, abs=1e-3)


# ──────────────────── JAX-composable MILP layer ────────────────────


def test_milp_objective_layer_composes_under_jax_grad():
    """make_milp_objective_layer is differentiable inside a jax.grad pipeline."""
    import jax
    import jax.numpy as jnp
    from discopt._jax.pounce_layer import make_milp_objective_layer

    m, p = _milp_fixed_charge(0.5)
    layer = make_milp_objective_layer(m, [p])

    obj = float(layer(jnp.array([0.5])))
    assert obj == pytest.approx(2.5, abs=1e-3)

    # d(obj)/dp = 1.
    g = jax.grad(lambda pv: layer(pv))(jnp.array([0.5]))
    assert float(np.asarray(g)[0]) == pytest.approx(1.0, abs=1e-3)

    # Composes through a downstream loss: d/dp (obj - 2)^2 = 2*(2.5-2)*1 = 1.
    lg = jax.grad(lambda pv: (layer(pv) - 2.0) ** 2)(jnp.array([0.5]))
    assert float(np.asarray(lg)[0]) == pytest.approx(1.0, abs=1e-3)
