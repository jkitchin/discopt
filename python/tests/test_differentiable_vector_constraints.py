"""Regression for issue #324: ``differentiable_solve`` on vector / mixed-shape
constraints.

The envelope-theorem sensitivity step assembled per-constraint values with
``jnp.array([...])``, which *stacks* — only valid when every constraint is
scalar. With vector constraints it either fails to stack (mixed shapes) or
mis-aligns the ``lambda·g`` dot (equal shapes), because the multipliers vector is
per constraint *row* while the stack is per *constraint*. The fix ravels +
concatenates each constraint value, matching the multiplier ordering row-for-row.

These tests guard both the crash AND the correctness of the resulting gradient
(a misaligned multiplier vector would silently return a *wrong* gradient, not
crash), via a finite-difference reference.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
from discopt._jax.differentiable import differentiable_solve  # noqa: E402


def _mixed_shape_model(pv: float):
    """MAX sum(x)+sum(y) s.t. x<=0.5p (shape 3), y<=0.3p (shape 2), x,y>=0.
    Both constraints bind at the optimum, so multipliers are nonzero and the
    sensitivity is meaningful: obj* = 3*0.5p + 2*0.3p = 2.1p (min-convention -2.1p).
    """
    m = dm.Model("mixed")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    y = m.continuous("y", shape=(2,), lb=0, ub=10)
    p = m.parameter("p", pv)
    m.subject_to(x <= 0.5 * p)
    m.subject_to(y <= 0.3 * p)
    m.maximize(dm.sum(x) + dm.sum(y))
    return m, p


def _scalar_model(pv: float):
    """Scalar-constraint control (the case that already worked): MAX x s.t. x<=2p."""
    m = dm.Model("scalar")
    x = m.continuous("x", lb=0, ub=10)
    p = m.parameter("p", pv)
    m.subject_to(x <= 2.0 * p)
    m.maximize(x)
    return m, p


def test_mixed_shape_constraints_do_not_crash():
    """The issue's reproducer: constraint bodies of differing shapes (3,) and (2,)."""
    m = dm.Model("repro")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    y = m.continuous("y", shape=(2,), lb=0, ub=1)
    p = m.parameter("p", 1.0)
    m.subject_to(x <= 0.5 * p)
    m.subject_to(y <= 0.5 * p)
    m.minimize(dm.sum(x) + dm.sum(y))
    r = differentiable_solve(m, nlp_solver="ipm")  # used to raise TypeError
    assert r.status == "optimal"


def test_equal_shape_vector_constraints_do_not_crash():
    """Equal-shape vector constraints (3,)+(3,) used to fail at the lambda·g dot."""
    m = dm.Model("eq")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", 1.0)
    m.subject_to(x <= 0.5 * p)
    m.subject_to(y <= 0.5 * p)
    m.minimize(dm.sum(x) + dm.sum(y))
    r = differentiable_solve(m, nlp_solver="ipm")
    assert r.status == "optimal"


def test_mixed_shape_gradient_matches_finite_difference():
    """The gradient must be CORRECT (multiplier ordering aligned), not just
    non-crashing — checked against a central finite difference of obj*(p)."""

    def obj_star(pv: float) -> float:
        m, _ = _mixed_shape_model(pv)
        return float(differentiable_solve(m, nlp_solver="ipm").objective)

    m, p = _mixed_shape_model(2.0)
    r = differentiable_solve(m, nlp_solver="ipm")
    g = float(np.ravel(np.asarray(r.gradient(p)))[0])

    eps = 1e-4
    fd = (obj_star(2.0 + eps) - obj_star(2.0 - eps)) / (2 * eps)

    # Closed form: obj (min-convention) = -2.1p -> d/dp = -2.1.
    assert np.isclose(g, fd, atol=1e-2), f"gradient {g} != finite-diff {fd}"
    assert np.isclose(g, -2.1, atol=1e-2), f"gradient {g} != analytic -2.1"


def test_l3_mixed_shape_gradient_matches_finite_difference():
    """The L3 (implicit-diff) path has its own Lagrangian assembly
    (`differentiable_solve_l3` / `implicit_differentiate`) — same #324 bug, same
    fix. Guard it too: vector constraints must not crash and the gradient must
    match a finite difference."""
    from discopt._jax.differentiable import differentiable_solve_l3

    def obj_star(pv: float) -> float:
        m, _ = _mixed_shape_model(pv)
        return float(differentiable_solve_l3(m, nlp_solver="ipm").objective)

    m, p = _mixed_shape_model(2.0)
    r = differentiable_solve_l3(m, nlp_solver="ipm")  # used to raise TypeError
    assert r.status == "optimal"
    g = float(np.ravel(np.asarray(r.gradient(p)))[0])
    eps = 1e-4
    fd = (obj_star(2.0 + eps) - obj_star(2.0 - eps)) / (2 * eps)
    assert np.isclose(g, fd, atol=1e-2), f"L3 gradient {g} != finite-diff {fd}"
    assert np.isclose(g, -2.1, atol=1e-2)


def test_scalar_constraint_gradient_unregressed():
    """Scalar constraints (the previously-working case) stay correct."""

    def obj_star(pv: float) -> float:
        m, _ = _scalar_model(pv)
        return float(differentiable_solve(m, nlp_solver="ipm").objective)

    m, p = _scalar_model(1.5)
    r = differentiable_solve(m, nlp_solver="ipm")
    g = float(np.ravel(np.asarray(r.gradient(p)))[0])
    eps = 1e-4
    fd = (obj_star(1.5 + eps) - obj_star(1.5 - eps)) / (2 * eps)
    # obj (min-convention) = -2p -> d/dp = -2.
    assert np.isclose(g, fd, atol=1e-2)
    assert np.isclose(g, -2.0, atol=1e-2)
