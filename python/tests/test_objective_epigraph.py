"""Tests for the objective-defining-equality relaxation.

The transform (``discopt._jax.objective_epigraph``) rewrites the SUSPECT
"objective constraint" pattern ``min z  s.t.  z = g(x)`` into the binding
inequality ``z >= g(x)``. The rewrite is exact at the optimum and unlocks
the convex solve path when ``g`` is convex.

These tests pin both directions of the soundness gate: the transform fires
only on the proven pattern, abstains conservatively otherwise, never
mutates the caller's model, and never changes the optimum.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._jax.convexity import classify_model
from discopt._jax.objective_epigraph import (
    _affine_coeff,
    _occurs,
    relax_objective_defining_equality,
)


def _min_sumsq_model(z_lb=-1e20, z_ub=1e20):
    """``min z  s.t.  z = x^2 + y^2`` — convex defining equality."""
    m = dm.Model("sumsq")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=z_lb, ub=z_ub)
    m.subject_to(z == x * x + y * y)
    m.minimize(z)
    return m


class TestFires:
    def test_relaxes_convex_defining_equality(self):
        m = _min_sumsq_model()
        assert classify_model(m, use_certificate=True)[0] is False
        m2, changed = relax_objective_defining_equality(m)
        assert changed is True
        # original is untouched; the copy carries the relaxed inequality.
        assert m._constraints[0].sense == "=="
        assert m2._constraints[0].sense == ">="
        assert classify_model(m2, use_certificate=True)[0] is True

    def test_exact_optimum_preserved(self):
        m = _min_sumsq_model()
        m2, changed = relax_objective_defining_equality(m)
        assert changed
        r = m2.solve(time_limit=30)
        assert r.status == "optimal"
        assert getattr(r, "gap_certified", False) is True
        # true minimum of x^2 + y^2 is 0 at x=y=0.
        assert r.objective == pytest.approx(0.0, abs=1e-5)

    def test_maximize_concave_defining_equality(self):
        # max z s.t. z = -(x^2) -> concave body -> relax to <=, convex.
        m = dm.Model("maxconcave")
        x = m.continuous("x", lb=-3, ub=3)
        z = m.continuous("z", lb=-1e20, ub=1e20)
        m.subject_to(z == -(x * x))
        m.maximize(z)
        m2, changed = relax_objective_defining_equality(m)
        assert changed is True
        assert m2._constraints[0].sense == "<="
        assert classify_model(m2, use_certificate=True)[0] is True


class TestAbstains:
    def test_bounded_below_does_not_fire_for_min(self):
        # z not free below -> the binding direction is blocked -> abstain.
        m = _min_sumsq_model(z_lb=0.0)
        _, changed = relax_objective_defining_equality(m)
        assert changed is False

    def test_affine_defining_equality_left_for_substitution(self):
        # z = 2x + 3y is affine -> presolve substitution is better; abstain.
        m = dm.Model("affine")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        z = m.continuous("z", lb=-1e20, ub=1e20)
        m.subject_to(z == 2 * x + 3 * y)
        m.minimize(z)
        _, changed = relax_objective_defining_equality(m)
        assert changed is False

    def test_z_in_two_constraints_abstains(self):
        m = _min_sumsq_model()
        x = m._variables[0]
        z = m._variables[2]
        m.subject_to(z >= x)  # second occurrence of z
        _, changed = relax_objective_defining_equality(m)
        assert changed is False

    def test_concave_g_for_min_abstains(self):
        # min z s.t. z = -(x^2): relaxing to z >= -(x^2) is convex BUT the
        # body z - (-(x^2)) = z + x^2 is convex, not concave -> not convex
        # as a >= constraint -> abstain (genuinely nonconvex problem).
        m = dm.Model("minconcave")
        x = m.continuous("x", lb=-3, ub=3)
        z = m.continuous("z", lb=-1e20, ub=1e20)
        m.subject_to(z == -(x * x))
        m.minimize(z)
        _, changed = relax_objective_defining_equality(m)
        assert changed is False

    def test_nonlinear_in_z_abstains(self):
        # z appears nonlinearly (z^2) in the defining constraint.
        m = dm.Model("nonlinz")
        x = m.continuous("x", lb=-3, ub=3)
        z = m.continuous("z", lb=-1e20, ub=1e20)
        m.subject_to(z * z == x * x)
        m.minimize(z)
        _, changed = relax_objective_defining_equality(m)
        assert changed is False

    def test_objective_not_single_variable_abstains(self):
        # objective is z + x, not a bare variable.
        m = dm.Model("notbare")
        x = m.continuous("x", lb=-3, ub=3)
        z = m.continuous("z", lb=-1e20, ub=1e20)
        m.subject_to(z == x * x)
        m.minimize(z + x)
        _, changed = relax_objective_defining_equality(m)
        assert changed is False


class TestStructuralAnalyzers:
    def test_affine_coeff_linear(self):
        m = dm.Model("c")
        x = m.continuous("x", lb=-1, ub=1)
        z = m.continuous("z", lb=-1, ub=1)
        body = z + 2.0 * x
        assert _affine_coeff(body, "z") == pytest.approx(1.0)
        assert _affine_coeff(2.0 * z + x, "z") == pytest.approx(2.0)
        assert _affine_coeff(x - z, "z") == pytest.approx(-1.0)

    def test_affine_coeff_nonlinear_returns_none(self):
        m = dm.Model("c")
        z = m.continuous("z", lb=-1, ub=1)
        assert _affine_coeff(z * z, "z") is None

    def test_occurs(self):
        m = dm.Model("c")
        x = m.continuous("x", lb=-1, ub=1)
        z = m.continuous("z", lb=-1, ub=1)
        assert _occurs(z + x, "z") is True
        assert _occurs(x * x, "z") is False
