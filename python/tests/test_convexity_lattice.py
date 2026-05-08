"""Unit tests for the three lattices and the composition rule."""

from __future__ import annotations

import pytest
from discopt._jax.convexity.lattice import (
    AtomProfile,
    Curvature,
    Monotonicity,
    Sign,
    combine_sum,
    compose,
    is_neg,
    is_nonneg,
    is_nonpos,
    is_pos,
    is_strict,
    negate,
    scale,
    sign_add,
    sign_from_bounds,
    sign_from_value,
    sign_mul,
    sign_negate,
    sign_reciprocal,
    unary_atom_profile,
)

pytestmark = pytest.mark.unit


class TestCurvatureLattice:
    def test_negate_flips_curvature(self):
        assert negate(Curvature.CONVEX) == Curvature.CONCAVE
        assert negate(Curvature.CONCAVE) == Curvature.CONVEX
        assert negate(Curvature.AFFINE) == Curvature.AFFINE
        assert negate(Curvature.UNKNOWN) == Curvature.UNKNOWN

    def test_combine_sum(self):
        assert combine_sum(Curvature.CONVEX, Curvature.CONVEX) == Curvature.CONVEX
        assert combine_sum(Curvature.CONVEX, Curvature.AFFINE) == Curvature.CONVEX
        assert combine_sum(Curvature.CONCAVE, Curvature.CONCAVE) == Curvature.CONCAVE
        assert combine_sum(Curvature.CONVEX, Curvature.CONCAVE) == Curvature.UNKNOWN
        assert combine_sum(Curvature.UNKNOWN, Curvature.AFFINE) == Curvature.UNKNOWN

    def test_scale(self):
        assert scale(Curvature.CONVEX, 1) == Curvature.CONVEX
        assert scale(Curvature.CONVEX, -1) == Curvature.CONCAVE
        assert scale(Curvature.CONVEX, 0) == Curvature.AFFINE
        assert scale(Curvature.AFFINE, -1) == Curvature.AFFINE


class TestSignLattice:
    def test_from_bounds_strict(self):
        assert sign_from_bounds(0.1, 10) == Sign.POS
        assert sign_from_bounds(-10, -0.1) == Sign.NEG
        assert sign_from_bounds(0, 0) == Sign.ZERO
        assert sign_from_bounds(0, 10) == Sign.NONNEG
        assert sign_from_bounds(-10, 0) == Sign.NONPOS
        assert sign_from_bounds(-5, 5) == Sign.UNKNOWN

    def test_from_value(self):
        assert sign_from_value(3.0) == Sign.POS
        assert sign_from_value(-3.0) == Sign.NEG
        assert sign_from_value(0.0) == Sign.ZERO
        assert sign_from_value([1.0, 2.0, 3.0]) == Sign.POS
        assert sign_from_value([0.0, 1.0]) == Sign.NONNEG
        assert sign_from_value([-1.0, 1.0]) == Sign.UNKNOWN

    def test_negate_sign(self):
        assert sign_negate(Sign.POS) == Sign.NEG
        assert sign_negate(Sign.NEG) == Sign.POS
        assert sign_negate(Sign.NONNEG) == Sign.NONPOS
        assert sign_negate(Sign.NONPOS) == Sign.NONNEG
        assert sign_negate(Sign.ZERO) == Sign.ZERO
        assert sign_negate(Sign.UNKNOWN) == Sign.UNKNOWN

    def test_predicates(self):
        assert is_pos(Sign.POS)
        assert not is_pos(Sign.NONNEG)
        assert is_neg(Sign.NEG)
        assert not is_neg(Sign.NONPOS)
        assert is_nonneg(Sign.POS) and is_nonneg(Sign.NONNEG) and is_nonneg(Sign.ZERO)
        assert is_nonpos(Sign.NEG) and is_nonpos(Sign.NONPOS) and is_nonpos(Sign.ZERO)
        assert is_strict(Sign.POS) and is_strict(Sign.NEG)
        assert not is_strict(Sign.UNKNOWN)

    def test_add(self):
        assert sign_add(Sign.POS, Sign.POS) == Sign.POS
        assert sign_add(Sign.POS, Sign.NONNEG) == Sign.POS
        assert sign_add(Sign.NONNEG, Sign.NONNEG) == Sign.NONNEG
        assert sign_add(Sign.POS, Sign.NEG) == Sign.UNKNOWN
        assert sign_add(Sign.ZERO, Sign.NEG) == Sign.NEG
        assert sign_add(Sign.ZERO, Sign.ZERO) == Sign.ZERO

    def test_mul(self):
        assert sign_mul(Sign.POS, Sign.POS) == Sign.POS
        assert sign_mul(Sign.NEG, Sign.NEG) == Sign.POS
        assert sign_mul(Sign.POS, Sign.NEG) == Sign.NEG
        assert sign_mul(Sign.NONNEG, Sign.NONNEG) == Sign.NONNEG
        assert sign_mul(Sign.ZERO, Sign.UNKNOWN) == Sign.ZERO
        assert sign_mul(Sign.POS, Sign.UNKNOWN) == Sign.UNKNOWN

    def test_reciprocal_requires_strict(self):
        assert sign_reciprocal(Sign.POS) == Sign.POS
        assert sign_reciprocal(Sign.NEG) == Sign.NEG
        # Non-strict signs could include zero — undefined, return UNKNOWN.
        assert sign_reciprocal(Sign.NONNEG) == Sign.UNKNOWN
        assert sign_reciprocal(Sign.NONPOS) == Sign.UNKNOWN
        assert sign_reciprocal(Sign.ZERO) == Sign.UNKNOWN


class TestCompose:
    """DCP composition: h = f(g)."""

    def test_affine_inner_preserves_outer(self):
        for gc in (Curvature.AFFINE,):
            for fc in Curvature:
                for fm in Monotonicity:
                    assert compose(fc, fm, gc) == fc

    def test_convex_nondec_convex(self):
        assert compose(Curvature.CONVEX, Monotonicity.NONDEC, Curvature.CONVEX) == Curvature.CONVEX

    def test_convex_nonincr_concave(self):
        assert compose(Curvature.CONVEX, Monotonicity.NONINC, Curvature.CONCAVE) == Curvature.CONVEX

    def test_concave_nondec_concave(self):
        result = compose(Curvature.CONCAVE, Monotonicity.NONDEC, Curvature.CONCAVE)
        assert result == Curvature.CONCAVE

    def test_concave_nonincr_convex(self):
        result = compose(Curvature.CONCAVE, Monotonicity.NONINC, Curvature.CONVEX)
        assert result == Curvature.CONCAVE

    def test_convex_wrong_monotonicity(self):
        # convex + nondec composed with concave: UNKNOWN.
        r1 = compose(Curvature.CONVEX, Monotonicity.NONDEC, Curvature.CONCAVE)
        r2 = compose(Curvature.CONVEX, Monotonicity.NONINC, Curvature.CONVEX)
        assert r1 == Curvature.UNKNOWN
        assert r2 == Curvature.UNKNOWN

    def test_outer_unknown_yields_unknown(self):
        result = compose(Curvature.UNKNOWN, Monotonicity.NONDEC, Curvature.CONVEX)
        assert result == Curvature.UNKNOWN


class TestAtomProfiles:
    def test_exp_always_convex_nondec(self):
        for s in Sign:
            prof = unary_atom_profile("exp", s)
            assert prof == AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)

    def test_log_requires_strict_positive(self):
        assert unary_atom_profile("log", Sign.POS) == AtomProfile(
            Curvature.CONCAVE, Monotonicity.NONDEC
        )
        assert unary_atom_profile("log", Sign.NONNEG) is None
        assert unary_atom_profile("log", Sign.UNKNOWN) is None

    def test_sqrt_requires_nonneg(self):
        assert unary_atom_profile("sqrt", Sign.POS) == AtomProfile(
            Curvature.CONCAVE, Monotonicity.NONDEC
        )
        assert unary_atom_profile("sqrt", Sign.NONNEG) == AtomProfile(
            Curvature.CONCAVE, Monotonicity.NONDEC
        )
        assert unary_atom_profile("sqrt", Sign.NEG) is None
        assert unary_atom_profile("sqrt", Sign.UNKNOWN) is None

    def test_abs_is_convex_with_sign_dependent_monotonicity(self):
        pos = unary_atom_profile("abs", Sign.POS)
        assert pos == AtomProfile(Curvature.CONVEX, Monotonicity.NONDEC)
        neg = unary_atom_profile("abs", Sign.NEG)
        assert neg == AtomProfile(Curvature.CONVEX, Monotonicity.NONINC)
        unknown = unary_atom_profile("abs", Sign.UNKNOWN)
        assert unknown == AtomProfile(Curvature.CONVEX, Monotonicity.UNKNOWN)

    def test_tanh_changes_curvature_by_sign(self):
        assert unary_atom_profile("tanh", Sign.POS) == AtomProfile(
            Curvature.CONCAVE, Monotonicity.NONDEC
        )
        assert unary_atom_profile("tanh", Sign.NEG) == AtomProfile(
            Curvature.CONVEX, Monotonicity.NONDEC
        )
        assert unary_atom_profile("tanh", Sign.UNKNOWN) is None

    def test_unknown_atom(self):
        assert unary_atom_profile("not_an_atom", Sign.POS) is None
