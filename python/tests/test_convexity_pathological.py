"""Pathological-case tests for the convexity detector.

These tests stress the detector on cases that are easy to misclassify.
The overriding contract is **soundness**: a CONVEX or CONCAVE verdict
is a proof, never a guess. A conservative UNKNOWN is always acceptable;
an over-claim is a bug.

Cases the Phase 1 SUSPECT-style rule extension is expected to tighten
from UNKNOWN to CONVEX/CONCAVE are marked with
``@pytest.mark.xfail(strict=True, reason="phase 1 extension")`` so they
flip to an xpass failure when the rule actually lands, forcing the
marker to be removed consciously.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._jax.convexity import Curvature, classify_expr
from discopt.modeling.core import FunctionCall, Model

# ──────────────────────────────────────────────────────────────────────
# Soundness: detector must never over-claim
# ──────────────────────────────────────────────────────────────────────


class TestSoundnessNeverOverClaims:
    """Canonical nonconvex expressions that must NOT be labeled CONVEX/CONCAVE."""

    def test_bilinear_positive_orthant_still_nonconvex(self):
        """x*y with x,y >= 0 remains indefinite (eigenvalues ±1)."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        assert classify_expr(x * y, m) == Curvature.UNKNOWN

    def test_gaussian_bump_not_concave(self):
        """exp(-x^2) has inflection points and is neither convex nor concave."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.exp(-(x**2)), m) == Curvature.UNKNOWN

    def test_sigmoid_not_classified(self):
        """1 / (1 + exp(-x)) is neither convex nor concave globally."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        # Built via exp and division — detector should yield UNKNOWN.
        expr = 1.0 / (1.0 + dm.exp(-x))
        assert classify_expr(expr, m) == Curvature.UNKNOWN

    def test_odd_power_mixed_sign_is_unknown(self):
        """x^3 on [-1,1] is not convex (concave on [-1,0], convex on [0,1])."""
        m = Model("t")
        x = m.continuous("x", lb=-1, ub=1)
        assert classify_expr(x**3, m) == Curvature.UNKNOWN

    def test_sin_on_unbounded_domain_is_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-10, ub=10)
        assert classify_expr(dm.sin(x), m) == Curvature.UNKNOWN

    def test_convex_minus_convex_is_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        assert classify_expr(x**2 - y**2, m) == Curvature.UNKNOWN

    def test_convex_times_convex_is_unknown(self):
        """Product of convex functions is not generally convex."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        # (x^2) * (y^2) — nonconvex bilinear-of-squares.
        assert classify_expr((x**2) * (y**2), m) == Curvature.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Composition traps
# ──────────────────────────────────────────────────────────────────────


class TestCompositionTraps:
    """Composition that breaks without the right monotonicity."""

    def test_exp_of_concave_is_unknown(self):
        """exp is nondecreasing but exp(concave) is not concave.

        With a concave argument, exp(g) is generally non-concave; the
        sound verdict is UNKNOWN.
        """
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.exp(dm.log(x)), m) == Curvature.UNKNOWN

    def test_log_of_convex_is_unknown(self):
        """log is concave+nondec; log(convex) yields no sound verdict."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.log(dm.exp(x)), m) == Curvature.UNKNOWN

    def test_sqrt_of_psd_quadratic_is_convex(self):
        """sqrt of a PSD quadratic form is a norm, hence convex.

        ``sqrt(x^2) = |x|``; the norm recogniser in :mod:`patterns`
        proves convexity directly even though the naive
        concave-of-convex DCP composition fails.
        """
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.sqrt(x**2), m) == Curvature.CONVEX


# ──────────────────────────────────────────────────────────────────────
# Identity-in-disguise — detector should stay conservative
# ──────────────────────────────────────────────────────────────────────


class TestIdentityInDisguise:
    """Expressions that simplify to something simpler.

    The detector walks the written expression tree; it is *not*
    expected to simplify, and it must remain sound on the literal form.
    """

    def test_sqrt_of_square_not_over_claimed(self):
        """(x^2)^0.5 = |x| is convex but written as sqrt(x^2) the
        detector sees sqrt(convex) and returns UNKNOWN. This is sound
        (we miss a convex case) but must not flip to a wrong verdict.
        """
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        verdict = classify_expr(dm.sqrt(x**2), m)
        assert verdict in (Curvature.CONVEX, Curvature.UNKNOWN)

    def test_exp_log_round_trip_is_unknown(self):
        """exp(log(x)) = x but the detector should not assert affine."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.exp(dm.log(x)), m) == Curvature.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Domain-restricted convexity
# ──────────────────────────────────────────────────────────────────────


class TestDomainRestricted:
    """Cases where domain matters — boundary sharpness."""

    def test_odd_power_zero_lower_bound(self):
        """x^3 on [0, ub] is convex (boundary case)."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(x**3, m) == Curvature.CONVEX

    def test_odd_power_just_negative_lower_bound(self):
        """x^3 on [-eps, ub] is NOT provably convex."""
        m = Model("t")
        x = m.continuous("x", lb=-1e-6, ub=10)
        assert classify_expr(x**3, m) == Curvature.UNKNOWN

    def test_fractional_power_at_zero(self):
        """x^0.5 on [0, ub] is concave (defined at the boundary)."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10)
        assert classify_expr(x**0.5, m) == Curvature.CONCAVE


# ──────────────────────────────────────────────────────────────────────
# Phase 1 extensions — expected to tighten to CONVEX/CONCAVE
# ──────────────────────────────────────────────────────────────────────


class TestSignAwareReciprocal:
    """Reciprocal with known sign (Phase 1 sign lattice)."""

    def test_reciprocal_positive_is_convex(self):
        """1/x on strictly positive domain is convex."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(1.0 / x, m) == Curvature.CONVEX

    def test_reciprocal_negative_is_concave(self):
        """1/x on strictly negative domain is concave."""
        m = Model("t")
        x = m.continuous("x", lb=-10, ub=-0.1)
        assert classify_expr(1.0 / x, m) == Curvature.CONCAVE

    def test_reciprocal_of_convex_is_unknown(self):
        """1/(1 + exp(-x)) — denominator positive but convex; sound verdict UNKNOWN."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(1.0 / (1.0 + dm.exp(-x)), m) == Curvature.UNKNOWN

    def test_reciprocal_of_concave_positive_is_convex(self):
        """1/sqrt(x) on x>0 — denominator concave + strictly positive → convex.

        This exercises the full composition path: the reciprocal's
        profile (CONVEX, NONINC) composed with a concave argument
        yields CONVEX.
        """
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10.0)
        assert classify_expr(1.0 / dm.sqrt(x), m) == Curvature.CONVEX


class TestMonotoneComposition:
    """Monotone composition rules already hardcoded for exp / log / sqrt.

    These baselines lock the hardcoded behavior so the Phase 1 rewrite
    (which generalizes to a curvature-by-monotonicity table) doesn't
    accidentally regress the atoms that already work.
    """

    def test_exp_of_convex_nonaffine(self):
        """exp(x^2) is convex: exp convex + nondec, x^2 convex."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(dm.exp(x**2), m) == Curvature.CONVEX

    def test_log_of_concave_nonaffine(self):
        """log(sqrt(x)) is concave: log concave + nondec, sqrt concave."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        assert classify_expr(dm.log(dm.sqrt(x)), m) == Curvature.CONCAVE


class TestNAryAtoms:
    """Multi-argument atoms (Phase 1)."""

    def test_max_of_convex(self):
        """max(x^2, y^2) is convex."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        expr = FunctionCall("max", x**2, y**2)
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_min_of_concave(self):
        """min(log(x), log(y)) is concave."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        y = m.continuous("y", lb=0.1, ub=10)
        expr = FunctionCall("min", dm.log(x), dm.log(y))
        assert classify_expr(expr, m) == Curvature.CONCAVE

    def test_max_of_mixed_is_unknown(self):
        """max(convex, concave) has no sound verdict."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        expr = FunctionCall("max", x**2, dm.log(x))
        assert classify_expr(expr, m) == Curvature.UNKNOWN

    def test_min_of_mixed_is_unknown(self):
        """min(convex, concave) has no sound verdict."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=10)
        expr = FunctionCall("min", x**2, dm.log(x))
        assert classify_expr(expr, m) == Curvature.UNKNOWN


class TestPhase1TrigOnBoundedDomain:
    """Trig functions are convex/concave on restricted domains (Phase 1)."""

    @pytest.mark.xfail(strict=True, reason="phase 1: sin concave on [0,π]")
    def test_sin_on_first_half_period(self):
        import math

        m = Model("t")
        x = m.continuous("x", lb=0.0, ub=math.pi)
        assert classify_expr(dm.sin(x), m) == Curvature.CONCAVE

    @pytest.mark.xfail(strict=True, reason="phase 1: cos concave on [-π/2,π/2]")
    def test_cos_on_central_interval(self):
        import math

        m = Model("t")
        x = m.continuous("x", lb=-math.pi / 2, ub=math.pi / 2)
        assert classify_expr(dm.cos(x), m) == Curvature.CONCAVE


# ──────────────────────────────────────────────────────────────────────
# Degenerate / numerical edge cases
# ──────────────────────────────────────────────────────────────────────


class TestDegenerate:
    """Edge cases that should not crash or mislabel."""

    def test_equal_bounds_single_variable(self):
        """A fixed variable is still affine."""
        m = Model("t")
        x = m.continuous("x", lb=3.0, ub=3.0)
        assert classify_expr(x, m) == Curvature.AFFINE

    def test_double_negation(self):
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(-(-x), m) == Curvature.AFFINE

    def test_double_negation_of_convex(self):
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(-(-(x**2)), m) == Curvature.CONVEX

    def test_tiny_positive_coefficient(self):
        """Scaling by a tiny positive coefficient preserves curvature."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(1e-20 * (x**2), m) == Curvature.CONVEX

    def test_tiny_negative_coefficient(self):
        """Scaling by a tiny negative coefficient flips curvature."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(-1e-20 * (x**2), m) == Curvature.CONCAVE

    def test_large_coefficient(self):
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr(1e20 * (x**2), m) == Curvature.CONVEX

    def test_division_by_negative_preserves_soundness(self):
        """Division by a negative constant flips curvature."""
        m = Model("t")
        x = m.continuous("x", lb=-5, ub=5)
        assert classify_expr((x**2) / -3.0, m) == Curvature.CONCAVE
