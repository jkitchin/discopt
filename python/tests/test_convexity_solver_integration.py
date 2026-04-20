"""Regression tests for the certificate's solver-side wiring.

The solver calls :func:`classify_model` at four sites; the
``use_certificate=True`` path asks the sound interval-Hessian
certificate to tighten UNKNOWN verdicts the syntactic walker left
open. These tests lock in the wiring so a future refactor can't
silently lose the tightening effect.
"""

from __future__ import annotations

import discopt.modeling as dm
from discopt._jax.convexity import Curvature, classify_expr, classify_model
from discopt._jax.convexity.certificate import certify_convex
from discopt.modeling.core import Model


class TestClassifyModelCertificateFallback:
    """``use_certificate`` flips UNKNOWN syntactic verdicts to CONVEX
    when the interval-Hessian certificate proves them on the root box."""

    def test_baseline_syntactic_leaves_quartic_unknown(self):
        """``x**4 - 2*x**2`` on a narrow box is convex but not by DCP."""
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=2.0)
        expr = x**4 - 2.0 * x**2
        # Syntactic rules can't prove this one.
        assert classify_expr(expr, m) == Curvature.UNKNOWN
        # The certificate can.
        assert certify_convex(expr, m) == Curvature.CONVEX

    def test_constraint_lifted_by_certificate(self):
        """A constraint that's convex on its box but UNKNOWN syntactically
        must be marked convex when ``use_certificate=True``."""
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=2.0)
        m.minimize(x)
        m.subject_to(x**4 - 2.0 * x**2 <= 10.0)
        # Syntactic only: constraint stays UNKNOWN, not convex.
        is_cvx_s, mask_s = classify_model(m)
        assert mask_s == [False]
        # With certificate: constraint is proven convex.
        is_cvx_c, mask_c = classify_model(m, use_certificate=True)
        assert mask_c == [True]
        assert is_cvx_c is True

    def test_objective_lifted_by_certificate(self):
        """Objective body that's provably convex on the box but UNKNOWN
        syntactically gets promoted with ``use_certificate``."""
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=2.0)
        m.minimize(x**4 - 2.0 * x**2)
        is_cvx_s, _ = classify_model(m)
        assert is_cvx_s is False
        is_cvx_c, _ = classify_model(m, use_certificate=True)
        assert is_cvx_c is True

    def test_certificate_does_not_downgrade_syntactic_verdicts(self):
        """The certificate must never demote a proven syntactic CONVEX."""
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        m.minimize(x**2)
        m.subject_to(dm.exp(x) <= 10.0)
        is_cvx_s, mask_s = classify_model(m)
        is_cvx_c, mask_c = classify_model(m, use_certificate=True)
        assert mask_s == mask_c
        assert is_cvx_s == is_cvx_c is True

    def test_certificate_silently_skips_when_body_has_array_vars(self):
        """Array-valued expressions aren't supported by the certificate;
        it must abstain gracefully rather than crash."""
        m = Model("t")
        x = m.continuous("x", shape=(3,), lb=-1.0, ub=1.0)
        m.minimize(dm.sum(x**2))
        # Certificate path shouldn't raise; just falls back to syntactic.
        is_cvx, _ = classify_model(m, use_certificate=True)
        assert isinstance(is_cvx, bool)

    def test_certificate_for_nonconvex_stays_nonconvex(self):
        """When neither syntactic nor certificate can prove convexity,
        the mask stays non-convex."""
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(x * y)  # indefinite — genuinely nonconvex
        is_cvx, mask = classify_model(m, use_certificate=True)
        assert is_cvx is False
