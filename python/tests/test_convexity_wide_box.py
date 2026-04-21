"""Wide-bounds convexity tests modelled on Suspect issue #11.

Context: the Suspect project (``cog-imperial/suspect``) tracks a class
of models whose detector verdicts degrade when variable bounds are
declared wide — interval-arithmetic certificates lose precision because
the interval Hessian grows loose (or blows up to ``inf``) before
eigenvalue bounds can prove PSD. discopt's box-local certificate has
the same failure mode when used in isolation, but the structural
pattern recognizers in :mod:`discopt._jax.convexity.patterns` do not
depend on box width and should remain sound regardless of bounds.

These tests assert the intended division of labour:

1. **Structural recognizers survive wide bounds.** Every recognised
   shape (PSD quadratic, norm, quadratic-over-affine, exp-perspective,
   weighted geometric mean, quadratic-over-affine epigraph) is
   classified ``CONVEX`` under aggressive (±1e6..±1e10) bounds. These
   are the positive regressions for the wide-bounds scenario.

2. **The certificate is permitted to abstain on wide boxes.** For
   expressions whose interval Hessian has exp/pow/div atoms, the
   pure box-local certificate may legitimately return ``None`` on a
   wide box (overflow or loose eigenvalue bound). We pin this
   behaviour down so a future tightening of the certificate is a
   controlled change, not an accidental one.

3. **End-to-end solver parity.** A convex model with wide bounds must
   still take the convex fast path via the structural layer.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt._jax.convexity import (
    Curvature,
    certify_convex,
    classify_constraint,
    classify_expr,
)
from discopt.modeling.core import Model

# Representative wide-box scales. Picked to stay within the pure-JAX IPM
# domain while being orders of magnitude wider than any declared bound
# in the MINLPTests cvx suite (those are typically in ±1e2..±1e3).
WIDE = 1.0e8
VERY_WIDE = 1.0e10


# ──────────────────────────────────────────────────────────────────────
# Structural recognizers under wide bounds
# ──────────────────────────────────────────────────────────────────────


class TestWideBoxStructuralRecognizers:
    """Pattern-level convexity proofs must not depend on box width.

    Each test takes a shape the recognizers already handle on the
    declared MINLPTests bounds and stretches the declared bounds by
    6–10 orders of magnitude. The detector verdict must not change.
    """

    def test_psd_quadratic_classified_convex_on_wide_box(self):
        m = Model("psd_quad_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=-WIDE, ub=WIDE)
        expr = x * x + y * y + x * y
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_psd_quadratic_with_cross_term_classified_convex_on_wide_box(self):
        # 2x^2 + 2y^2 + 2xy = [x y] [[2,1],[1,2]] [x y]^T, eigenvalues 1 and 3.
        m = Model("psd_cross_wide")
        x = m.continuous("x", lb=-VERY_WIDE, ub=VERY_WIDE)
        y = m.continuous("y", lb=-VERY_WIDE, ub=VERY_WIDE)
        expr = 2.0 * x * x + 2.0 * y * y + 2.0 * x * y
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_norm_sqrt_of_psd_quadratic_convex_on_wide_box(self):
        m = Model("norm_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=-WIDE, ub=WIDE)
        expr = dm.sqrt(x * x + y * y)
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_quadratic_over_linear_convex_on_wide_box(self):
        # Numerator is a PSD quadratic, denominator strictly positive.
        m = Model("qol_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=1.0e-3, ub=WIDE)
        expr = (x * x) / y
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_exp_perspective_convex_on_wide_box(self):
        # y * exp(x / y), y > 0. exp's argument can span a very wide
        # interval without defeating the pattern recogniser — it does
        # not evaluate exp over the box.
        m = Model("persp_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=1.0e-3, ub=1.0e6)
        expr = y * dm.exp(x / y)
        assert classify_expr(expr, m) == Curvature.CONVEX

    def test_weighted_geometric_mean_concave_on_wide_box(self):
        m = Model("geom_mean_wide")
        x = m.continuous("x", lb=0.0, ub=WIDE)
        y = m.continuous("y", lb=0.0, ub=WIDE)
        expr = (x**0.5) * (y**0.5)
        assert classify_expr(expr, m) == Curvature.CONCAVE

    def test_fractional_epigraph_constraint_convex_on_wide_box(self):
        # nlp_cvx_108-style rearrangement: y * (d*x + e) + (a*x^2 + b*x + c) <= 0
        # with (d*x + e) strictly positive on the box. Convex iff
        # a*e^2 - b*d*e + c*d^2 >= 0.
        m = Model("frac_epi_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=-WIDE, ub=WIDE)
        # Coefficients chosen so (2*x + (2*WIDE + 1)) > 0 on [-WIDE, WIDE].
        d = 2.0
        e = 2.0 * WIDE + 1.0
        # q(x) = x^2, so (a, b, c0) = (1, 0, 0) and a*e^2 - b*d*e + c0*d^2 = e^2 >= 0.
        con = (d * x + e) * (-y) + x * x <= 0  # arranges to y >= x^2 / (d*x+e)
        m.subject_to(con)
        assert classify_constraint(m._constraints[0], m) is True


# ──────────────────────────────────────────────────────────────────────
# Certificate abstention on wide boxes (documenting the known limit)
# ──────────────────────────────────────────────────────────────────────


class TestWideBoxCertificateAbstainsAsDocumented:
    """The pure interval-Hessian certificate may abstain on wide boxes.

    These tests lock in the current behaviour of :func:`certify_convex`
    for shapes whose Hessian contains exp/pow/div atoms: on a wide box
    the interval enclosure overflows or loses enough precision for
    Gershgorin to fail, and the certificate returns ``None``. This is
    exactly the failure mode David flagged on ``nlp_cvx_205_010``, and
    it is *expected* under the current certificate — the structural
    recognizers are the reason those MINLPTests cases still pass.

    If a future certificate tightening (e.g. αBB-style Hessian bounds,
    affine arithmetic) turns any of these into successful proofs, the
    test will surface that change rather than let it slip through.
    """

    def test_exp_perspective_certificate_abstains_on_wide_box(self):
        m = Model("persp_cert_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=1.0e-3, ub=1.0e6)
        # Suppress the overflow warnings that interval arithmetic emits
        # when evaluating exp over the wide box — they are expected here
        # and noise in the test output otherwise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            verdict = certify_convex(y * dm.exp(x / y), m)
        assert verdict is None

    def test_quadratic_over_linear_certificate_abstains_on_wide_box(self):
        m = Model("qol_cert_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=1.0e-3, ub=WIDE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            verdict = certify_convex((x * x) / y, m)
        # The certificate is allowed to succeed or abstain; what we do
        # NOT allow is a contradiction (CONCAVE). Soundness first.
        assert verdict in (Curvature.CONVEX, None)

    def test_structural_layer_strictly_stronger_than_certificate_on_perspective(self):
        """On the exp-perspective shape under wide bounds, the
        structural layer should prove CONVEX while the certificate
        abstains — establishing that the patterns are doing real work
        the certificate cannot."""
        m = Model("persp_combo_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=1.0e-3, ub=1.0e6)
        expr = y * dm.exp(x / y)
        assert classify_expr(expr, m) == Curvature.CONVEX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert certify_convex(expr, m) is None


# ──────────────────────────────────────────────────────────────────────
# Sanity: bilinear stays UNKNOWN/non-convex at any box width
# ──────────────────────────────────────────────────────────────────────


class TestWideBoxSoundnessNegatives:
    """Shapes that are genuinely non-convex must not be promoted to
    CONVEX by any wide-bounds code path."""

    def test_bilinear_not_classified_convex_on_wide_box(self):
        m = Model("bilinear_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=-WIDE, ub=WIDE)
        assert classify_expr(x * y, m) != Curvature.CONVEX
        assert classify_expr(x * y, m) != Curvature.CONCAVE

    def test_indefinite_quadratic_not_classified_convex_on_wide_box(self):
        # x^2 - y^2 has eigenvalues ±2, indefinite.
        m = Model("indef_wide")
        x = m.continuous("x", lb=-WIDE, ub=WIDE)
        y = m.continuous("y", lb=-WIDE, ub=WIDE)
        expr = x * x - y * y
        curv = classify_expr(expr, m)
        assert curv != Curvature.CONVEX
        assert curv != Curvature.CONCAVE


# ──────────────────────────────────────────────────────────────────────
# End-to-end: solver still takes the convex fast path
# ──────────────────────────────────────────────────────────────────────


class TestWideBoxSolverIntegration:
    """Convex models with wide bounds must still take the convex fast
    path — the structural classification flows through to the solver
    gate even when the certificate alone would abstain."""

    @pytest.mark.parametrize(
        "ub",
        [1.0e3, 1.0e6, 1.0e8],
        ids=["ub_1e3", "ub_1e6", "ub_1e8"],
    )
    def test_psd_quadratic_solves_via_convex_fast_path(self, ub):
        m = Model("psd_quad_solver_wide")
        x = m.continuous("x", lb=-ub, ub=ub)
        y = m.continuous("y", lb=-ub, ub=ub)
        m.minimize(x * x + y * y + x * y)
        result = m.solve(time_limit=60.0)
        assert result.status == "optimal"
        assert result.convex_fast_path is True
        # Global optimum is 0 at (0, 0).
        assert abs(result.objective) <= 1.0e-6 + 1.0e-6 * ub**2

    def test_exp_perspective_solves_via_convex_fast_path(self):
        # Objective: minimise y * exp(x / y) + y^2 on a wide box.
        m = Model("persp_solver_wide")
        x = m.continuous("x", lb=-1.0e3, ub=1.0e3)
        y = m.continuous("y", lb=1.0e-2, ub=1.0e3)
        m.minimize(y * dm.exp(x / y) + y * y)
        result = m.solve(time_limit=60.0)
        assert result.status == "optimal"
        assert result.convex_fast_path is True
