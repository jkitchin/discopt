"""Tests for the §4 signomial overestimator (item 4) and §5 products/ratios
recognition (item 5).

Item 4: the tighter per-monomial *secant* overestimator for signed
signomials — validity (``cv ≤ s ≤ cc`` on the box) and that it never loosens
(and strictly tightens on a single monomial) relative to the constant
corner-maximum.

Item 5: the products/ratios recognizer scoping the general detector to the
KMS §5 structural class — structural gate + sound G-convexity verdicts on
products/ratios that are neither convex nor concave.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.convexity.g_products_ratios import (
    classify_product_ratio,
    is_product_or_ratio,
)
from discopt._jax.symbolic.signed_signomial import (
    _posynomial_parts,
    signed_signomial_dc_envelope,
)
from discopt.modeling.core import Model

# ──────────────────────────────────────────────────────────────────────
# Item 4 — §4 secant overestimator
# ──────────────────────────────────────────────────────────────────────


def _s(u, terms):
    pp, pm = _posynomial_parts(u, terms)
    return float(pp - pm)


class TestSignomialSecantOverestimator:
    terms = [
        (1.0, 0.0, np.array([1.0, 0.0])),  # +exp(u0)
        (1.0, np.log(2.0), np.array([0.5, 1.0])),  # +2 exp(0.5 u0 + u1)
        (-1.0, 0.0, np.array([0.0, -1.0])),  # -exp(-u1)
    ]
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])

    def test_secant_envelope_is_valid(self):
        # Vectorized over the 2000-point fuzz: the signomial value and the DC
        # envelope are ``jax.vmap``'d across the whole sample block in one
        # dispatch each. ``rng.uniform`` fills the block row-major, so these are
        # the identical points the per-point loop drew — same validity assertion.
        rng = np.random.default_rng(1)
        u = jnp.asarray(rng.uniform(self.lb, self.ub, size=(2000, self.lb.shape[0])))
        pp, pm = jax.vmap(lambda uu: _posynomial_parts(uu, self.terms))(u)
        sv = np.asarray(pp - pm)
        cv, cc = (
            np.asarray(v)
            for v in jax.vmap(
                lambda uu: signed_signomial_dc_envelope(
                    uu, self.terms, self.lb, self.ub, overestimator="secant"
                )
            )(u)
        )
        assert np.all(cv - 1e-9 <= sv) and np.all(sv <= cc + 1e-9)

    def test_secant_never_looser_than_corner(self):
        rng = np.random.default_rng(2)
        u = jnp.asarray(rng.uniform(self.lb, self.ub, size=(2000, self.lb.shape[0])))
        cvc, ccc = (
            np.asarray(v)
            for v in jax.vmap(
                lambda uu: signed_signomial_dc_envelope(
                    uu, self.terms, self.lb, self.ub, overestimator="corner"
                )
            )(u)
        )
        cvs, ccs = (
            np.asarray(v)
            for v in jax.vmap(
                lambda uu: signed_signomial_dc_envelope(
                    uu, self.terms, self.lb, self.ub, overestimator="secant"
                )
            )(u)
        )
        # secant tightens: cc no larger, cv no smaller
        assert np.all(ccs <= ccc + 1e-9)
        assert np.all(cvs >= cvc - 1e-9)

    def test_single_monomial_strictly_tighter_interior(self):
        terms = [(1.0, 0.0, np.array([1.0]))]  # exp(u)
        lb, ub = np.array([-1.0]), np.array([1.0])
        u = np.array([0.0])  # interior
        _, ccc = signed_signomial_dc_envelope(u, terms, lb, ub, overestimator="corner")
        _, ccs = signed_signomial_dc_envelope(u, terms, lb, ub, overestimator="secant")
        # corner = constant max = exp(1); secant = chord value < exp(1); both ≥ exp(0)
        assert float(ccs) < float(ccc)
        assert float(ccs) >= np.exp(0.0) - 1e-12

    def test_invalid_overestimator_rejected(self):
        with pytest.raises(ValueError):
            signed_signomial_dc_envelope(
                self.ub, self.terms, self.lb, self.ub, overestimator="bogus"
            )

    def test_corner_default_unchanged(self):
        # Default path stays the constant corner-max (bound-neutral).
        u = np.array([0.3, -0.2])
        cv_def, cc_def = signed_signomial_dc_envelope(u, self.terms, self.lb, self.ub)
        cv_c, cc_c = signed_signomial_dc_envelope(
            u, self.terms, self.lb, self.ub, overestimator="corner"
        )
        assert float(cv_def) == float(cv_c)
        assert float(cc_def) == float(cc_c)


# ──────────────────────────────────────────────────────────────────────
# Item 5 — products & ratios recognition
# ──────────────────────────────────────────────────────────────────────


class TestProductsRatios:
    def test_structural_gate(self):
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=2.0)
        y = m.continuous("y", lb=1.0, ub=2.0)
        assert is_product_or_ratio(x * y) == "product"
        assert is_product_or_ratio(x / y) == "ratio"
        assert is_product_or_ratio(x + y) is None
        assert is_product_or_ratio(x**2) is None

    def test_non_product_returns_none(self):
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=2.0)
        assert classify_product_ratio(x + x, m) is None

    def test_bilinear_product_is_g_concave(self):
        # x*y on the positive orthant near (1,1): neither convex nor concave,
        # but G-concave (log(xy)=log x+log y concave).
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=1.1)
        y = m.continuous("y", lb=1.0, ub=1.1)
        c = classify_product_ratio(x * y, m)
        assert c is not None and c.kind == "product"
        assert c.certificate is not None and c.certificate.kind == "g_concave"

    def test_product_of_convex_recognized(self):
        # x²·y² (product of two convex factors) is a §5 form and G-recognized.
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=1.05)
        y = m.continuous("y", lb=1.0, ub=1.05)
        c = classify_product_ratio((x**2) * (y**2), m)
        assert c is not None
        assert c.left_curvature.value == "convex"
        assert c.right_curvature.value == "convex"
        assert c.certificate is not None  # sound verdict (g_convex or g_concave)

    def test_convex_ratio_is_g_convex_superset(self):
        # x²/y is convex ⇒ must be recognized g_convex with ρ=0 (superset).
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=1.1)
        y = m.continuous("y", lb=1.0, ub=1.1)
        c = classify_product_ratio((x**2) / y, m)
        assert c is not None and c.kind == "ratio"
        assert c.certificate is not None
        assert c.certificate.kind == "g_convex" and c.certificate.rho == 0.0

    def test_abstention_is_none_not_false(self):
        # A ratio the detector can't certify on a wide box abstains (None),
        # never a false claim.
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=5.0)
        y = m.continuous("y", lb=0.5, ub=5.0)
        c = classify_product_ratio(x / y, m)
        assert c is not None and c.kind == "ratio"
        assert c.certificate is None  # honest abstention
