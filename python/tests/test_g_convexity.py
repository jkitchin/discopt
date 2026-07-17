"""Tests for the box-local G-convexity (convex-transformability) detector.

Covers the enabling primitive of issue #181 (Khajavirad–Michalek–Sahinidis
2012, item 1): the augmented-Hessian test that recognizes G-convex
intermediates — a strictly larger class than DCP/eigenvalue convexity.

Three layers, mirroring ``test_convexity_certificate.py``:

1. **Correctness of the pointwise linear algebra** — the float diagnostic
   ``is_g_convex_pointwise`` matches the "``∇²φ`` PSD on ``∇φ^⊥``"
   definition on hand-worked matrices.
2. **The sound box certificate** — certifies genuinely G-convex (but not
   convex) bodies, degrades to ordinary convex/concave with ``ρ=0``, and
   abstains (returns ``None``) on indefinite bodies.
3. **Soundness (the non-negotiable guard)** — every ``g_convex`` /
   ``g_concave`` verdict is independently re-checked at random points of
   the box with exact ``eigvalsh``: the witnessed augmented Hessian must
   actually be PSD / NSD there. A false positive here would be a
   certificate that lies, the one thing this layer may never do.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.convexity import (
    Curvature,
    GConvexCertificate,
    certify_convex,
    certify_g_convex,
    is_g_convex_pointwise,
    least_convexifying_rho,
)
from discopt._jax.convexity.interval import Interval
from discopt._jax.convexity.interval_ad import interval_hessian
from discopt.modeling.core import Model

# ──────────────────────────────────────────────────────────────────────
# 1. Pointwise linear algebra
# ──────────────────────────────────────────────────────────────────────


class TestPointwise:
    def test_psd_is_g_convex_with_rho_zero(self):
        H = np.array([[2.0, 0.0], [0.0, 3.0]])
        g = np.array([1.0, -1.0])
        assert is_g_convex_pointwise(H, g)
        assert least_convexifying_rho(H, g) == pytest.approx(0.0, abs=1e-9)

    def test_indefinite_saddle_not_g_convex(self):
        # x*y at (1.5, 1.2): Hessian [[0,1],[1,0]], gradient (1.2, 1.5).
        # Restricted to g^⊥ the quadratic form is negative -> not G-convex.
        H = np.array([[0.0, 1.0], [1.0, 0.0]])
        g = np.array([1.2, 1.5])
        assert not is_g_convex_pointwise(H, g)
        assert least_convexifying_rho(H, g) is None

    def test_log_sum_squares_is_g_convex_not_convex(self):
        # φ = log(x²+y²): ∇²φ + ∇φ∇φᵀ = 2I/(x²+y²) ≻ 0, so G-convex with
        # ρ=1, yet ∇²φ itself is indefinite (not convex).
        x, y = 1.4, 1.7
        s = x * x + y * y
        g = np.array([2 * x / s, 2 * y / s])
        H = np.array(
            [
                [2 / s - 4 * x * x / s**2, -4 * x * y / s**2],
                [-4 * x * y / s**2, 2 / s - 4 * y * y / s**2],
            ]
        )
        assert np.linalg.eigvalsh(H).min() < 0.0  # not convex
        assert is_g_convex_pointwise(H, g)  # but G-convex
        rho = least_convexifying_rho(H, g)
        assert rho is not None and rho > 0.0
        # Witness: H + rho·ggᵀ is PSD (allow tiny bisection slack).
        assert np.linalg.eigvalsh(H + rho * np.outer(g, g)).min() >= -1e-7

    def test_pseudolinear_linear_fractional(self):
        # φ = x/y is pseudolinear: PSD on g^⊥ with the restricted form == 0.
        x, y = 2.0, 1.0
        g = np.array([1 / y, -x / y**2])
        H = np.array([[0.0, -1 / y**2], [-1 / y**2, 2 * x / y**3]])
        assert is_g_convex_pointwise(H, g)

    def test_zero_gradient_reduces_to_ordinary_psd(self):
        assert is_g_convex_pointwise(np.eye(2), np.zeros(2))
        assert not is_g_convex_pointwise(np.diag([1.0, -1.0]), np.zeros(2))


# ──────────────────────────────────────────────────────────────────────
# 2. Sound box certificate
# ──────────────────────────────────────────────────────────────────────


class TestBoxCertificate:
    def test_convex_body_reports_g_convex_rho_zero(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        cert = certify_g_convex(x**2, m)
        assert cert == GConvexCertificate("g_convex", 0.0)
        assert cert.is_convex and not cert.strictly_transformed

    def test_concave_body_reports_g_concave_rho_zero(self):
        # log is ordinarily concave: the natural verdict is g_concave, ρ=0,
        # NOT an escalated transformed g_convex.
        m = Model("t")
        b = m.continuous("b", lb=0.5, ub=2.0)
        cert = certify_g_convex(dm.log(b), m)
        assert cert == GConvexCertificate("g_concave", 0.0)

    def test_g_convex_not_convex_is_certified(self):
        # log(x²+y²) on a tight box: G-convex with ρ≈1, though not convex.
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        phi = dm.log(x**2 + y**2)
        # Ordinary convexity certificate abstains (body is not convex).
        assert certify_convex(phi, m) is None
        cert = certify_g_convex(phi, m)
        assert cert is not None
        assert cert.kind == "g_convex"
        assert cert.strictly_transformed  # ρ > 0 — genuinely transformed

    def test_indefinite_bilinear_abstains(self):
        m = Model("t")
        c = m.continuous("c", lb=1.0, ub=2.0)
        d = m.continuous("d", lb=1.0, ub=2.0)
        assert certify_g_convex(c * d, m) is None

    def test_wide_box_abstains_never_false_positive(self):
        # On a wide box the interval outer-product slack swamps the diagonal
        # margin; the certificate must abstain, never claim falsely.
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=3.0)
        y = m.continuous("y", lb=1.0, ub=3.0)
        cert = certify_g_convex(dm.log(x**2 + y**2), m)
        assert cert is None  # abstention, not a false claim

    def test_box_override_tightens(self):
        # A tightened box (as from FBBT/branching) can certify where the
        # declared-bounds box abstains.
        m = Model("t")
        x = m.continuous("x", lb=1.0, ub=3.0)
        y = m.continuous("y", lb=1.0, ub=3.0)
        assert certify_g_convex(dm.log(x**2 + y**2), m) is None
        tight = {x: Interval.from_bounds(1.4, 1.5), y: Interval.from_bounds(1.6, 1.7)}
        assert certify_g_convex(dm.log(x**2 + y**2), m, box=tight) is not None

    def test_g_convex_superset_of_convex(self):
        # Every body the ordinary certificate proves convex must also be
        # G-convex (with ρ=0) — G-convexity is a superset of convexity.
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        for expr in (x**2, dm.exp(x), x**2 + dm.exp(x)):
            if certify_convex(expr, m) == Curvature.CONVEX:
                cert = certify_g_convex(expr, m)
                assert cert is not None and cert.kind == "g_convex"
                assert cert.rho == 0.0


# ──────────────────────────────────────────────────────────────────────
# 3. Soundness — the non-negotiable guard
# ──────────────────────────────────────────────────────────────────────


def _sample_augmented_psd_ok(expr, model, box, cert, *, seed=0, n_samples=64):
    """Re-check the witnessed augmented Hessian at random box points.

    Uses exact float ``eigvalsh`` (independent of the interval machinery)
    to confirm ``∇²φ(x) + ρ∇φ(x)∇φ(x)ᵀ`` is PSD (for ``g_convex``) or that
    ``∇²φ(x) - ρ∇φ(x)∇φ(x)ᵀ`` is NSD (for ``g_concave``) at every sample.
    A violation means the certificate lied.
    """
    rng = np.random.default_rng(seed)
    lbs, ubs = [], []
    for v in model._variables:
        lo = np.asarray(box[v].lo, dtype=float).ravel()
        hi = np.asarray(box[v].hi, dtype=float).ravel()
        lbs.append(lo)
        ubs.append(hi)
    lb = np.concatenate(lbs)
    ub = np.concatenate(ubs)
    for _ in range(n_samples):
        pt = rng.uniform(lb, ub)
        pbox = {}
        off = 0
        for v in model._variables:
            sz = v.size
            pbox[v] = Interval.from_bounds(pt[off : off + sz], pt[off : off + sz])
            off += sz
        ad = interval_hessian(expr, model, box=pbox)
        H = 0.5 * (ad.hess.mid + ad.hess.mid.T)
        g = ad.grad.mid
        rank1 = np.outer(g, g)
        if cert.kind == "g_convex":
            A = H + cert.rho * rank1
            assert np.linalg.eigvalsh(0.5 * (A + A.T)).min() >= -1e-6, (
                f"certificate CLAIMED g_convex but augmented Hessian not PSD at {pt}"
            )
        else:
            A = H - cert.rho * rank1
            assert np.linalg.eigvalsh(0.5 * (A + A.T)).max() <= 1e-6, (
                f"certificate CLAIMED g_concave but augmented Hessian not NSD at {pt}"
            )


class TestSoundness:
    def _box_of(self, model):
        return {
            v: Interval.from_bounds(np.asarray(v.lb).ravel(), np.asarray(v.ub).ravel())
            for v in model._variables
        }

    def test_g_convex_verdict_holds_at_samples(self):
        m = Model("t")
        x = m.continuous("x", lb=1.4, ub=1.5)
        y = m.continuous("y", lb=1.6, ub=1.7)
        phi = dm.log(x**2 + y**2)
        cert = certify_g_convex(phi, m)
        assert cert is not None
        _sample_augmented_psd_ok(phi, m, self._box_of(m), cert)

    def test_convex_verdict_holds_at_samples(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        phi = x**2
        cert = certify_g_convex(phi, m)
        _sample_augmented_psd_ok(phi, m, self._box_of(m), cert)

    def test_concave_verdict_holds_at_samples(self):
        m = Model("t")
        b = m.continuous("b", lb=0.5, ub=2.0)
        phi = dm.log(b)
        cert = certify_g_convex(phi, m)
        _sample_augmented_psd_ok(phi, m, self._box_of(m), cert)

    @pytest.mark.parametrize("seed", range(6))
    def test_never_false_positive_on_random_indefinite(self, seed):
        # Random bilinear-ish bodies that are generically indefinite: any
        # verdict returned must survive the independent augmented-PSD check.
        rng = np.random.default_rng(seed)
        m = Model("t")
        x = m.continuous("x", lb=rng.uniform(0.5, 1.0), ub=rng.uniform(1.5, 2.0))
        y = m.continuous("y", lb=rng.uniform(0.5, 1.0), ub=rng.uniform(1.5, 2.0))
        a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
        phi = a * x * y + b * x
        cert = certify_g_convex(phi, m)
        if cert is not None:
            _sample_augmented_psd_ok(phi, m, self._box_of(m), cert, seed=seed)
