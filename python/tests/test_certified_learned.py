"""Tests for the certified-learned-envelope prototype (Phase 8).

Demonstrates the central thesis: a *constant* certified margin restores soundness
of a convex/concave relaxation **while preserving curvature**, unlike pointwise
clamping against ``f`` (which re-introduces non-convexity). The mechanism is
tested on a controlled convex base derived by the symbolic engine, and the
learned-relaxation adapter is smoke-tested when ``equinox`` is available.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

sp = pytest.importorskip("sympy")

from discopt._jax.symbolic import (  # noqa: E402
    derive_envelope,
    lambdify_envelope,
    verify_envelope,
)
from discopt._jax.symbolic.certified_learned import (  # noqa: E402
    certify_relaxation,
)

pytestmark = pytest.mark.relaxation

X = sp.Symbol("x", real=True)


def _convex_but_unsound_base(bias: float):
    """Exact x^2 envelope shifted by a constant so cv > f and cc < f (unsound)."""
    exact = lambdify_envelope(derive_envelope(X**2, X))

    def raw_fn(x, lb, ub):
        cv, cc = exact(x, lb, ub)
        # Constant shift preserves convexity of cv and concavity of cc.
        return cv + bias, cc - bias

    return raw_fn


def test_raw_base_is_convex_but_unsound():
    raw_fn = _convex_but_unsound_base(bias=0.7)
    report = verify_envelope(raw_fn, lambda x: x**2, domain=(-2.0, 2.0), n_boxes=300, seed=5)
    assert not report.sound  # the constant shift breaks containment
    # ...but curvature is intact (convex cv, concave cc).
    assert report.max_convexity_violation <= 1e-6
    assert report.max_concavity_violation <= 1e-6


def test_certification_restores_soundness_and_preserves_curvature():
    raw_fn = _convex_but_unsound_base(bias=0.7)
    cert = certify_relaxation(
        raw_fn, lambda x: x**2, domain=(-2.0, 2.0), n_boxes=300, seed=5, safety_factor=1.2
    )
    rep = cert.certified_report
    assert rep.sound, (
        f"certified relaxation still violates containment: "
        f"lower={rep.max_lower_violation:.3e} upper={rep.max_upper_violation:.3e}"
    )
    # Curvature is preserved by the constant margin (the whole point).
    assert rep.max_convexity_violation <= 1e-6
    assert rep.max_concavity_violation <= 1e-6
    # Margin brackets the injected bias (>= bias, and not wildly larger).
    assert cert.lower_margin >= 0.7 - 1e-6
    assert cert.upper_margin >= 0.7 - 1e-6


def test_margin_is_constant_in_x_so_curvature_cannot_change():
    """A margin constant in x cannot alter convexity: cv-δ is convex iff cv is."""
    raw_fn = _convex_but_unsound_base(bias=0.5)
    cert = certify_relaxation(raw_fn, lambda x: x**2, domain=(-2.0, 2.0), n_boxes=200, seed=3)
    # Same convexity violation before and after the shift (up to sampling noise).
    assert (
        abs(cert.raw_report.max_convexity_violation - cert.certified_report.max_convexity_violation)
        <= 1e-6
    )


def test_clamping_against_f_destroys_convexity():
    """Contrast: pointwise min/max clamp against f is sound but non-convex."""
    f = lambda x: x**2  # noqa: E731

    def clamped(x, lb, ub):
        # A flat convex "prediction" cv_pred = 1.0 that exceeds f near 0; the
        # existing learned wrapper would clamp cv = min(cv_pred, f). min(const, x^2)
        # is sound but has a concave kink -> non-convex.
        cv = jnp.minimum(jnp.ones_like(x), f(x))
        cc = jnp.maximum(-jnp.ones_like(x), f(x))
        return cv, cc

    rep = verify_envelope(clamped, f, domain=(-2.0, 2.0), n_boxes=300, seed=5)
    assert rep.sound  # clamping bounds pointwise
    # ...but min(1, x^2) is non-convex: a positive convexity violation appears.
    assert rep.max_convexity_violation > 1e-3


# --------------------------------------------------------------------------
# Learned-relaxation adapter (requires equinox)
# --------------------------------------------------------------------------


def test_learned_adapter_certifies_to_sound():
    """certify_relaxation makes an (untrained) ICNN relaxation sound on fresh boxes."""
    pytest.importorskip("equinox")
    from discopt._jax.learned_relaxations import create_learned_relaxation
    from discopt._jax.symbolic.certified_learned import raw_learned_relax_fn

    lr = create_learned_relaxation(jax.random.PRNGKey(0), "square")
    raw_fn = raw_learned_relax_fn(lr)
    cert = certify_relaxation(
        raw_fn, lambda x: x**2, domain=(-2.0, 2.0), n_boxes=200, seed=2, safety_factor=1.5
    )
    assert cert.certified_report.sound
    # Regression guard: the ICNN must be convex-in-x by construction (its output
    # layer was previously unconstrained, breaking the guarantee). cv stays convex
    # and cc stays concave under the constant margin.
    assert cert.certified_report.max_convexity_violation <= 1e-6
    assert cert.certified_report.max_concavity_violation <= 1e-6
