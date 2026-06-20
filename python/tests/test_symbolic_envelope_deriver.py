"""Theorem-style tests for the SymPy envelope-derivation engine (Phase 1).

These validate :mod:`discopt._jax.symbolic` against the same discipline as the
hand-written relaxations: every derived envelope must be *sound*
(``cv <= f <= cc``), correctly *curved* (``cv`` convex, ``cc`` concave), and —
for the McCormick-exact convex/concave atoms — must reproduce the existing
``discopt._jax.mccormick`` primitives bit-for-bit.

The engine itself imports SymPy (a design-time ``[sympy]`` extra); the whole
module is skipped if SymPy is unavailable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sp = pytest.importorskip("sympy")

from discopt._jax.symbolic import (  # noqa: E402
    Curvature,
    derive_envelope,
    lambdify_envelope,
    verify_envelope,
)
from discopt._jax.symbolic.envelope_deriver import (  # noqa: E402
    EnvelopeDerivationError,
)

pytestmark = pytest.mark.relaxation

X = sp.Symbol("x", real=True)


# --------------------------------------------------------------------------
# Curvature classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, expected",
    [
        (X**2, Curvature.CONVEX),
        (sp.exp(X), Curvature.CONVEX),
        (sp.log(X), Curvature.CONCAVE),
        (sp.sqrt(X), Curvature.CONCAVE),
        (X**3, Curvature.CONCAVO_CONVEX),
        (-(X**3), Curvature.CONVEXO_CONCAVE),
        (X * sp.Abs(X), Curvature.CONCAVO_CONVEX),  # Weymouth f|f|
        (X**5, Curvature.CONCAVO_CONVEX),
    ],
)
def test_curvature_classification(expr, expected):
    r = derive_envelope(expr, X)
    assert r.curvature == expected


def test_weymouth_kink_not_misclassified_as_convex():
    """f|f| must be CONCAVO_CONVEX, not silently CONVEX (a soundness trap)."""
    r = derive_envelope(X * sp.Abs(X), X)
    assert r.curvature == Curvature.CONCAVO_CONVEX
    assert float(r.inflection) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Soundness + curvature certification over randomized boxes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, domain",
    [
        (X**2, (-3.0, 3.0)),
        (sp.exp(X), (-2.0, 2.0)),
        (sp.log(X), (0.1, 5.0)),
        (sp.sqrt(X), (0.01, 5.0)),
        (X**3, (-2.0, 2.0)),
        (-(X**3), (-2.0, 2.0)),
        (X * sp.Abs(X), (-3.0, 3.0)),
        (X**5, (-1.5, 1.5)),
    ],
)
def test_envelope_is_sound_and_correctly_curved(expr, domain):
    r = derive_envelope(expr, X)
    fn = lambdify_envelope(r)
    f_num = sp.lambdify([X], expr, "jax")
    report = verify_envelope(fn, f_num, domain=domain, n_boxes=400, seed=7)
    assert report.sound, (
        f"containment violated: lower={report.max_lower_violation:.2e} "
        f"upper={report.max_upper_violation:.2e}"
    )
    assert report.max_convexity_violation <= 1e-6
    assert report.max_concavity_violation <= 1e-6


# --------------------------------------------------------------------------
# Parity with the existing hand-written McCormick primitives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, mc_name, domain",
    [
        (X**2, "relax_square", (-3.0, 3.0)),
        (sp.exp(X), "relax_exp", (-2.0, 2.0)),
        (sp.log(X), "relax_log", (0.1, 5.0)),
        (sp.sqrt(X), "relax_sqrt", (0.05, 5.0)),
    ],
)
def test_matches_handwritten_mccormick(expr, mc_name, domain):
    from discopt._jax import mccormick

    mc_fn = getattr(mccormick, mc_name)
    sym_fn = lambdify_envelope(derive_envelope(expr, X))

    lo, hi = domain
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = sorted(rng.uniform(lo, hi, size=2))
        if b - a < 1e-3:
            b = min(a + 1e-3, hi)
        xs = jnp.linspace(a, b, 25)
        for x in xs:
            cv_s, cc_s = sym_fn(x, a, b)
            cv_m, cc_m = mc_fn(x, a, b)
            assert float(cv_s) == pytest.approx(float(cv_m), abs=1e-9, rel=1e-7)
            assert float(cc_s) == pytest.approx(float(cc_m), abs=1e-9, rel=1e-7)


# --------------------------------------------------------------------------
# Tangent points match the known closed forms
# --------------------------------------------------------------------------


def test_cubic_tangent_points_closed_form():
    """For x^3, the convex-envelope tangent point from a is -a/2."""
    r = derive_envelope(X**3, X)
    a = r.lower
    assert sp.simplify(r.cv_tangent.point - (-a / 2)) == 0


def test_weymouth_tangent_points_closed_form():
    """For x|x|, both tangent points are e*(1 - sqrt(2))."""
    r = derive_envelope(X * sp.Abs(X), X)
    a, b = r.lower, r.upper
    assert sp.simplify(r.cv_tangent.point - a * (1 - sp.sqrt(2))) == 0
    assert sp.simplify(r.cc_tangent.point - b * (1 - sp.sqrt(2))) == 0


# --------------------------------------------------------------------------
# Boxes that do not straddle the inflection reduce to pure curvature
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a, b", [(0.5, 3.0), (-3.0, -0.5)])
def test_single_inflection_one_sided_box_is_sound(a, b):
    """x^3 on a box entirely on one side of 0 must still be sound."""
    fn = lambdify_envelope(derive_envelope(X**3, X))
    xs = jnp.linspace(a, b, 60)
    for x in xs:
        cv, cc = fn(x, a, b)
        assert float(cv) <= float(x**3) + 1e-9
        assert float(cc) >= float(x**3) - 1e-9


# --------------------------------------------------------------------------
# Graceful rejection of out-of-scope cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, domain",
    [
        (1 / (1 + sp.exp(-X)), (-5.0, 5.0)),  # sigmoid (convexo-concave, numeric)
        (sp.tanh(X), (-4.0, 4.0)),
        (sp.atan(X), (-5.0, 5.0)),
        (X * sp.Abs(X) ** sp.Rational(85, 100), (-6.0, 9.0)),  # Panhandle (concavo-convex)
    ],
)
def test_numeric_tangent_solver_is_sound(expr, domain):
    """Transcendental single-inflection atoms use the JAX bisection fallback."""
    r = derive_envelope(expr, X)
    assert r.is_single_inflection
    assert r.cv_tangent.point is None  # no closed form -> numeric path
    fn = lambdify_envelope(r)
    f_num = sp.lambdify([X], expr, "jax")
    report = verify_envelope(fn, f_num, domain=domain, n_boxes=500, seed=11)
    assert report.sound
    assert report.max_convexity_violation <= 1e-6
    assert report.max_concavity_violation <= 1e-6


def test_asymmetric_box_falls_back_to_secant():
    """When the tangent point exits its branch, the secant is the envelope."""
    fn = lambdify_envelope(derive_envelope(X**3, X))
    f_num = sp.lambdify([X], X**3, "jax")
    report = verify_envelope(fn, f_num, domain=(-10.0, 1.0), n_boxes=300, seed=2)
    assert report.sound


def test_diracdelta_derivative_rejected():
    """sign(x)*x^2 has a DiracDelta gradient; engine rejects with guidance."""
    with pytest.raises(EnvelopeDerivationError):
        derive_envelope(sp.sign(X) * X**2, X)


def test_jit_and_vmap_compatible():
    """Generated closures must survive jit + vmap (single compiled B&B module)."""
    fn = lambdify_envelope(derive_envelope(X * sp.Abs(X), X))
    jfn = jax.jit(jax.vmap(fn, in_axes=(0, None, None)))
    xs = jnp.linspace(-2.0, 2.0, 32)
    cv, cc = jfn(xs, -2.0, 2.0)
    fx = xs * jnp.abs(xs)
    assert jnp.all(cv <= fx + 1e-7)
    assert jnp.all(cc >= fx - 1e-7)
