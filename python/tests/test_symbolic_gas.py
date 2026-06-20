"""Tests for the gas-network envelope pack (Phase 3): Weymouth ``f|f|``.

Validates that (1) the hand-written ``weymouth_relax`` closure is sound and
tighter than the generic ``f * |f|`` bilinear decomposition, (2) it agrees with
the SymPy-derived envelope (certifying the committed JAX), and (3) the relaxation
compiler routes ``f * abs(f)`` to the tight envelope end-to-end.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from discopt._jax.symbolic.domains.gas import weymouth_relax  # noqa: E402

pytestmark = pytest.mark.relaxation


def _f(x):
    return x * jnp.abs(x)


# --------------------------------------------------------------------------
# Soundness + curvature over realistic flow ranges
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lb, ub", [(-50.0, 50.0), (-10.0, 30.0), (-5.0, 0.0), (0.0, 20.0)])
def test_weymouth_sound(lb, ub):
    xs = jnp.linspace(lb, ub, 200)
    cv, cc = jax.vmap(weymouth_relax, in_axes=(0, None, None))(xs, lb, ub)
    fx = _f(xs)
    assert jnp.all(cv <= fx + 1e-7), f"cv overshoots f on [{lb},{ub}]"
    assert jnp.all(cc >= fx - 1e-7), f"cc undershoots f on [{lb},{ub}]"


def test_weymouth_convex_concave():
    """cv convex, cc concave over a straddling box (Jensen on random chords)."""
    lb, ub = -8.0, 12.0
    rng = np.random.default_rng(0)
    x1 = jnp.asarray(rng.uniform(lb, ub, 500))
    x2 = jnp.asarray(rng.uniform(lb, ub, 500))
    xm = 0.5 * (x1 + x2)
    relax = jax.vmap(weymouth_relax, in_axes=(0, None, None))
    cv1, cc1 = relax(x1, lb, ub)
    cv2, cc2 = relax(x2, lb, ub)
    cvm, ccm = relax(xm, lb, ub)
    assert jnp.all(cvm <= 0.5 * (cv1 + cv2) + 1e-7)
    assert jnp.all(ccm >= 0.5 * (cc1 + cc2) - 1e-7)


def test_weymouth_tighter_than_bilinear():
    """The dedicated envelope must be no looser than relax_bilinear(f, |f|)."""
    from discopt._jax.mccormick import relax_abs, relax_bilinear

    lb, ub = -10.0, 10.0
    xs = jnp.linspace(lb, ub, 101)

    def bilinear_gap(x):
        # |f| envelope over the box, then bilinear of f and |f|.
        abs_lb, abs_ub = 0.0, max(abs(lb), abs(ub))
        _, _ = relax_abs(x, lb, ub)
        cv, cc = relax_bilinear(x, jnp.abs(x), lb, ub, abs_lb, abs_ub)
        return cc - cv

    wgap = jax.vmap(lambda x: (lambda c: c[1] - c[0])(weymouth_relax(x, lb, ub)))(xs)
    bgap = jax.vmap(bilinear_gap)(xs)
    # Mean tight-envelope gap is strictly smaller than the bilinear gap.
    assert float(jnp.mean(wgap)) < float(jnp.mean(bgap))


# --------------------------------------------------------------------------
# Certification: hand-written closure == SymPy-derived envelope
# --------------------------------------------------------------------------


def test_handwritten_matches_symbolic():
    pytest.importorskip("sympy")
    from discopt._jax.symbolic.domains.gas import derive_weymouth_symbolic

    sym_fn = derive_weymouth_symbolic()
    rng = np.random.default_rng(1)
    for _ in range(40):
        lb, ub = sorted(rng.uniform(-20, 20, size=2))
        if ub - lb < 1e-2:
            ub = lb + 1e-2
        for x in jnp.linspace(lb, ub, 25):
            cv_h, cc_h = weymouth_relax(x, lb, ub)
            cv_s, cc_s = sym_fn(x, lb, ub)
            assert float(cv_h) == pytest.approx(float(cv_s), abs=1e-9, rel=1e-7)
            assert float(cc_h) == pytest.approx(float(cc_s), abs=1e-9, rel=1e-7)


# --------------------------------------------------------------------------
# End-to-end: compiler routes f*abs(f) to the tight envelope
# --------------------------------------------------------------------------


def test_compiler_routes_weymouth_pattern():
    from discopt._jax.relaxation_compiler import compile_relaxation
    from discopt.modeling.core import Model

    m = Model("gas")
    f = m.continuous("f", lb=-10.0, ub=10.0)
    m.minimize(f)
    expr = f * abs(f)  # Weymouth pressure-drop term

    relax_fn = compile_relaxation(expr, m)
    lb = jnp.array([-10.0])
    ub = jnp.array([10.0])
    xs = jnp.linspace(-10.0, 10.0, 60)
    for xv in xs:
        x = jnp.array([float(xv)])
        cv, cc = relax_fn(x, x, lb, ub)
        true = float(xv) * abs(float(xv))
        assert float(cv) <= true + 1e-6
        assert float(cc) >= true - 1e-6
        # Tight envelope matches the standalone closure (i.e. routing happened).
        cv_w, cc_w = weymouth_relax(xv, -10.0, 10.0)
        assert float(cv) == pytest.approx(float(cv_w), abs=1e-7)
        assert float(cc) == pytest.approx(float(cc_w), abs=1e-7)


# --------------------------------------------------------------------------
# Generalized signed-power flow term (Panhandle exponent), design-time
# --------------------------------------------------------------------------


def test_signed_power_panhandle_sound():
    pytest.importorskip("sympy")
    from discopt._jax.symbolic.domains.gas import derive_signed_power_symbolic
    from discopt._jax.symbolic.envelope_deriver import EnvelopeDerivationError

    try:
        fn = derive_signed_power_symbolic(1.85)
    except EnvelopeDerivationError:
        pytest.skip("closed-form tangent unavailable for beta=1.85")
    lb, ub = -6.0, 9.0
    xs = jnp.linspace(lb, ub, 150)
    f = xs * jnp.abs(xs) ** 0.85
    cv, cc = jax.vmap(fn, in_axes=(0, None, None))(xs, lb, ub)
    assert jnp.all(cv <= f + 1e-6)
    assert jnp.all(cc >= f - 1e-6)
