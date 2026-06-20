"""Tests for automated constraint-chain cut derivation (issue #15 automation).

Drives the full pipeline end-to-end on the gas pipe+compressor chain: symbolic
elimination -> coupling beta >= sqrt(phi(w)) -> univariate underestimator h(w) of
the compressor power term -> soundness certification. Reproduces the cut that was
hand-derived to close the gas-network bound from 67% to 0%.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

sp = pytest.importorskip("sympy")

from discopt._jax.symbolic.constraint_cuts import (  # noqa: E402
    eliminate_chain_coupling,
    power_term_underestimator,
    verify_cut,
)

pytestmark = pytest.mark.relaxation

# Gas-network constants (issue #15 benchmark).
C0, C2 = 3.5, 5.0
PS1 = 50.0
PN5_2 = 45.0**2 + 40.0**2 / 3.5  # demand-forced p_n5^2 lower bound
KAPPA = 0.2857
K_POWER = 0.828


def _build_coupling():
    w, beta, p1, p2, p5 = sp.symbols("w beta p1 p2 p5", positive=True)
    eqs = [
        sp.Eq(w**2, C0 * (PS1**2 - p1**2)),  # Weymouth s1->n1
        sp.Eq(p2, beta * p1),  # compressor
        sp.Eq(w**2, C2 * (p2**2 - p5**2)),  # Weymouth n2->n5
    ]
    pn5 = sp.sqrt(sp.Float(PN5_2))
    return eliminate_chain_coupling(
        eqs,
        target=beta,
        keep=w,
        eliminate=[p1, p2],
        lower_bounds={p5: pn5},
        sample={w: 35.0, p5: 50.0},
    )


def test_elimination_reproduces_phi():
    """The derived coupling matches the hand-derived phi(w) numerically."""
    cpl = _build_coupling()
    w = cpl.keep
    phi_fn = sp.lambdify([w], cpl.target_lower**2, "numpy")
    # Hand-derived phi(w) = (PN5_2 + w^2/C2)/(PS1^2 - w^2/C0)
    for wv in (10.0, 35.0, 60.0):
        hand = (PN5_2 + wv**2 / C2) / (PS1**2 - wv**2 / C0)
        assert float(phi_fn(wv)) == pytest.approx(hand, rel=1e-9)


def test_coupling_tight_at_optimum():
    """beta >= sqrt(phi(w)) is tight at the known optimum (w=35, beta=1.1263)."""
    cpl = _build_coupling()
    bl = sp.lambdify([cpl.keep], cpl.target_lower, "numpy")
    assert float(bl(35.0)) == pytest.approx(1.1263, rel=1e-3)


def test_power_underestimator_is_convex_and_sound():
    """h(w) underestimates the compressor power term and is convex."""
    cpl = _build_coupling()
    under = power_term_underestimator(cpl, exponent=KAPPA, coefficient=K_POWER, domain=(0.0, 70.0))
    assert under.is_convex
    # h(35) should match the optimum per-compressor power ~1.0
    assert float(under.h_fn(35.0)) == pytest.approx(1.0, abs=0.05)

    # Soundness over the feasible manifold: term >= h(w).

    coupling_fn = sp.lambdify([cpl.keep], cpl.target_lower, "numpy")
    term_fn = lambda w, beta: K_POWER * w * (beta**KAPPA - 1.0)  # noqa: E731
    report = verify_cut(term_fn, under, coupling_fn, domain=(0.0, 70.0), target_max=2.0, n=4000)
    assert report["sound"], f"cut unsound: max_violation={report['max_violation']:.2e}"


def test_tangent_cuts_are_valid_underestimators():
    """Tangent lines of the convex h are global underestimators (sound cuts)."""
    cpl = _build_coupling()
    under = power_term_underestimator(cpl, exponent=KAPPA, coefficient=K_POWER, domain=(0.0, 70.0))
    import jax.numpy as jnp

    for wk in (15.0, 35.0, 55.0):
        v, slope = under.tangent_cut(wk)
        ws = jnp.linspace(0.0, 70.0, 100)
        h = under.h_fn(ws)
        tangent = v + slope * (ws - wk)
        assert bool(jnp.all(tangent <= h + 1e-6))  # tangent below h everywhere
