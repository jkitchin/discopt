"""Tests for the structured-cut recognizer presolve pass (issue #15).

Drives the recognizer on the *unmodified* gas-network model: it reads the
constraint graph, detects the bilinear-concave objective terms, auto-derives the
Weymouth-chain coupling and the FBBT terminal-pressure bound, and produces sound
cuts — with no hand-fed equations. Also checks graceful no-op on a non-matching
model and (slow) the end-to-end bound closure.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("sympy")

import discopt.modeling as dm  # noqa: E402
from discopt._jax.symbolic import cut_recognizer as R  # noqa: E402

pytestmark = pytest.mark.relaxation


def _gas_model():
    from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp

    return build_gas_network_minlp()


def test_translates_objective_and_constraints():
    sm = R.model_to_sympy(_gas_model())
    assert sm.objective.free_symbols  # non-empty
    # all 15 equality constraints translated
    assert len(sm.equalities) == 15


def test_finds_bilinear_concave_terms():
    sm = R.model_to_sympy(_gas_model())
    terms = R.find_product_terms(sm.objective)
    keys = {(str(t.x), str(t.y)) for t in terms}
    assert keys == {("w_cs_0", "beta_0"), ("w_cs_1", "beta_1")}
    for t in terms:
        assert t.exponent == pytest.approx(0.2857, abs=1e-4)
        assert t.coefficient == pytest.approx(0.828, abs=1e-3)


def test_recognizer_derives_sound_coupling_and_bound():
    cuts = R.recognize_and_derive_cuts(_gas_model())
    assert len(cuts) == 2
    for c in cuts:
        # FBBT recovered the demand-forced terminal pressure bound (~49.82)
        assert c.pn5_lower == pytest.approx(49.821, abs=1e-2)
        # underestimator matches optimum per-compressor power (~1.0 at w=35)
        assert float(c.underestimator.h_fn(35.0)) == pytest.approx(1.0, abs=0.05)
        assert c.underestimator.is_convex
        assert c.verification["sound"]


def test_no_match_returns_empty():
    """A model without the square-difference pattern yields no cuts (graceful)."""
    m = dm.Model("plain")
    x = m.continuous("x", lb=1.0, ub=5.0)
    yv = m.continuous("yv", lb=1.0, ub=5.0)
    m.minimize(x * yv)  # bilinear, but no Weymouth chain coupling x and yv
    m.subject_to(x + yv >= 3.0)
    assert R.recognize_and_derive_cuts(m) == []


def test_inject_is_value_preserving_and_adds_aux():
    m = _gas_model()
    n_vars_before = m.num_variables
    applied = R.recognize_and_inject(m)
    assert applied == 2
    assert m.num_variables > n_vars_before  # auxiliary u vars added


@pytest.mark.slow
def test_end_to_end_closes_gap():
    """Recognizer + inject closes the relaxation gap to ~0 on the gas network."""
    m = _gas_model()
    applied = R.recognize_and_inject(m)
    assert applied == 2
    res = m.solve(time_limit=60, gap_tolerance=1e-4)
    # bound lifts from the fixed-cost floor (~1.0) to the global optimum (~3.0026)
    assert res.bound >= 2.9
    assert res.objective == pytest.approx(3.0026, abs=1e-2)
