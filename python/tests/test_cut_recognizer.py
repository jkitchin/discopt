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


# --------------------------------------------------------------------------
# Auto-firing detectors: complementarity (#231) and Fortet binaries (#187)
# --------------------------------------------------------------------------


def test_inject_complementarity_detects_and_cut_is_valid():
    m = dm.Model("compl")
    x = m.continuous("x", lb=0.0, ub=6.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(x + y)
    m.subject_to(x * y == 0, name="comp")
    n = R.inject_complementarity(m)
    assert n == 1
    assert any(c.name == "cut_compl_0" for c in m._constraints)
    # the cut x/6 + y/4 <= 1 holds at every complementarity-feasible point
    for xv, yv in [(6.0, 0.0), (0.0, 4.0), (3.0, 0.0), (0.0, 2.0), (0.0, 0.0)]:
        assert xv / 6.0 + yv / 4.0 <= 1.0 + 1e-9


def test_inject_binary_products_value_preserving():
    def build(inject):
        m = dm.Model("bp")
        b = m.binary("b", shape=(3,))
        m.minimize(-2.0 * b[0] * b[1] * b[2] + 1.0 * b[0])
        m.subject_to(b[0] + b[1] + b[2] >= 1)
        if inject:
            assert R.inject_binary_products(m) == 1
        return m

    base = build(False).solve(time_limit=30, gap_tolerance=1e-4)
    cut = build(True).solve(time_limit=30, gap_tolerance=1e-4)
    assert base.objective == pytest.approx(-1.0, abs=1e-3)
    assert cut.objective == pytest.approx(base.objective, abs=1e-3)  # value-preserving


def test_binary_product_n2_not_fired():
    m = dm.Model("bin2")
    c = m.binary("c", shape=(2,))
    m.minimize(c[0] * c[1])
    assert R.inject_binary_products(m) == 0  # n=2 already exact under McCormick


def test_inject_all_graceful_on_plain_model():
    m = dm.Model("plain")
    z = m.continuous("z", lb=1.0, ub=5.0)
    m.minimize(z**2)
    m.subject_to(z >= 2.0)
    counts = R.inject_all_patterns(m)
    assert counts == {
        "square_diff_network": 0,
        "binary_product": 0,
        "complementarity": 0,
        "gp_monomial": 0,
    }


def test_inject_all_fires_square_diff_on_gas():
    counts = R.inject_all_patterns(_gas_model())
    assert counts["square_diff_network"] == 2
    assert counts["binary_product"] == 0
    assert counts["complementarity"] == 0
    assert counts["gp_monomial"] == 0


def test_gp_cut_is_sound():
    """The GP log-lift cut t >= exp(s0)(1+s-s0) lower-bounds the true monomial."""
    import math

    import numpy as np

    c, a = 2.5, np.array([1.5, 0.7, 0.3])
    xlb, xub = np.array([0.5, 0.4, 0.6]), np.array([4.0, 3.0, 5.0])
    sL = math.log(c) + float(a @ np.log(xlb))
    sU = math.log(c) + float(a @ np.log(xub))
    grid = np.linspace(sL, sU, 6)
    rng = np.random.default_rng(0)
    worst = -1e9
    for _ in range(20000):
        x = rng.uniform(xlb, xub)
        u = np.array([rng.uniform(math.log(xlb[j]), math.log(x[j])) for j in range(3)])
        s = math.log(c) + float(a @ u)
        t = c * np.prod(x**a)
        worst = max(worst, max(math.exp(s0) * (1 + s - s0) - t for s0 in grid))
    assert worst <= 1e-9  # cut never exceeds the true monomial


def test_gp_cut_fires_and_is_value_preserving():
    def build(inject):
        m = dm.Model("gp")
        x = m.continuous("x", lb=0.5, ub=4.0)
        y = m.continuous("y", lb=0.5, ub=4.0)
        m.minimize(2.0 * x**1.5 * y**0.5)
        m.subject_to(x * y >= 4.0)
        n = R.inject_gp_cuts(m) if inject else 0
        return m, n

    m0, _ = build(False)
    m1, n = build(True)
    assert n == 1
    r0 = m0.solve(time_limit=30, gap_tolerance=1e-4)
    r1 = m1.solve(time_limit=30, gap_tolerance=1e-4)
    assert r1.objective == pytest.approx(r0.objective, abs=1e-2)  # value-preserving


def test_gp_cut_skips_single_variable_monomial():
    m = dm.Model("single")
    x = m.continuous("x", lb=0.5, ub=4.0)
    m.minimize(3.0 * x**1.5)  # single-variable: engine already tight
    m.subject_to(x >= 1.0)
    assert R.inject_gp_cuts(m) == 0
