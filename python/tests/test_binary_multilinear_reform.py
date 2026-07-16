"""Tests for the pure-binary multilinear exact linearization (issue #187).

The pass turns a polynomial objective over binary-valued variables into an
equivalent pure MILP (Fortet/Glover per-monomial linearization + integer-point
secant envelopes for squared integer forms) so the solver routes to the MILP
branch-and-bound instead of the spatial/JAX path whose full-DAG Jacobian XLA
compile is the ``autocorr_bern*`` wall.

Soundness gates from the issue:
- Fortet linearization exact for 0/1 products — property-tested
  ``aux == prod(binaries)`` at all integer points.
- Secant envelope exact at every attainable integer point.
- Clean fallback to the existing paths when not pure-binary-multilinear.
- Value preservation: reformulated optimum == brute-force optimum.
"""

import itertools
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt import Model
from discopt._jax import binary_multilinear_reform as B

pytestmark = pytest.mark.relaxation


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_autocorr(n, K, objvar_sense=None):
    """Bernasconi low-autocorrelation instance: minimize
    ``sum_{k=1..K} (sum_i s_i s_{i+k})**2`` with ``s = 2b - 1`` and ``b`` typed
    as {0,1}-bounded INTEGER (matching ``from_nl``'s typing of MINLPLib's 0/1
    columns — the issue #187 correction). ``objvar_sense`` switches to the
    `.nl` objvar convention: ``min tau`` with ``tau <sense> E`` as a row."""
    m = Model(name=f"autocorr{n}_{K}")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(n)]
    s = [2 * bi - 1 for bi in b]
    E = None
    for k in range(1, K + 1):
        Ck = None
        for i in range(n - k):
            t = s[i] * s[i + k]
            Ck = t if Ck is None else Ck + t
        term = Ck * Ck
        E = term if E is None else E + term
    if objvar_sense is None:
        m.minimize(E)
    else:
        tau = m.continuous("objvar", lb=-1e20, ub=1e20)
        if objvar_sense == "==":
            m.subject_to(tau == E)
        else:
            m.subject_to(tau >= E)
        m.minimize(tau)
    return m, b


def autocorr_energy(bits, K):
    n = len(bits)
    s = [2 * x - 1 for x in bits]
    return sum(sum(s[i] * s[i + k] for i in range(n - k)) ** 2 for k in range(1, K + 1))


def brute_force_autocorr(n, K):
    return min(autocorr_energy(bits, K) for bits in itertools.product([0, 1], repeat=n))


# ---------------------------------------------------------------------------
# Property tests (issue soundness gates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deg", [2, 3, 4, 5])
def test_fortet_aux_equals_product_at_all_integer_points(deg):
    """The Fortet rows admit exactly ``z == prod(b)`` at every binary vertex:
    the feasible z-interval [max(0, sum b - (n-1)), min b] is the singleton
    ``prod b`` for every 0/1 assignment."""
    for bits in itertools.product([0, 1], repeat=deg):
        z_lo = max(0.0, sum(bits) - (deg - 1))
        z_hi = min(bits)
        prod = float(np.prod(bits))
        assert z_lo == z_hi == prod


@pytest.mark.parametrize("lo,hi,step", [(-3, 4, 1), (-6, 6, 2), (0, 9, 3), (-5, -1, 2)])
def test_secant_envelope_exact_at_attainable_points(lo, hi, step):
    """max over grid secants ``(u+v)*y - u*v`` equals ``y**2`` exactly at every
    attainable grid point, and lies at/above the parabola in between (chords of
    a convex function) — over-approximating only where no attainable value
    exists, which tightens the relaxation without cutting any integer point."""
    secants = [(u, u + step) for u in range(lo, hi, step)]
    for y in range(lo, hi + 1, step):
        env = max((u + v) * y - u * v for u, v in secants)
        assert env == y * y
    for y in np.linspace(lo, hi, 37):
        env = max((u + v) * y - u * v for u, v in secants)
        assert env >= y * y - 1e-9


def test_expansion_matches_direct_evaluation_at_all_vertices():
    """The multilinear expansion is exact at every binary vertex."""
    n, K = 6, 5
    m, b = build_autocorr(n, K)
    ctx = B._ExpandCtx()
    poly = B._expand_to_multilinear(m._objective.expression, ctx)
    pos = {bv._index: i for i, bv in enumerate(b)}
    for bits in itertools.product([0, 1], repeat=n):
        val = sum(c * np.prod([bits[pos[vi]] for vi, _ in mono]) for mono, c in poly.items())
        assert abs(val - autocorr_energy(bits, K)) < 1e-9


# ---------------------------------------------------------------------------
# Value preservation (reformulated MILP optimum == brute force)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,K", [(6, 5), (8, 7)])
def test_reformulated_optimum_matches_brute_force(n, K):
    ref = brute_force_autocorr(n, K)
    m, _ = build_autocorr(n, K)
    m2 = B.reformulate_binary_multilinear(m)
    assert m2 is not m
    res = m2.solve(time_limit=120, nlp_solver="simplex", presolve=False)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("row_sense", ["==", ">="])
def test_objvar_defining_row_form(row_sense):
    """The `.nl` objvar convention (objective = bare free var pinned by one
    defining row) is recognized and secant-encoded through the row."""
    n, K = 6, 5
    ref = brute_force_autocorr(n, K)
    m, _ = build_autocorr(n, K, objvar_sense=row_sense)
    m2 = B.reformulate_binary_multilinear(m)
    assert m2 is not m
    assert any(v.name.startswith("_bml_t") for v in m2._variables), (
        "expected the defining row's squares to be secant-encoded"
    )
    res = m2.solve(time_limit=120, nlp_solver="simplex", presolve=False)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-5)


def test_maximize_sense_preserved():
    m = Model()
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(4)]
    m.maximize(3 * b[0] * b[1] * b[2] - 2 * b[1] * b[2] * b[3] + b[0] - b[3])
    ref = max(
        3 * v[0] * v[1] * v[2] - 2 * v[1] * v[2] * v[3] + v[0] - v[3]
        for v in itertools.product([0, 1], repeat=4)
    )
    m2 = B.reformulate_binary_multilinear(m)
    assert m2 is not m
    res = m2.solve(time_limit=60, nlp_solver="simplex", presolve=False)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-5)


def test_wrong_direction_square_falls_back_to_flat_and_stays_exact():
    """A square whose objective pressure points the wrong way (maximizing
    ``+E**2`` / minimizing ``-E**2``) must NOT be secant-encoded (the epigraph
    would be unsound) — it flat-expands and the optimum stays exact."""
    for sense in ("min", "max"):
        m = Model()
        b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(4)]
        s = [2 * bi - 1 for bi in b]
        C = s[0] * s[1] + s[1] * s[2] + s[2] * s[3]
        E = C * C
        vals = [
            (lambda sv: (sv[0] * sv[1] + sv[1] * sv[2] + sv[2] * sv[3]) ** 2)(
                [2 * v[0] - 1, 2 * v[1] - 1, 2 * v[2] - 1, 2 * v[3] - 1]
            )
            for v in itertools.product([0, 1], repeat=4)
        ]
        if sense == "max":
            m.maximize(E)
            ref = max(vals)
        else:
            m.minimize(-E)
            ref = -max(vals)
        m2 = B.reformulate_binary_multilinear(m)
        assert m2 is not m
        assert not any(v.name.startswith("_bml_t") for v in m2._variables)
        res = m2.solve(time_limit=60, nlp_solver="simplex", presolve=False)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(ref, abs=1e-5)


# ---------------------------------------------------------------------------
# Clean fallback (the pass must abstain, never mis-rewrite)
# ---------------------------------------------------------------------------


def test_mixed_continuous_factor_not_fired():
    m = Model()
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(2)]
    x = m.continuous("x", lb=0, ub=2)
    m.minimize(b[0] * b[1] * x + x)
    assert not B.has_binary_multilinear_work(m)
    assert B.reformulate_binary_multilinear(m) is m


def test_binary_quadratic_not_fired():
    """Degree-2-only binary models keep their current path (McCormick is
    already exact per bilinear term there)."""
    m = Model()
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(3)]
    m.minimize(b[0] * b[1] + b[1] * b[2] - b[0])
    assert not B.has_binary_multilinear_work(m)
    assert B.reformulate_binary_multilinear(m) is m


def test_general_integer_factor_not_fired():
    m = Model()
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(2)]
    w = m.integer("w", lb=0, ub=3)
    m.minimize(b[0] * b[1] * w)
    assert B.reformulate_binary_multilinear(m) is m


def test_transcendental_not_fired():
    import discopt.modeling as dm

    m = Model()
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(3)]
    m.minimize(dm.exp(b[0] * b[1] * b[2]))
    assert not B.has_binary_multilinear_work(m)
    assert B.reformulate_binary_multilinear(m) is m


def test_mixed_model_with_binary_witness_still_abstains():
    """Gate sees an all-binary cubic witness, but the model also carries a
    continuous factor in another degree>=2 monomial — the whole pass must
    abstain (partial rewrites are not exactness-preserving)."""
    m = Model()
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(3)]
    x = m.continuous("x", lb=0, ub=2)
    m.minimize(b[0] * b[1] * b[2] + b[0] * x)
    assert B.has_binary_multilinear_work(m)
    assert B.reformulate_binary_multilinear(m) is m


def test_deterministic_rebuild():
    m1, _ = build_autocorr(8, 7)
    m2, _ = build_autocorr(8, 7)
    r1 = B.reformulate_binary_multilinear(m1)
    r2 = B.reformulate_binary_multilinear(m2)
    assert [v.name for v in r1._variables] == [v.name for v in r2._variables]
    assert len(r1._constraints) == len(r2._constraints)
    assert repr(r1._objective.expression) == repr(r2._objective.expression)


# ---------------------------------------------------------------------------
# End-to-end: default solve() fires the pass and clears the wall (regression)
# ---------------------------------------------------------------------------


def test_autocorr_end_to_end_certifies_within_budget(caplog):
    """Issue #187 acceptance shape: an autocorr instance certifies quickly on
    a default solve. Before the pass this model burned the whole budget on the
    spatial/JAX path (n=8 dense: >60 s, no certificate); with it, the MILP
    route certifies the brute-force optimum in seconds."""
    import logging

    n, K = 8, 7
    ref = brute_force_autocorr(n, K)
    m, _ = build_autocorr(n, K)
    t0 = time.time()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        res = m.solve(time_limit=120)
    wall = time.time() - t0
    assert any("binary-multilinear linearization" in r.message for r in caplog.records), (
        "the exact linearization did not auto-fire on a default solve"
    )
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-5)
    assert wall < 60, f"autocorr n=8 took {wall:.1f}s — the #187 wall is back"


def test_from_nl_round_trip_fires_and_certifies(tmp_path):
    """The real target path: a `.nl`-loaded model ({0,1} INTEGER typing,
    left-associative sum chains from reconstruction) must pass the gate,
    secant-encode its squares, and certify the brute-force optimum."""
    n, K = 6, 5
    from discopt.modeling.core import from_nl

    ref = brute_force_autocorr(n, K)
    m, _ = build_autocorr(n, K)
    path = str(tmp_path / "autocorr.nl")
    m.to_nl(path)
    loaded = from_nl(path)
    assert all(v.var_type.value == "integer" for v in loaded._variables)
    assert B.has_binary_multilinear_work(loaded)
    m2 = B.reformulate_binary_multilinear(loaded)
    assert m2 is not loaded
    assert any(v.name.startswith("_bml_t") for v in m2._variables)
    res = loaded.solve(time_limit=120)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-5)


def test_binary_valued_integer_recognized_in_model_to_sympy():
    """{0,1}-bounded INTEGER columns count as binary in the symbolic
    recognizer layer too (issue #187 correction 1)."""
    pytest.importorskip("sympy")
    from discopt._jax.symbolic.cut_recognizer import model_to_sympy

    m = Model()
    bi = m.integer("bi", lb=0, ub=1)
    bb = m.binary("bb")
    w = m.integer("w", lb=0, ub=3)
    m.minimize(bi * bb * w)
    sm = model_to_sympy(m)
    names = {s.name for s in sm.binaries}
    assert "bi" in names and "bb" in names and "w" not in names
