"""Registered domain operators — component A of #1248 (plugin needs, #1249).

A plugin knows structure that generic factorable relaxation throws away. Written
in primitives, the Redlich-Kister binary the CALPHAD plugin prices phases with —
``x(1-x)(L0 + L1(2x-1)) + RT[x ln x + (1-x) ln(1-x)]`` — is relaxed term by term,
which loses every cancellation between the terms.

**RETRACTION (CLAUDE.md §11).** This file first reported 56x-216x node-count
ratios for naming the composite. That measurement was taken against a primitive
arm in which every ``dm.xlogx`` term reached the relaxation engine's INTERVAL
FLOOR, because ``entropy`` had no ``_UNIVARIATE_FN`` entry (#1277). Almost the
whole ratio was that missing envelope, not this mechanism. With the envelope in
place (20 interleaved solves, same optimum in every one):

    L0      L1       RT   primitive   registered   ratio   was reported
     3.0    0.0      1.0        51          35     1.46x      216x
     5.0    0.0      1.0        51          47     1.09x     56.5x
     3.0    1.5      1.0        23          15     1.53x      116x
     8.0   -4.0      1.0        29          25     1.16x     69.2x
 20000.0 5000.0   8314.0        29          19     1.53x      106x

The mechanism still earns its keep — a strict node-count reduction in every row,
at the same optimum, and it is general (any user composite, not this family) —
but the honest figure is 1.1x-1.5x, not two orders of magnitude.

The mechanism is not new either: it is the secant/tangent envelope the engine
already emits for a univariate node of known curvature; what registration changes
is that the whole composite is the node.

**Nothing is taken on trust.** The lowering is the definition; ``f``/``f'`` are
evaluated on it through the solver's own tape, and the per-box curvature verdict
is an interval enclosure of ``f''`` — a proof on that box, not a sample. There is
no user-supplied envelope, so there is none to be unsound, which is a stronger
answer than #1248's "reject an unsound ``relax`` at registration". Accepting a
user envelope (with that sampling gate, and the trust caveat it implies) is
deliberately left to a follow-up.

The soundness tests below are the point of this file: a tighter relaxation that
cuts the optimum is worse than no relaxation at all.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.operators import clear_registered, get_registered, registered_names

pytestmark = [pytest.mark.smoke]

R_GAS = 8.314
_GRID = np.linspace(1e-9, 1 - 1e-9, 200_001)


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-global; never leak a registration into the suite."""
    clear_registered()
    yield
    clear_registered()


def _rk_body(x, L0, L1, rt):
    return x * (1 - x) * (L0 + L1 * (2 * x - 1)) + rt * (dm.xlogx(x) + dm.xlogx(1 - x))


def _true_min(L0, L1, rt):
    """Dense-sampling truth for the univariate family (the envelope's referee)."""
    x = _GRID
    ent = x * np.log(x) + (1 - x) * np.log(1 - x)
    return float(np.min(x * (1 - x) * (L0 + L1 * (2 * x - 1)) + rt * ent))


def _primitive_model(L0, L1, rt):
    m = dm.Model("primitive")
    x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)
    m.minimize(_rk_body(x, L0, L1, rt))
    return m


def _registered_model(L0, L1, rt, name="rk_binary"):
    fn = dm.register_function(name, lambda x: _rk_body(x, L0, L1, rt), replace=True)
    m = dm.Model("registered")
    x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)
    m.minimize(fn(x))
    return m


# --------------------------------------------------------------------------- #
# Soundness first
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "L0,L1,rt",
    [(3.0, 0.0, 1.0), (5.0, 0.0, 1.0), (3.0, 1.5, 1.0), (8.0, -4.0, 1.0), (-6.0, 2.0, 2.5)],
)
def test_the_registered_envelope_never_cuts_the_optimum(L0, L1, rt):
    """A valid lower bound may not exceed the true minimum, and the incumbent may
    not fall below it. This is the whole risk of enveloping a composite on a
    derived curvature verdict."""
    r = _registered_model(L0, L1, rt).solve(time_limit=60)
    truth = _true_min(L0, L1, rt)
    scale = max(1.0, abs(truth))
    assert r.bound is not None
    assert (r.bound - truth) / scale <= 1e-6, (r.bound, truth)
    assert (r.objective - truth) / scale >= -1e-6, (r.objective, truth)
    assert r.objective == pytest.approx(truth, abs=1e-4 * scale)


def test_registered_and_primitive_agree_on_the_optimum():
    for L0, L1, rt in [(3.0, 0.0, 1.0), (8.0, -4.0, 1.0), (20000.0, 5000.0, R_GAS * 1000.0)]:
        a = _primitive_model(L0, L1, rt).solve(time_limit=60)
        b = _registered_model(L0, L1, rt).solve(time_limit=60)
        assert a.status == b.status == "optimal"
        assert b.objective == pytest.approx(a.objective, rel=1e-6, abs=1e-6)


# --------------------------------------------------------------------------- #
# The payoff
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("L0,L1,rt", [(3.0, 0.0, 1.0), (3.0, 1.5, 1.0)])
def test_naming_the_composite_cuts_the_node_count(L0, L1, rt):
    """The measured ratios are 1.46x and 1.53x on these two rows (see the
    retraction in the module docstring: the 216x first reported here was the
    missing `entropy` envelope of #1277, not this mechanism). The bar is a strict
    reduction rather than a factor, so the test pins the mechanism without
    pinning a machine-specific number."""
    a = _primitive_model(L0, L1, rt).solve(time_limit=60)
    b = _registered_model(L0, L1, rt).solve(time_limit=60)
    assert a.status == b.status == "optimal"
    assert b.node_count < a.node_count, (a.node_count, b.node_count)


# --------------------------------------------------------------------------- #
# The model still carries the lowering — every other consumer is untouched
# --------------------------------------------------------------------------- #
def test_the_model_holds_the_lowering_not_an_unknown_opcode():
    """Registration adds no opcode: the expression in the model is the primitive
    body, so evaluation, export and the Rust core need no new support. (An
    unregistered name reaches the Rust doorway as ``Unknown MathFunc``, which is
    exactly what this design avoids.)"""
    m = _registered_model(3.0, 1.5, 1.0)
    from discopt._rust import model_to_repr

    model_to_repr(m)  # must not raise

    # `.nl` export: the atom exports exactly as well as its lowering does — no
    # better and no worse. The RK family's lowering contains `xlogx`, which #1242
    # refuses to export because `.nl` has no entropy opcode and `x*log(x)` is not
    # an exact substitute; that refusal must still fire through a registration
    # rather than be bypassed by it.
    with pytest.raises(ValueError, match="no .nl opcode"):
        m.to_nl()

    poly = dm.register_function("rk_poly", lambda x: x * (1 - x) * (3.0 + 1.5 * (2 * x - 1)))
    mp = dm.Model("poly")
    xp = mp.continuous("x", lb=0.0, ub=1.0)
    mp.minimize(poly(xp))
    text = mp.to_nl()
    assert isinstance(text, str) and text

    from discopt._tape_nlp_evaluator import make_evaluator

    ev = make_evaluator(m)
    prim = _primitive_model(3.0, 1.5, 1.0)
    ev_prim = make_evaluator(prim)
    for xv in (0.1, 0.35, 0.5, 0.9):
        got = float(ev.evaluate_objective(np.array([xv])))
        want = float(ev_prim.evaluate_objective(np.array([xv])))
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), xv


def test_a_dropped_registration_degrades_to_term_by_term():
    """A tag whose registration is gone must relax as the plain expression, not
    raise: the lowering is still a complete, valid model."""
    m = _registered_model(3.0, 0.0, 1.0)
    clear_registered()
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(_true_min(3.0, 0.0, 1.0), abs=1e-4)


# --------------------------------------------------------------------------- #
# The curvature derivation
# --------------------------------------------------------------------------- #
def test_curvature_is_derived_per_box_and_abstains_when_it_straddles():
    """``x ln x + (1-x) ln(1-x) + 3x(1-x)`` is convex near the ends of [0,1] and
    concave in the middle. The derived verdict must say so — and abstain on a box
    that straddles the inflection rather than guessing a sign."""
    dm.register_function("rk_curv", lambda x: _rk_body(x, 3.0, 0.0, 1.0))
    fn = get_registered("rk_curv")
    _f, _fp, curvature = fn._derived()

    assert curvature(0.001, 0.05) == "convex"
    assert curvature(0.45, 0.55) == "concave"
    assert curvature(0.001, 0.999) is None  # straddles: abstain, never guess


def test_derived_value_and_derivative_match_the_lowering():
    dm.register_function("rk_vals", lambda x: _rk_body(x, 3.0, 1.5, 1.0))
    f, fp, _curv = get_registered("rk_vals")._derived()
    for xv in (0.05, 0.25, 0.5, 0.75, 0.95):
        ent = xv * np.log(xv) + (1 - xv) * np.log(1 - xv)
        want = xv * (1 - xv) * (3.0 + 1.5 * (2 * xv - 1)) + ent
        assert f(xv) == pytest.approx(want, rel=1e-12, abs=1e-12)
        # derivative by central difference on the same definition
        h = 1e-6
        ent_p = (xv + h) * np.log(xv + h) + (1 - xv - h) * np.log(1 - xv - h)
        ent_m = (xv - h) * np.log(xv - h) + (1 - xv + h) * np.log(1 - xv + h)
        fp_num = (
            ((xv + h) * (1 - xv - h) * (3.0 + 1.5 * (2 * (xv + h) - 1)) + ent_p)
            - ((xv - h) * (1 - xv + h) * (3.0 + 1.5 * (2 * (xv - h) - 1)) + ent_m)
        ) / (2 * h)
        assert fp(xv) == pytest.approx(fp_num, rel=1e-5, abs=1e-5)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_a_builtin_name_cannot_be_shadowed():
    """Registering ``exp`` would re-define the operator's envelope for every model
    in the process."""
    with pytest.raises(ValueError, match="built-in"):
        dm.register_function("exp", lambda x: x)
    with pytest.raises(ValueError, match="built-in"):
        dm.register_function("entropy", lambda x: x)


def test_duplicate_registration_needs_replace():
    dm.register_function("dup", lambda x: x * x)
    with pytest.raises(ValueError, match="already registered"):
        dm.register_function("dup", lambda x: x * x * x)
    dm.register_function("dup", lambda x: x * x * x, replace=True)
    assert "dup" in registered_names()


def test_argument_validation():
    with pytest.raises(ValueError, match="non-empty"):
        dm.register_function("  ", lambda x: x)
    with pytest.raises(TypeError, match="callable"):
        dm.register_function("bad", "not a callable")


def test_a_non_expression_lowering_is_refused_at_use():
    """``lower`` must build a symbolic body; an opaque numeric callable cannot be
    relaxed, and saying so at the call site names the alternative."""
    fn = dm.register_function("numeric", lambda x: 1.0)
    m = dm.Model("bad")
    x = m.continuous("x", lb=0, ub=1)
    with pytest.raises(TypeError, match="dm.custom"):
        fn(x)


def test_the_engine_really_took_the_registered_envelope():
    """Node count alone cannot distinguish "the envelope fired" from "the search
    got lucky"; the use counter is the direct evidence (CLAUDE.md §6)."""
    m = _registered_model(3.0, 1.5, 1.0, name="rk_used")
    fn = get_registered("rk_used")
    assert fn.use_count == 0
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert fn.use_count > 0, "the registered envelope was never consulted"


def test_a_replaced_registration_never_envelopes_a_model_built_before_it():
    """The model keeps the body it was built with; ``replace=True`` must not hand
    the relaxer the NEW function's envelope for it. Before the fix the replaced
    (``+100``) envelope was emitted for the old body and its rows cut the true
    point ``(t, exp(t) + t^2)`` by ~100 — a false bound. A stale tag now falls
    back to term-by-term relaxation of the body the model actually carries."""
    from discopt._relax.uniform_relax import build_uniform_relaxation, clear_analysis_cache

    def _kinds(model):
        clear_analysis_cache(model)  # as solve_model does at the start of every solve
        return sorted(k for k, _ in build_uniform_relaxation(model).coverage.values())

    fn = dm.register_function("stale_probe", lambda x: dm.exp(x) + x * x, replace=True)
    m_old = dm.Model("stale")
    x = m_old.continuous("x", lb=0.5, ub=2.0)
    m_old.minimize(fn(x))
    assert _kinds(m_old) == ["univariate_call"], "control: the fresh tag must be an atom"

    new = dm.register_function("stale_probe", lambda x: dm.exp(x) + x * x + 100.0, replace=True)
    assert _kinds(m_old) != ["univariate_call"], (
        "a model built before replace=True was still enveloped as the (new) atom"
    )
    uses_before = new.use_count
    r = m_old.solve(time_limit=60)
    truth = float(np.exp(0.5) + 0.25)
    assert r.status == "optimal"
    assert r.bound <= truth + 1e-6, (r.bound, truth)
    assert abs(r.objective - truth) <= 1e-5, (r.objective, truth)
    assert new.use_count == uses_before, "the replaced envelope was applied to the old body"

    # Control arm: a model built AFTER the replacement is the new atom.
    m_new = dm.Model("fresh")
    y = m_new.continuous("y", lb=0.5, ub=2.0)
    m_new.minimize(new(y))
    assert _kinds(m_new) == ["univariate_call"]
