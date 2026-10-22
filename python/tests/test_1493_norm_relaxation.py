"""``dm.norm`` of a vector gets a relaxation (#1493).

Before this fix ``canonical_expr`` turned the vector argument of ``norm{p}`` into an
array-valued opaque child, ``uniform_relax.bounds`` called ``float()`` on its vector
enclosure, and the whole relaxation build raised ``TypeError`` -- caught and logged at
DEBUG on every node. No model containing ``dm.norm(v, ...)`` of a vector ever had a
dual bound: ``norm(v, 1) + 0.5*sin(3*v0)`` ended ``unknown`` with no point, and the
convex ``min norm(v, inf)`` stalled at 0.545 (optimum 0.5) with nothing to fall back to.

The fix spells a 1-D norm out exactly as the POUNCE tape lowers it (``sum|e_i|``,
``max|e_i|``, ``sqrt(sum e_i^2)``, ``(sum|e_i|^p)^(1/p)``), so it canonicalizes to the
node the user's own scalar spelling produces and relaxes through the existing
envelopes -- the tests below assert exactly that equality, plus soundness against
sampled box minima. A >=2-D argument is ``jnp.linalg.norm``'s MATRIX norm, which is
not a fold over elements, and keeps no envelope (its interval is ``[0, inf)``).
"""

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt._relax.discretization import DiscretizationState
from discopt._relax.milp_relaxation import build_milp_relaxation
from discopt._relax.term_classifier import classify_nonlinear_terms

ORDERS = [1, 2, "inf", 3]


def _np_norm(x, p):
    return float(np.linalg.norm(x, np.inf if p == "inf" else p))


def _scalar_spelling(v, p, n):
    if p == 1:
        return sum(abs(v[i]) for i in range(n))
    if p == 2:
        return dm.sqrt(sum(v[i] * v[i] for i in range(n)))
    if p == "inf":
        return dm.maximum(*[abs(v[i]) for i in range(n)]) if n > 1 else abs(v[0])
    return sum(abs(v[i]) ** p for i in range(n)) ** (1.0 / p)


def _root_bound(form, p, n, lo, hi, with_sin):
    m = dm.Model("norm_relax")
    v = m.continuous("v", shape=(n,), lb=-5, ub=5)
    e = dm.norm(v, p) if form == "norm" else _scalar_spelling(v, p, n)
    if with_sin:
        e = e + 0.5 * dm.sin(3 * v[0])
    m.minimize(e)
    milp, _ = build_milp_relaxation(
        m, classify_nonlinear_terms(m), DiscretizationState(), bound_override=(lo, hi)
    )
    res = milp.solve()
    return res.bound


# --------------------------------------------------------------------------- #
# relaxation: equal to the scalar spelling, and below the true box minimum
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("p", ORDERS)
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("with_sin", [False, True], ids=["pure", "sin"])
def test_norm_root_bound_matches_scalar_spelling_and_is_sound(p, n, with_sin):
    rng = np.random.default_rng(1493 + n)
    checked = 0
    for _ in range(4):
        lo = rng.uniform(-3, 3, n)
        hi = lo + rng.uniform(0.1, 3, n)
        b_norm = _root_bound("norm", p, n, lo, hi, with_sin)
        b_spell = _root_bound("spell", p, n, lo, hi, with_sin)
        assert b_norm is not None, "norm of a vector must have a root bound"
        assert b_spell is not None
        assert b_norm == pytest.approx(b_spell, rel=1e-9, abs=1e-9)
        pts = rng.uniform(lo, hi, size=(3000, n))
        f = np.array([_np_norm(x, p) for x in pts])
        if with_sin:
            f = f + 0.5 * np.sin(3 * pts[:, 0])
        assert b_norm <= float(f.min()) + 1e-7, "bound above a feasible objective value"
        checked += 1
    assert checked == 4


@pytest.mark.unit
@pytest.mark.parametrize("p", ORDERS)
def test_norm_interval_is_scalar_and_encloses_samples(p):
    rng = np.random.default_rng(7)
    m = dm.Model("iv")
    v = m.continuous("v", shape=(3,), lb=-5, ub=5)
    checked = 0
    for _ in range(20):
        lo = rng.uniform(-3, 3, 3)
        hi = lo + rng.uniform(0.0, 3, 3)
        enc = evaluate_interval(dm.norm(v, p), m, {v: Interval(lo, hi)})
        assert np.asarray(enc.lo).shape == () and np.asarray(enc.hi).shape == ()
        pts = rng.uniform(lo, hi, size=(500, 3))
        vals = np.array([_np_norm(x, p) for x in pts])
        assert float(enc.lo) <= vals.min() and vals.max() <= float(enc.hi)
        checked += 1
    assert checked == 20


@pytest.mark.unit
@pytest.mark.parametrize("p", [1, 2, "inf"])
def test_matrix_norm_interval_is_only_nonnegativity(p):
    """A 2-D norm is jnp's MATRIX norm; entrywise bounds do not enclose it."""
    m = dm.Model("mat")
    X = m.continuous("X", shape=(3, 3), lb=-1, ub=1)
    # ||I_3||_2 = 1 but the entrywise 2-norm is sqrt(3): an entrywise lower bound
    # would cut the identity off.
    box = {X: Interval(np.eye(3), np.eye(3))}
    enc = evaluate_interval(dm.norm(X, p), m, box)
    assert float(enc.lo) == 0.0 and float(enc.hi) == np.inf


@pytest.mark.unit
def test_matrix_norm_relaxation_builds_with_a_valid_bound():
    m = dm.Model("mat")
    X = m.continuous("X", shape=(2, 2), lb=-1, ub=1)
    m.subject_to(X[0, 0] + X[1, 1] >= 1)
    m.minimize(dm.norm(X, 2))
    milp, _ = build_milp_relaxation(m, classify_nonlinear_terms(m), DiscretizationState())
    res = milp.solve()
    # The spectral norm here is >= 0.5; the only valid claim the layer makes is >= 0.
    assert res.bound is not None and res.bound <= 0.5


# --------------------------------------------------------------------------- #
# end to end: the issue's models
# --------------------------------------------------------------------------- #
def _issue_model(p, with_sin):
    m = dm.Model("n")
    v = m.continuous("v", shape=(2,), lb=-2, ub=2)
    m.subject_to(v[0] + v[1] >= 1)
    e = dm.norm(v, p)
    if with_sin:
        e = e + 0.5 * dm.sin(3 * v[0])
    m.minimize(e)
    return m


@pytest.mark.unit
@pytest.mark.parametrize("p,opt", [("inf", 0.5), (1, 1.0), (2, 0.5**0.5)])
def test_convex_norm_models_certify(p, opt):
    r = _issue_model(p, False).solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.error)
    assert r.gap_certified
    assert r.objective == pytest.approx(opt, abs=1e-5)
    assert r.bound is not None and r.bound <= opt + 1e-6


# Oracle: 1601x1601 grid over [-2, 2]^2 (v0 + v1 >= 1), refined by a local SLSQP.
_SIN_GRID_OPT = {1: 0.954876, 2: 0.933036, "inf": 0.907678}


@pytest.mark.unit
@pytest.mark.parametrize("p", [1, 2, "inf"])
def test_norm_plus_sin_gets_a_sound_dual_bound(p):
    r = _issue_model(p, True).solve(time_limit=15)
    assert r.x is not None, (r.status, r.error)
    assert r.bound is not None, "no dual bound for a norm model (#1493)"
    assert r.bound <= _SIN_GRID_OPT[p] + 1e-4
    assert r.objective >= _SIN_GRID_OPT[p] - 1e-4
    if r.status == "optimal":
        assert r.objective == pytest.approx(_SIN_GRID_OPT[p], abs=1e-3)


@pytest.mark.unit
def test_norm_constraint_certifies():
    m = dm.Model("con")
    v = m.continuous("v", shape=(3,), lb=-3, ub=3)
    m.subject_to(dm.norm(v, 1) <= 1)
    m.maximize(v[0] + 2 * v[1] + 3 * v[2])
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the swallowed failure is no longer silent; routing
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_relaxation_build_failure_is_reported_once_at_warning(monkeypatch, caplog):
    import discopt._relax.mccormick_lp as mlp

    m = _issue_model(1, True)
    relaxer = mlp.MccormickLPRelaxer(m)

    def boom(*a, **k):
        raise TypeError("synthetic build failure")

    monkeypatch.setattr(mlp, "build_milp_relaxation", boom)
    lb = np.array([-2.0, -2.0])
    ub = np.array([2.0, 2.0])
    with caplog.at_level(logging.DEBUG, logger="discopt._relax.mccormick_lp"):
        for _ in range(3):
            assert relaxer.solve_at_node(lb, ub).status == "error"
    warns = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "relaxation build failed" in r.getMessage()
    ]
    assert len(warns) == 1
    assert "TypeError: synthetic build failure" in warns[0].getMessage()


@pytest.mark.unit
def test_l1_linf_norms_are_nonsmooth_but_the_spectral_norm_is_not():
    """Vector and (since the #1873 follow-up) 2-D 1-/inf-norms are spelled out as
    abs/max and route to the B&B; the spectral norm has no relaxation and stays
    on the NLP."""
    from discopt.solver import _model_contains_nonsmooth_node

    for p, expect in [(1, True), ("inf", True), (2, False), (3, False)]:
        m = dm.Model("s")
        v = m.continuous("v", shape=(2,), lb=-2, ub=2)
        m.minimize(dm.norm(v, p))
        assert _model_contains_nonsmooth_node(m) is expect, p
    for p, expect in [(1, True), ("inf", True), (2, False)]:
        m = dm.Model("s")
        X = m.continuous("X", shape=(2, 2), lb=-2, ub=2)
        m.minimize(dm.norm(X, p))
        assert _model_contains_nonsmooth_node(m) is expect, p
