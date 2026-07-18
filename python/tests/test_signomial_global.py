"""Regression tests for the mixed-sign signomial global solver (issue #114).

Covers the certified box-program solve, its soundness (dual bound never exceeds
the true optimum; node bound underestimates every sampled feasible value),
equivalence of the numpy DC core with the certified envelope, and the sound
abstention boundary (``classify_signomial_global`` returns ``None`` off-class).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from discopt import Model
from discopt._jax.convexity.signomial import is_signomial, signomial_dc_terms
from discopt._jax.convexity.signomial_global import (
    _cv_and_grad,
    _node_lower_bound,
    _pack,
    classify_signomial_global,
    solve_signomial_global,
)
from discopt._jax.symbolic.signed_signomial import signed_signomial_dc_envelope

# ── model builders ────────────────────────────────────────────────────


def _st_e36_like():
    """2 x0^2 + 0.008 x1^3 - 3.2 x0 x1 - 2 x1 over [3,5.5] x [15,25].

    The mixed-sign signomial objective of MINLPLib ``st_e36`` with both variables
    continuous (its integer var relaxed). ``x0, x1`` are shared between the + and
    - terms — the shared-variable looseness the box bound cannot handle. True box
    minimum is -304.5 at the corner (5.5, 25).
    """
    m = Model()
    x0 = m.continuous("x0", lb=3.0, ub=5.5)
    x1 = m.continuous("x1", lb=15.0, ub=25.0)
    m.minimize(2.0 * x0**2 + 0.008 * x1**3 - 3.2 * x0 * x1 - 2.0 * x1)
    return m


def _interior_shared():
    """x^2 y^2 + 1/(x^2 y^2) - 3 x y over [0.2, 5]^2 (interior min, fully shared).

    g(t) = t^2 + 1/t^2 - 3 t with t = x y; the minimum is in the box interior,
    the hardest case for the corner/secant bound.
    """
    m = Model()
    x = m.continuous("x", lb=0.2, ub=5.0)
    y = m.continuous("y", lb=0.2, ub=5.0)
    m.minimize(x**2 * y**2 + x ** (-2) * y ** (-2) - 3.0 * x * y)
    return m


def _grid_min_2d(model, n=1200):
    """True box minimum of a 2-var signomial objective by dense grid."""
    form = is_signomial(model._objective.expression, model)
    terms, offsets = signomial_dc_terms(form)
    sig, log_c, exps = _pack(terms)
    v = model._variables
    lb = [math.log(float(np.asarray(vi.lb).ravel()[0])) for vi in v]
    ub = [math.log(float(np.asarray(vi.ub).ravel()[0])) for vi in v]
    us = np.linspace(lb[0], ub[0], n)
    vs = np.linspace(lb[1], ub[1], n)
    best = math.inf
    for a in us:
        vals = sig[:, None] * np.exp(
            log_c[:, None] + np.outer(exps[:, 0], [a]) + np.outer(exps[:, 1], vs)
        )
        s = vals.sum(axis=0)
        best = min(best, float(s.min()))
    return best


# ── 1. certification ──────────────────────────────────────────────────


def test_certifies_mixed_sign_box_program():
    """The SGO solve certifies the global optimum of a mixed-sign box program."""
    m = _st_e36_like()
    assert classify_signomial_global(m) is not None  # in-class
    res = solve_signomial_global(m, gap_tolerance=1e-6, max_nodes=100000)
    true_min = _grid_min_2d(m)  # -304.5
    assert res is not None
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(true_min, abs=1e-3)
    # Certificate invariant: bound <= optimum <= incumbent.
    assert res.bound <= res.objective + 1e-6
    assert res.bound <= true_min + 1e-6
    assert res.bound == pytest.approx(true_min, abs=1e-3)


def test_solver_integration_flag_gated(monkeypatch):
    """`Model.solve()` routes to the SGO path only when DISCOPT_SGO is enabled."""
    m = _st_e36_like()
    monkeypatch.setenv("DISCOPT_SGO", "1")
    res = m.solve()
    true_min = _grid_min_2d(m)
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(true_min, abs=1e-3)
    assert res.bound <= res.objective + 1e-6


# ── 2. soundness: bound never exceeds the true optimum ─────────────────


def test_bound_sound_when_not_converged():
    """Even truncated at few nodes, the dual bound stays <= the true optimum."""
    m = _interior_shared()
    true_min = _grid_min_2d(m)
    res = solve_signomial_global(m, gap_tolerance=1e-6, max_nodes=20)
    assert res is not None
    # Not necessarily converged, but the bound must remain a valid underestimate.
    assert res.bound <= true_min + 1e-6
    assert res.bound <= res.objective + 1e-6
    if not res.gap_certified:
        assert res.status in ("feasible", "node_limit", "time_limit")


def test_node_bound_underestimates_all_sampled_values():
    """A single-box node bound is <= S at every feasible point in that box."""
    m = _interior_shared()
    form = is_signomial(m._objective.expression, m)
    terms, offsets = signomial_dc_terms(form)
    sig, log_c, exps = _pack(terms)
    u_lb = np.log(np.array([0.2, 0.2]))
    u_ub = np.log(np.array([5.0, 5.0]))
    lb, _u, _v = _node_lower_bound(sig, log_c, exps, u_lb, u_ub)
    rng = np.random.default_rng(0)
    U = u_lb + rng.random((40000, 2)) * (u_ub - u_lb)
    s_vals = (sig[None, :] * np.exp(log_c[None, :] + U @ exps.T)).sum(axis=1)
    assert lb <= float(s_vals.min()) + 1e-9


# ── 3. equivalence with the certified envelope ─────────────────────────


def test_cv_matches_certified_secant_envelope():
    """The numpy DC core equals signed_signomial_dc_envelope(secant) pointwise."""
    m = _interior_shared()
    form = is_signomial(m._objective.expression, m)
    terms, offsets = signomial_dc_terms(form)
    sig, log_c, exps = _pack(terms)
    u_lb = np.log(np.array([0.2, 0.2]))
    u_ub = np.log(np.array([5.0, 5.0]))
    rng = np.random.default_rng(1)
    for _ in range(25):
        u = u_lb + rng.random(2) * (u_ub - u_lb)
        cv_np, _g = _cv_and_grad(sig, log_c, exps, u, u_lb, u_ub)
        cv_ref, _cc = signed_signomial_dc_envelope(u, terms, u_lb, u_ub, overestimator="secant")
        assert cv_np == pytest.approx(float(cv_ref), rel=1e-9, abs=1e-9)


# ── 4. sound abstention boundary ───────────────────────────────────────


def test_abstains_on_integer_variable():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    y = m.integer("y", lb=1, ub=5)
    m.minimize(x**2 - 3.0 * x * y)
    assert classify_signomial_global(m) is None


def test_accepts_signomial_inequality_constraint():
    """A signomial *inequality* constraint is now IN class (issue #114 follow-up):
    ``classify`` accepts it and records the normalised ``body <= 0`` term list."""
    m = _st_e36_like()
    x0v, x1v = m._variables[0], m._variables[1]
    m.subject_to(x0v * x1v >= 60.0)  # signomial inequality -> in class
    struct = classify_signomial_global(m)
    assert struct is not None
    assert len(struct.constraint_terms) == 1


def test_abstains_on_signomial_equality_constraint():
    """A signomial *equality* is out of class (needs both DC sides) -> abstain."""
    m = _st_e36_like()
    x0v, x1v = m._variables[0], m._variables[1]
    m.subject_to(x0v * x1v == 60.0)
    assert classify_signomial_global(m) is None


def test_abstains_on_nonsignomial_constraint():
    """A non-signomial constraint body (transcendental) -> sound abstention."""
    import discopt.modeling as dm

    m = _st_e36_like()
    x0v = m._variables[0]
    m.subject_to(dm.exp(x0v) <= 100.0)
    assert classify_signomial_global(m) is None


def test_abstains_on_pure_posynomial_program():
    """Posynomial objective + posynomial ``<=`` constraint is a convex GP the
    exact GP path owns; the SGO classifier abstains (no negative term anywhere)."""
    m = Model()
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x**2 + 3.0 * x * y + y**2)  # posynomial
    m.subject_to(x * y <= 4.0)  # posynomial <= const, no negative term
    assert classify_signomial_global(m) is None


def test_abstains_on_maximize():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    y = m.continuous("y", lb=1.0, ub=5.0)
    m.maximize(x**2 - 3.0 * x * y)
    assert classify_signomial_global(m) is None


def test_abstains_on_posynomial_single_sign():
    """All-positive (posynomial) objective is not mixed-sign -> SGO abstains
    (the exact GP path owns it)."""
    m = Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    y = m.continuous("y", lb=1.0, ub=5.0)
    m.minimize(x**2 + 3.0 * x * y + y**2)
    assert classify_signomial_global(m) is None


def test_abstains_on_nonpositive_lower_bound():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=5.0)  # 0 lb -> no log domain
    y = m.continuous("y", lb=1.0, ub=5.0)
    m.minimize(x**2 - 3.0 * x * y)
    assert classify_signomial_global(m) is None


def test_abstains_on_unbounded_box():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=math.inf)
    y = m.continuous("y", lb=1.0, ub=5.0)
    m.minimize(x**2 - 3.0 * x * y)
    assert classify_signomial_global(m) is None


# ── 5. constrained signomial programs (issue #114 follow-up) ───────────


def _constrained_2d():
    """min x^2 + y^2 - 3 x y  over [0.5, 3]^2  s.t.  x y >= 1.

    Mixed-sign objective with a signomial inequality constraint; exercises the
    constrained node relaxation and feasible-incumbent recovery. Optimum verified
    by dense grid.
    """
    m = Model()
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x**2 + y**2 - 3.0 * x * y)
    m.subject_to(x * y >= 1.0)
    return m


def _constrained_grid_min(model, n=800):
    """True constrained min of a 2-var signomial obj + signomial constraints."""
    obj = is_signomial(model._objective.expression, model)
    o_terms, _o_off = signomial_dc_terms(obj)
    o_sig, o_lc, o_ex = _pack(o_terms)
    cons = []
    for c in model._constraints:
        cf = is_signomial(c.body, model)
        c_terms, _c_off = signomial_dc_terms(cf)
        sig, lc, ex = _pack(c_terms)
        if c.sense == ">=":
            sig = -sig
        cons.append((sig, lc, ex))
    v = model._variables
    lb = [math.log(float(np.asarray(vi.lb).ravel()[0])) for vi in v]
    ub = [math.log(float(np.asarray(vi.ub).ravel()[0])) for vi in v]
    us = np.linspace(lb[0], ub[0], n)
    vs = np.linspace(lb[1], ub[1], n)
    best = math.inf
    for a in us:
        for b in vs:
            u = np.array([a, b])
            if any(float(np.sum(s * np.exp(lc + e @ u))) > 1e-9 for s, lc, e in cons):
                continue
            best = min(best, float(np.sum(o_sig * np.exp(o_lc + o_ex @ u))))
    return best


def test_constrained_certifies_matches_grid():
    """The constrained SGO solve certifies a mixed-sign signomial program with a
    signomial inequality constraint, matching a dense-grid true optimum."""
    m = _constrained_2d()
    assert classify_signomial_global(m) is not None
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=20000, time_limit=60)
    true_min = _constrained_grid_min(m)
    assert res is not None
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(true_min, abs=5e-3)
    assert res.bound <= res.objective + 1e-6  # certificate invariant
    assert res.bound <= true_min + 1e-3


def test_constrained_incumbent_is_genuinely_feasible():
    """The reported incumbent satisfies the true constraint (never a false
    'feasible' point) and the dual bound never exceeds the true optimum."""
    m = _constrained_2d()
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=20000, time_limit=60)
    assert res is not None and res.x is not None
    xv = float(res.x["x"][0])
    yv = float(res.x["y"][0])
    assert xv * yv >= 1.0 - 1e-5  # true constraint honoured
    true_min = _constrained_grid_min(m)
    assert res.bound <= true_min + 1e-3  # sound dual bound


def test_constrained_solver_integration_flag_gated(monkeypatch):
    """`Model.solve()` routes a *constrained* signomial program to the SGO path
    when DISCOPT_SGO is enabled, and certifies a feasible global optimum."""
    m = _constrained_2d()
    monkeypatch.setenv("DISCOPT_SGO", "1")
    res = m.solve()
    true_min = _constrained_grid_min(m)
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(true_min, abs=5e-3)
    assert res.bound <= res.objective + 1e-6
    # incumbent honours the true constraint x*y >= 1
    assert float(res.x["x"][0]) * float(res.x["y"][0]) >= 1.0 - 1e-5


def test_constrained_node_bound_underestimates_feasible_samples():
    """A single constrained node's rigorous bound is <= the true objective at
    every TRUE-feasible sampled point in that box (valid relaxation, no cut)."""
    from discopt._jax.convexity.signomial_global import _constrained_node_bound

    m = _constrained_2d()
    struct = classify_signomial_global(m)
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    u_lb, u_ub = struct.u_lb, struct.u_ub
    lb, _u = _constrained_node_bound(obj_pack, con_packs, u_lb, u_ub)
    rng = np.random.default_rng(0)
    U = u_lb + rng.random((60000, len(u_lb))) * (u_ub - u_lb)
    o_sig, o_lc, o_ex = obj_pack
    ovals = (o_sig[None, :] * np.exp(o_lc[None, :] + U @ o_ex.T)).sum(axis=1)
    feas = np.ones(U.shape[0], dtype=bool)
    for sig, lc, ex in con_packs:
        cvals = (sig[None, :] * np.exp(lc[None, :] + U @ ex.T)).sum(axis=1)
        feas &= cvals <= 1e-9
    if feas.any():
        assert lb <= float(ovals[feas].min()) + 1e-6
