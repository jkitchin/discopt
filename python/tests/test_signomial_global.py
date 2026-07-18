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
    SignomialGlobalStructure,
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


def test_accepts_positive_bounded_integer_variable():
    """Integer variables with a strictly-positive box are now IN class (issue
    #741 Task 2): ``classify`` accepts and records the integer ``u``-coordinates
    for integer branching."""
    m = Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    y = m.integer("y", lb=1, ub=5)
    m.minimize(x**2 - 3.0 * x * y)  # mixed-sign -> in class
    struct = classify_signomial_global(m)
    assert struct is not None
    # y is the second offset (column 1); recorded as an integer coordinate.
    assert struct.integer_coords == [1]


def test_abstains_on_binary_variable():
    """A binary variable has lb ``0`` -> no log domain -> sound abstention."""
    m = Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    y = m.binary("y")
    m.minimize(x**2 - 3.0 * x * y)
    assert classify_signomial_global(m) is None


def test_abstains_on_integer_with_zero_lower_bound():
    """An integer that can be ``0`` breaks the ``u = log x`` lift -> abstain."""
    m = Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    y = m.integer("y", lb=0, ub=5)  # 0 in range -> log undefined
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


# ── 6. tightened constrained relaxation (issue #741) ───────────────────


def _medium_4var():
    """4-var wide-box mixed-sign signomial program (the ≥4-var class of #741).

    min  x1 + x2 + x3 x4
    s.t. x1 x4 >= 8,  x2 x3 >= 6,  x3 + x4 <= 10,  x in [0.5, 20]^4.

    The optimum is interior and analytic: eliminating x1 = 8/x4, x2 = 6/x3
    gives f = 8/x4 + 6/x3 + x3 x4 whose stationary point is x4 = (32/3)^(1/3),
    x3 = 8/x4^2 (the x3 + x4 <= 10 row is slack there). The pre-#741 secant
    relaxation's tree bound on this class stalls hundreds of units below the
    optimum; the tightened relaxation certifies it.
    """
    m = Model()
    x1 = m.continuous("x1", lb=0.5, ub=20.0)
    x2 = m.continuous("x2", lb=0.5, ub=20.0)
    x3 = m.continuous("x3", lb=0.5, ub=20.0)
    x4 = m.continuous("x4", lb=0.5, ub=20.0)
    m.minimize(x1 + x2 + x3 * x4)
    m.subject_to(x1 * x4 >= 8.0)
    m.subject_to(x2 * x3 >= 6.0)
    m.subject_to(x3 + x4 <= 10.0)
    return m


def _medium_4var_optimum():
    x4 = (32.0 / 3.0) ** (1.0 / 3.0)
    x3 = 8.0 / x4**2
    return 8.0 / x4 + 6.0 / x3 + x3 * x4


def _sample_feasible(struct, n=60000, seed=0):
    """Uniform log-box samples split into (U, objective values, feasible mask)."""
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    u_lb, u_ub = struct.u_lb, struct.u_ub
    rng = np.random.default_rng(seed)
    U = u_lb + rng.random((n, len(u_lb))) * (u_ub - u_lb)
    o_sig, o_lc, o_ex = obj_pack
    ovals = (o_sig[None, :] * np.exp(o_lc[None, :] + U @ o_ex.T)).sum(axis=1)
    feas = np.ones(U.shape[0], dtype=bool)
    for sig, lc, ex in con_packs:
        cvals = (sig[None, :] * np.exp(lc[None, :] + U @ ex.T)).sum(axis=1)
        feas &= cvals <= 1e-9
    return U, ovals, feas


def test_medium_4var_certifies_and_legacy_does_not():
    """The #741 acceptance probe for the class: a ≥4-variable wide-box
    constrained signomial certifies under the tightened relaxation (the exact
    single-negative-monomial transform makes every ``x_i x_j >= c`` / ``sum <=
    c`` row exact, so the convex root already closes), while the pre-#741
    relaxation (``obbt=False``) is left hundreds of units below the optimum at a
    generous node budget (the measured blocker)."""
    m = _medium_4var()
    res = solve_signomial_global(m, gap_tolerance=1e-2, max_nodes=5000, time_limit=240)
    f_star = _medium_4var_optimum()
    assert res is not None
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(f_star, rel=1e-4)
    # Certificate invariant: bound <= optimum <= incumbent.
    assert res.bound <= res.objective + 1e-9
    assert res.bound <= f_star + 1e-6
    # Differential reference: the legacy DC relaxation, at a generous budget,
    # stays sound but hopelessly loose (this is the class being fixed).
    legacy = solve_signomial_global(
        m, gap_tolerance=1e-2, max_nodes=3000, time_limit=240, obbt=False
    )
    assert legacy.gap_certified is False
    assert legacy.bound <= f_star + 1e-6  # still sound
    assert legacy.bound < f_star - 100.0  # and hopelessly loose
    assert res.bound > legacy.bound + 100.0  # tightened path dramatically tighter


def test_differential_node_bound_dominates_legacy_on_fixed_boxes():
    """Bound-changing policy: on fixed boxes the tightened node bound is >= the
    legacy bound AND <= the true (sampled) constrained optimum of the box."""
    from discopt._jax.convexity.signomial_global import _constrained_node_bound

    m = _medium_4var()
    struct = classify_signomial_global(m)
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    u_lb0, u_ub0 = struct.u_lb, struct.u_ub
    rng = np.random.default_rng(3)
    for _ in range(6):
        # Random sub-box of the root box (fixed via the seeded rng).
        a = u_lb0 + rng.random(4) * (u_ub0 - u_lb0)
        b = u_lb0 + rng.random(4) * (u_ub0 - u_lb0)
        alb, aub = np.minimum(a, b), np.maximum(a, b)
        legacy, _ = _constrained_node_bound(obj_pack, con_packs, alb, aub)
        tight, _ = _constrained_node_bound(obj_pack, con_packs, alb, aub, tighten=True)
        assert tight >= legacy - 1e-9  # never looser than the reference
        sub = SignomialGlobalStructure(
            terms=struct.terms,
            offsets=struct.offsets,
            u_lb=alb,
            u_ub=aub,
            offset_to_var=struct.offset_to_var,
            constraint_terms=struct.constraint_terms,
        )
        _U, ovals, feas = _sample_feasible(sub, n=40000, seed=11)
        if feas.any():
            true_box_min = float(ovals[feas].min())
            assert tight <= true_box_min + 1e-6  # never above the box optimum


def test_obbt_never_cuts_retained_feasible_points():
    """Feasible-point sampling for the OBBT device: every sampled TRUE-feasible
    point with objective <= incumbent stays inside the tightened box."""
    from discopt._jax.convexity.signomial_global import _obbt_tighten

    m = _medium_4var()
    struct = classify_signomial_global(m)
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    u_lb, u_ub = struct.u_lb, struct.u_ub
    U, ovals, feas = _sample_feasible(struct, n=80000, seed=1)
    assert feas.any()
    incumbent = float(np.percentile(ovals[feas], 25.0))  # a genuine feasible UB
    tt = _obbt_tighten(obj_pack, con_packs, u_lb, u_ub, incumbent, rounds=8)
    assert tt is not None  # retained points exist -> never certified empty
    tlb, tub = tt
    retained = feas & (ovals <= incumbent)
    assert retained.any()
    inside = np.all((U[retained] >= tlb[None, :]) & (U[retained] <= tub[None, :]), axis=1)
    assert bool(inside.all())  # no retained feasible point is ever cut
    # And the box genuinely tightened somewhere (the device does something).
    assert float(np.sum(tub - tlb)) < float(np.sum(u_ub - u_lb)) - 1e-6


def test_interval_min_devices_are_rigorous():
    """The interval floor and the weighted-Lagrangian interval floor
    under-estimate their targets at every sampled point (for any lam >= 0)."""
    from discopt._jax.convexity.signomial_global import (
        _cv_and_grad,
        _interval_min,
        _interval_min_weighted,
    )

    m = _medium_4var()
    struct = classify_signomial_global(m)
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    u_lb, u_ub = struct.u_lb, struct.u_ub
    U, ovals, _feas = _sample_feasible(struct, n=30000, seed=2)
    # _interval_min bounds the TRUE signomial everywhere on the box.
    ivl = _interval_min(*obj_pack, u_lb, u_ub)
    assert ivl <= float(ovals.min()) + 1e-9
    # _interval_min_weighted bounds L(u) = cv_obj + sum lam_i cv_i pointwise.
    rng = np.random.default_rng(4)
    lam = rng.random(len(con_packs)) * 3.0
    weighted = [(obj_pack, 1.0)] + [(p, float(w)) for p, w in zip(con_packs, lam)]
    wmin = _interval_min_weighted(weighted, u_lb, u_ub)
    for k in range(0, 3000, 97):
        u = U[k]
        L = _cv_and_grad(*obj_pack, u, u_lb, u_ub)[0]
        for pack, w in zip(con_packs, lam):
            L += w * _cv_and_grad(*pack, u, u_lb, u_ub)[0]
        assert wmin <= L + 1e-9


def test_certified_infeasible_signomial_program():
    """A constrained signomial program with a provably empty feasible region is
    reported ``infeasible`` with ``gap_certified=True`` (interval certificate),
    never a fabricated incumbent."""
    m = Model()
    x = m.continuous("x", lb=0.5, ub=1.0)
    y = m.continuous("y", lb=0.5, ub=1.0)
    m.minimize(x * y - 2.0 * x)  # mixed-sign objective (in class)
    m.subject_to(x * y >= 10.0)  # impossible: xy <= 1 on the box
    assert classify_signomial_global(m) is not None
    res = solve_signomial_global(m, gap_tolerance=1e-6, max_nodes=1000, time_limit=60)
    assert res is not None
    assert res.status == "infeasible"
    assert res.gap_certified is True
    assert res.objective is None


def test_legacy_obbt_off_path_still_certifies_small_case():
    """The frozen pre-#741 reference path (``obbt=False``) still certifies the
    small 2-var constrained probe — the differential baseline stays alive."""
    m = _constrained_2d()
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=20000, time_limit=60, obbt=False)
    true_min = _constrained_grid_min(m)
    assert res is not None
    assert res.gap_certified is True
    assert res.objective == pytest.approx(true_min, abs=5e-3)
    assert res.bound <= true_min + 1e-3


# ── 7. exact single-negative-monomial convex transform (issue #741 Task 1
#       lever 2 / the prerequisite for Task 2) ──────────────────────────


def test_exact_convex_pack_preserves_feasible_set_and_is_convex():
    """A single-negative-monomial row's exact convex transform crosses zero with
    the original body (identical feasible set — divisor is > 0) and its DC
    underestimator equals itself (a convex posynomial-minus-constant, no secant
    looseness)."""
    from discopt._jax.convexity.signomial_global import (
        _cv_and_grad,
        _exact_convex_pack,
        _true_value,
    )

    # body(u) = 6 - exp(u0 + u1)  (row: x0*x1 >= 6, normalised to <= 0)
    pack = _pack([(1.0, math.log(6.0), np.zeros(2)), (-1.0, 0.0, np.array([1.0, 1.0]))])
    tpack = _exact_convex_pack(pack)
    assert tpack is not pack  # single negative term -> transformed
    u_lb = np.log(np.array([0.5, 0.5]))
    u_ub = np.log(np.array([3.0, 3.0]))
    rng = np.random.default_rng(0)
    for _ in range(2000):
        u = u_lb + rng.random(2) * (u_ub - u_lb)
        b = _true_value(*pack, u)
        t = _true_value(*tpack, u)
        # same feasible set: body <= 0  <=>  transform <= 0
        assert (b <= 1e-9) == (t <= 1e-9)
        # the transform's convex underestimator is itself (exact, no gap)
        cv, _g = _cv_and_grad(*tpack, u, u_lb, u_ub)
        assert cv == pytest.approx(t, abs=1e-9)


def test_exact_convex_pack_leaves_multi_negative_rows_alone():
    """A row with ≥2 negative monomials is a genuine DC difference and must be
    left for the secant envelope (transform returns it unchanged)."""
    from discopt._jax.convexity.signomial_global import _exact_convex_pack

    pack = _pack(
        [
            (1.0, 0.0, np.array([2.0, 0.0])),
            (-1.0, 0.0, np.array([1.0, 1.0])),
            (-1.0, 0.0, np.array([0.0, 2.0])),
        ]
    )
    assert _exact_convex_pack(pack) is pack


# ── 8. integer signomial MINLPs (issue #741 Task 2) ────────────────────


def _small_int_minlp():
    """min 2x + 3y - 5 sqrt(xy)  s.t. xy >= 6 ; x,y integer in [1,6].

    Small enough to brute-force the integer optimum, mixed-sign objective +
    signomial inequality; exercises integer branching, integer rounding, and
    integer-feasible incumbent recovery.
    """
    m = Model()
    x = m.integer("x", lb=1, ub=6)
    y = m.integer("y", lb=1, ub=6)
    m.minimize(2.0 * x + 3.0 * y - 5.0 * x**0.5 * y**0.5)
    m.subject_to(x * y >= 6.0)
    return m


def _brute_int_min(fn, ranges, feas=None):
    """Exhaustive integer optimum over a product of integer ranges."""
    best = math.inf
    best_pt = None
    import itertools as it

    for pt in it.product(*[range(lo, hi + 1) for lo, hi in ranges]):
        if feas is not None and not feas(*pt):
            continue
        v = fn(*pt)
        if v < best:
            best = v
            best_pt = pt
    return best, best_pt


def test_integer_minlp_certifies_bruteforce_optimum():
    """The constrained integer signomial MINLP certifies the true integer
    optimum (matching brute force), with a sound dual bound."""
    m = _small_int_minlp()
    struct = classify_signomial_global(m)
    assert struct is not None
    assert struct.integer_coords == [0, 1]
    best, best_pt = _brute_int_min(
        lambda x, y: 2 * x + 3 * y - 5 * math.sqrt(x * y),
        [(1, 6), (1, 6)],
        feas=lambda x, y: x * y >= 6,
    )
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=20000, time_limit=120)
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(best, abs=1e-4)
    assert res.bound <= best + 1e-4  # certificate invariant vs true integer opt
    # incumbent is genuinely integer-feasible and honours the true constraint
    xv, yv = float(res.x["x"][0]), float(res.x["y"][0])
    assert xv == pytest.approx(round(xv), abs=1e-6)
    assert yv == pytest.approx(round(yv), abs=1e-6)
    assert xv * yv >= 6.0 - 1e-6


def test_box_only_integer_certifies():
    """A box-only (no constraints) mixed-sign integer signomial certifies its
    integer optimum via integer branching + relaxed continuous node bounds."""
    m = Model()
    x = m.integer("x", lb=1, ub=5)
    y = m.integer("y", lb=1, ub=5)
    m.minimize(2.0 * x**2 + 3.0 * y**2 - 7.0 * x * y)  # mixed-sign
    best, _pt = _brute_int_min(lambda x, y: 2 * x * x + 3 * y * y - 7 * x * y, [(1, 5), (1, 5)])
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=20000, time_limit=120)
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(best, abs=1e-4)
    assert res.bound <= best + 1e-4


def test_integer_node_bound_underestimates_all_integer_points():
    """The relaxed node bound (integers relaxed to the continuous box) is <= the
    true objective at EVERY integer-feasible point in the box — a valid dual
    bound for the integer problem, never cutting an integer solution."""
    from discopt._jax.convexity.signomial_global import _constrained_node_bound, _relaxation_packs

    m = _small_int_minlp()
    struct = classify_signomial_global(m)
    obj_pack = _pack(struct.terms)
    con_packs = [_pack(t) for t in struct.constraint_terms]
    relax = _relaxation_packs(con_packs, exact=True)
    u_lb, u_ub = struct.u_lb, struct.u_ub
    lb, _u = _constrained_node_bound(obj_pack, relax, u_lb, u_ub, tighten=True)
    o_sig, o_lc, o_ex = obj_pack
    for xi in range(1, 7):
        for yi in range(1, 7):
            if xi * yi < 6:
                continue
            u = np.array([math.log(xi), math.log(yi)])
            val = float(np.sum(o_sig * np.exp(o_lc + o_ex @ u)))
            assert lb <= val + 1e-6  # bound never exceeds an integer-feasible value


def test_certified_infeasible_integer_program():
    """An integer program whose integer-feasible region is provably empty is
    reported ``infeasible`` with ``gap_certified=True`` — never a fabricated
    incumbent."""
    m = Model()
    x = m.integer("x", lb=1, ub=2)
    y = m.integer("y", lb=1, ub=2)
    m.minimize(x * y - 3.0 * x)  # mixed-sign objective (in class)
    m.subject_to(x * y >= 9.0)  # impossible: max xy = 4
    res = solve_signomial_global(m, gap_tolerance=1e-6, max_nodes=5000, time_limit=60)
    assert res.status == "infeasible"
    assert res.gap_certified is True
    assert res.objective is None


def test_integer_solver_integration_flag_gated(monkeypatch):
    """`Model.solve()` routes an integer signomial MINLP to the SGO path when
    DISCOPT_SGO is enabled and certifies its integer optimum."""
    m = _small_int_minlp()
    monkeypatch.setenv("DISCOPT_SGO", "1")
    res = m.solve()
    best, _pt = _brute_int_min(
        lambda x, y: 2 * x + 3 * y - 5 * math.sqrt(x * y),
        [(1, 6), (1, 6)],
        feas=lambda x, y: x * y >= 6,
    )
    assert res.gap_certified is True
    assert res.objective == pytest.approx(best, abs=1e-4)


@pytest.mark.slow
def test_nsig30_family_admitted_sound_incumbent():
    """The ``cvxnonsep_nsig*`` family (issue #741 Task 2) is admitted: the
    30-var ``cvxnonsep_nsig30`` classifies, its dual bound stays sound (<=
    oracle), and integer-feasible incumbent recovery reaches the known integer
    optimum 130.6287 (full certification of the 30-var box is not expected
    in-budget — the same wide-box certification frontier as ex7_2_3)."""
    import os

    from discopt.modeling.core import from_nl

    path = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "cvxnonsep_nsig30.nl")
    if not os.path.exists(path):
        pytest.skip("cvxnonsep_nsig30.nl not present")
    m = from_nl(path)
    struct = classify_signomial_global(m)
    assert struct is not None
    assert len(struct.integer_coords) == 15
    res = solve_signomial_global(m, gap_tolerance=1e-4, max_nodes=2000, time_limit=150)
    oracle = 130.6287
    assert res.objective is not None
    # sound dual bound below the oracle; incumbent at (or above) the oracle
    if res.bound is not None:
        assert res.bound <= oracle + 1e-2
    assert res.objective >= oracle - 1e-2
    # every integer variable in the incumbent is genuinely integral
    for name in list(res.x)[15:30]:
        xv = float(res.x[name][0])
        assert xv == pytest.approx(round(xv), abs=1e-6)
