"""Tests for the geometric programming pipeline (issue #41, phases 2-5).

Covers: GP classification of a model, the log-space reformulation
round-trip, negative controls (signomials / non-positive variables must
not be reformulated), an end-to-end solve of a classical GP whose
optimum is known in closed form, and the #40 ``hda`` equilibrium
acceptance criterion (log-convex structure the direct DCP walker cannot
see, recognised by the GP pipeline).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from discopt._jax.convexity.rules import classify_constraint, classify_model
from discopt.gp import (
    as_geometric_program,
    classify_gp,
    solve_gp,
)
from discopt.modeling.core import Model, SolveResult

POS = dict(lb=1e-4, ub=1e4)


# ──────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────


class TestClassification:
    def test_posynomial_min_with_posynomial_constraint_is_gp(self):
        m = Model("gp1")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x / y + 2.0 * x * y)
        m.subject_to(x * y + x / y <= 1.0)
        struct = classify_gp(m)
        assert struct is not None
        assert struct.minimize
        assert len(struct.constraints) == 1
        assert not struct.constraints[0].is_equality

    def test_monomial_equality_is_gp(self):
        m = Model("gp_eq")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x * y)
        m.subject_to(x * y == 4.0)  # monomial == constant
        struct = classify_gp(m)
        assert struct is not None
        assert struct.constraints[0].is_equality

    def test_monomial_maximisation_is_gp(self):
        m = Model("gp_max")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.maximize(x * y)
        m.subject_to(x + y <= 1.0)
        struct = classify_gp(m)
        assert struct is not None
        assert not struct.minimize

    def test_signomial_constraint_is_not_gp(self):
        m = Model("sig")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x * y)
        m.subject_to(x * y - 3.0 * x <= 1.0)  # negative coefficient term
        assert classify_gp(m) is None
        assert as_geometric_program(m) is None

    def test_posynomial_maximisation_is_not_gp(self):
        m = Model("posy_max")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.maximize(x * y + x / y)  # multi-term posynomial objective
        assert classify_gp(m) is None

    def test_posynomial_equality_is_not_gp(self):
        m = Model("posy_eq")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x * y)
        m.subject_to(x * y + x / y == 1.0)  # posynomial == 1, not monomial
        assert classify_gp(m) is None

    def test_nonpositive_variable_is_not_gp(self):
        m = Model("nonpos")
        x = m.continuous("x", lb=0.0, ub=10.0)  # lb == 0
        y = m.continuous("y", **POS)
        m.minimize(x * y + 1.0)
        assert classify_gp(m) is None

    def test_integer_variable_is_not_gp(self):
        m = Model("intgp")
        x = m.continuous("x", **POS)
        n = m.integer("n", lb=1, ub=5)
        m.minimize(x * n)
        assert classify_gp(m) is None

    def test_posynomial_le_posynomial_is_not_gp(self):
        # posynomial <= posynomial (two-monomial RHS) is not GP.
        m = Model("posy_le_posy")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x * y)
        m.subject_to(x * y <= x + y)  # RHS has two monomials
        assert classify_gp(m) is None


# ──────────────────────────────────────────────────────────────────────
# Reformulation round-trip
# ──────────────────────────────────────────────────────────────────────


class TestReformulation:
    def test_log_model_is_built(self):
        m = Model("reform")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x * y + x / y)
        m.subject_to(2.0 * x * y <= 1.0)
        gp = as_geometric_program(m)
        assert gp is not None
        assert gp.n_scalars == 2
        # The log model has a single y vector variable of length 2.
        assert len(gp.log_model._variables) == 1
        assert gp.log_model._variables[0].size == 2

    def test_recover_x_is_exp_of_y(self):
        m = Model("recover")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(x * y)
        gp = as_geometric_program(m)
        assert gp is not None
        y_vals = np.array([math.log(2.0), math.log(5.0)])
        recovered = gp.recover_x(y_vals)
        assert recovered["x"] == pytest.approx(2.0)
        assert recovered["y"] == pytest.approx(5.0)

    def test_objective_value_matches_posynomial(self):
        m = Model("objval")
        x = m.continuous("x", **POS)
        y = m.continuous("y", **POS)
        m.minimize(3.0 * x * y + x / y)
        gp = as_geometric_program(m)
        assert gp is not None
        # At x=2, y=4: 3*2*4 + 2/4 = 24.5
        y_vals = np.array([math.log(2.0), math.log(4.0)])
        assert gp.objective_value(y_vals) == pytest.approx(24.5)

    def test_log_bounds_are_log_of_box(self):
        m = Model("bounds")
        x = m.continuous("x", lb=2.0, ub=8.0)
        m.minimize(x)
        gp = as_geometric_program(m)
        assert gp is not None
        yvar = gp.log_model._variables[0]
        assert float(np.asarray(yvar.lb)[0]) == pytest.approx(math.log(2.0))
        assert float(np.asarray(yvar.ub)[0]) == pytest.approx(math.log(8.0))


# ──────────────────────────────────────────────────────────────────────
# End-to-end solve against closed-form optima
# ──────────────────────────────────────────────────────────────────────


class TestSolveGP:
    def test_unconstrained_monomial_balance(self):
        # minimize x/y + y/x over x,y > 0. Optimum at x == y, value 2.
        m = Model("balance")
        x = m.continuous("x", lb=1e-3, ub=1e3)
        y = m.continuous("y", lb=1e-3, ub=1e3)
        m.minimize(x / y + y / x)
        result = solve_gp(m)
        assert result is not None
        assert result.status in ("optimal", "feasible")
        assert result.objective == pytest.approx(2.0, abs=1e-4)

    def test_classical_box_volume_gp(self):
        # Classic GP (Boyd & Vandenberghe Ex. 4.5 form):
        #   maximize  h*w*d   (box volume)
        #   s.t.  2(h*w + h*d) <= A_wall,  w*d <= A_floor,
        #         alpha <= h/w <= beta,  gamma <= d/w <= delta
        # Solved here in monomial-maximisation form.
        #
        # We use a simplified instance with a known analytic optimum:
        #   maximize x*y  s.t.  x*y <= 6,  x/y <= 3, y/x <= 3
        # The volume bound x*y <= 6 is tight => optimum 6.
        m = Model("boxvol")
        x = m.continuous("x", lb=1e-3, ub=1e3)
        y = m.continuous("y", lb=1e-3, ub=1e3)
        m.maximize(x * y)
        m.subject_to(x * y <= 6.0)
        m.subject_to(x / y <= 3.0)
        m.subject_to(y / x <= 3.0)
        result = solve_gp(m)
        assert result is not None
        assert result.objective == pytest.approx(6.0, abs=1e-4)

    def test_posynomial_objective_gp(self):
        # minimize  x + 1/(x*y) + y  s.t.  x*y >= 1, over x,y>0.
        # Lagrangian/AM-GM: unconstrained min of x + y + 1/(xy).
        # Stationarity: 1 - 1/(x^2 y) = 0, 1 - 1/(x y^2) = 0 => x = y,
        # 1 = 1/x^3 => x = 1, objective = 1 + 1 + 1 = 3.
        m = Model("posyobj")
        x = m.continuous("x", lb=1e-3, ub=1e3)
        y = m.continuous("y", lb=1e-3, ub=1e3)
        m.minimize(x + 1.0 / (x * y) + y)
        result = solve_gp(m)
        assert result is not None
        assert result.objective == pytest.approx(3.0, abs=1e-4)
        assert result.x["x"] == pytest.approx(1.0, abs=1e-3)
        assert result.x["y"] == pytest.approx(1.0, abs=1e-3)

    def test_solve_gp_returns_none_for_non_gp(self):
        m = Model("nongp")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.minimize(x * x)
        assert solve_gp(m) is None


# ──────────────────────────────────────────────────────────────────────
# Solver fast path: model.solve(solver="gp")
# ──────────────────────────────────────────────────────────────────────


class TestSolverFastPath:
    def test_solve_solver_gp_matches_solve_gp(self):
        # model.solve(solver="gp") must reproduce the closed-form optimum
        # and agree with the direct solve_gp() entry point.
        m = Model("fastpath")
        x = m.continuous("x", lb=1e-3, ub=1e3)
        y = m.continuous("y", lb=1e-3, ub=1e3)
        m.minimize(x / y + y / x)
        result = m.solve(solver="gp")
        assert result is not None
        assert result.objective == pytest.approx(2.0, abs=1e-4)

    def test_solve_solver_gp_rejects_non_gp(self):
        # A non-GP model with solver="gp" raises a clear error rather than
        # silently falling through to branch-and-bound.
        m = Model("notgp")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.minimize(x * x)
        with pytest.raises(ValueError, match="not a geometric program"):
            m.solve(solver="gp")

    def test_gp_fast_path_does_not_recurse(self):
        # The log-space model has y-variables with negative lower bounds,
        # so classify_gp(log_model) is None: solving it must not re-enter
        # the GP fast path. We assert the log model is itself not a GP.
        from discopt.gp import as_geometric_program, classify_gp

        m = Model("norecurse")
        x = m.continuous("x", lb=1e-3, ub=1e3)
        y = m.continuous("y", lb=1e-3, ub=1e3)
        m.minimize(x / y + y / x)
        gp = as_geometric_program(m)
        assert gp is not None
        assert classify_gp(gp.log_model) is None


# ──────────────────────────────────────────────────────────────────────
# #40 acceptance criterion: hda equilibrium constraints
# ──────────────────────────────────────────────────────────────────────


def _hda_equilibrium_model() -> Model:
    """A faithful fragment of the MINLPLib ``hda`` process.

    The hydrodealkylation side reaction ``2 C6H6 <=> C12H10 + H2`` has
    the equilibrium relation ``K * x_bz^2 == x_dp * x_h2`` — a signomial
    on strictly-positive concentrations. It is *not* convex in ``x``
    (it is a non-affine equality), but it is a monomial equality, hence
    affine in ``y = log x`` and recognised by the GP pipeline.
    """
    m = Model("hda_equilibrium")
    bz = m.continuous("bz", lb=1e-3, ub=1.0)  # benzene
    dp = m.continuous("dp", lb=1e-3, ub=1.0)  # diphenyl
    h2 = m.continuous("h2", lb=1e-3, ub=1.0)  # hydrogen
    m.minimize(dp + h2 + 1.0 / bz)  # posynomial cost surrogate
    m.subject_to(0.5 * bz**2 == dp * h2)  # equilibrium: monomial == monomial
    m.subject_to(dp / bz <= 0.2)  # selectivity: monomial <= monomial
    m.subject_to(0.05 / (bz * h2) <= 1.0)  # min conversion: posynomial <= monomial
    return m


class TestHdaEquilibrium:
    def test_dcp_walker_cannot_prove_equilibrium_convex(self):
        # The direct DCP walker leaves the equilibrium constraint UNKNOWN:
        # a non-affine equality is not convex in x-space, and the sound
        # numerical certificate cannot rescue it either.
        m = _hda_equilibrium_model()
        eq = m._constraints[0]
        assert classify_constraint(eq, m) is False
        assert classify_constraint(eq, m, use_certificate=True) is False
        is_convex, _ = classify_model(m)
        assert is_convex is False

    def test_gp_pipeline_recognises_equilibrium(self):
        # The GP pipeline classifies the same model: the equilibrium is a
        # monomial equality, and the log-space model is fully convex.
        m = _hda_equilibrium_model()
        struct = classify_gp(m)
        assert struct is not None
        # The first constraint is the monomial == monomial equilibrium.
        assert struct.constraints[0].is_equality
        gp = as_geometric_program(m)
        assert gp is not None
        # In log-space every constraint — including the equilibrium row —
        # is convex, which the DCP walker can now verify directly.
        log_eq = gp.log_model._constraints[0]
        assert classify_constraint(log_eq, gp.log_model) is True
        log_convex, log_mask = classify_model(gp.log_model)
        assert log_convex is True
        assert all(log_mask)

    def test_equilibrium_model_solves_via_gp_fast_path(self):
        # End-to-end: the GP fast path solves the model and the recovered
        # x-solution satisfies the equilibrium relation to tolerance.
        m = _hda_equilibrium_model()
        result = m.solve(solver="gp")
        assert result is not None
        assert result.status in ("optimal", "feasible")
        bz = float(np.asarray(result.x["bz"]))
        dp = float(np.asarray(result.x["dp"]))
        h2 = float(np.asarray(result.x["h2"]))
        # Equilibrium K*bz^2 == dp*h2 holds at the recovered point.
        assert 0.5 * bz**2 == pytest.approx(dp * h2, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────
# GP-2: a certified GP optimum is reported as certified (#413)
# ──────────────────────────────────────────────────────────────────────


class TestGP2Certification:
    """A converged GP solve is a zero-gap *certified* global optimum.

    The log-space program is convex and exact, so an ``optimal`` convex solve
    proves the GP optimum. ``solve_gp`` must therefore report
    ``gap_certified=True``, ``gap == 0``, and a populated ``bound`` mapped back
    from log-space. Before the fix the result was built with ``bound=None`` and
    ``SolveResult.__post_init__`` silently downgraded ``gap_certified`` to
    ``False`` (a conservative mislabel of a genuinely-certified result).
    """

    def test_solved_gp_is_certified(self):
        m = Model("gp2cert")
        x = m.continuous("x", lb=0.5, ub=10.0)
        m.minimize(1.0 / x)  # monomial min -> optimum at x=10, obj=0.1
        result = solve_gp(m)
        assert result is not None
        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.1, abs=1e-5)
        # The certification fields — these fail on main (gap_certified=False).
        assert result.gap_certified is True
        assert result.gap == pytest.approx(0.0, abs=1e-9)
        assert result.bound is not None
        assert result.bound == pytest.approx(result.objective, abs=1e-6)
        assert result.convex_fast_path is True
        # GP-6 rider: the source model is attached for .gradient()/.explain().
        assert result._model is m

    def test_certified_via_default_solve_path(self):
        # The auto-GP fast path inside a plain ``m.solve()`` must also certify.
        m = Model("gp2auto")
        x = m.continuous("x", lb=0.5, ub=10.0)
        y = m.continuous("y", lb=0.5, ub=10.0)
        m.minimize(x * y + 1.0 / (x * y))
        m.subject_to(x * y >= 2.0)
        result = m.solve()
        assert result.status == "optimal"
        assert result.gap_certified is True
        assert result.gap == pytest.approx(0.0, abs=1e-9)
        assert result.bound is not None

    def test_non_optimal_gp_does_not_overclaim(self):
        # A non-``optimal`` convex termination must NOT be certified: no bound,
        # no gap, gap_certified=False (soundness — never fabricate a bound).
        m = Model("gp2tl")
        x = m.continuous("x", lb=0.5, ub=10.0)
        m.minimize(1.0 / x)
        gp = as_geometric_program(m)
        assert gp is not None

        # Force a time_limit termination from the convex solve.
        def _fake_solve(**_kw):
            return SolveResult(status="time_limit", objective=1.23, gap=0.5, x={"y": [0.0]})

        gp.log_model.solve = _fake_solve  # type: ignore[method-assign]
        import discopt.gp as _gp

        orig = _gp.as_geometric_program
        _gp.as_geometric_program = lambda _model: gp  # type: ignore[assignment]
        try:
            result = solve_gp(m)
        finally:
            _gp.as_geometric_program = orig
        assert result is not None
        assert result.status == "time_limit"
        assert result.gap_certified is False
        assert result.bound is None
        assert result.gap is None
        assert result.convex_fast_path is False

    def test_non_gp_still_returns_none(self):
        # Negative control: a non-GP model must not be routed through solve_gp
        # at all (no over-claim can occur).
        m = Model("gp2nongp")
        x = m.integer("x", lb=1, ub=10)
        m.minimize(x)
        assert solve_gp(m) is None


# ──────────────────────────────────────────────────────────────────────
# GP-3: products distribute over sums during recognition (#413)
# ──────────────────────────────────────────────────────────────────────


class TestGP3Distribution:
    """The natural factored Boyd box-volume GP classifies and solves.

    ``2*(h*w + h*d)`` parses as one ``*`` node with a summand factor. Before the
    fix ``_parse_monomial`` refused it and the whole model silently lost GP
    status (falling back to spatial B&B — sound but slow). Distribution over
    the sum is exactly posynomial-preserving, so the factored form must now
    classify and reach the same optimum as the hand-expanded form.
    """

    @staticmethod
    def _box(factored: bool) -> Model:
        m = Model("box")
        h = m.continuous("h", lb=1e-3, ub=1e3)
        w = m.continuous("w", lb=1e-3, ub=1e3)
        d = m.continuous("d", lb=1e-3, ub=1e3)
        m.maximize(h * w * d)
        if factored:
            m.subject_to(2 * (h * w + h * d) <= 10)
        else:
            m.subject_to(2 * h * w + 2 * h * d <= 10)
        m.subject_to(w / d <= 4)
        m.subject_to(d / w <= 4)
        return m

    def test_factored_form_classifies(self):
        # Fails on main: the factored form is refused.
        assert classify_gp(self._box(factored=True)) is not None

    def test_factored_matches_expanded_optimum(self):
        rf = solve_gp(self._box(factored=True))
        re = solve_gp(self._box(factored=False))
        assert rf is not None and re is not None
        assert rf.status == "optimal" and re.status == "optimal"
        assert rf.objective == pytest.approx(re.objective, rel=1e-6)
        assert rf.gap_certified is True

    def test_distribution_preserves_signomial_refusal(self):
        # Negative control: distributing must NOT turn a signomial into a GP.
        # ``2*(h*w - h*d)`` has a negative monomial after distribution.
        m = Model("sig")
        h = m.continuous("h", lb=1e-3, ub=1e3)
        w = m.continuous("w", lb=1e-3, ub=1e3)
        d = m.continuous("d", lb=1e-3, ub=1e3)
        m.minimize(h * w * d)
        m.subject_to(2 * (h * w - h * d) <= 10)
        assert classify_gp(m) is None

    def test_oversized_product_of_sums_refuses(self):
        # Negative control: a product whose distribution exceeds the expansion
        # budget must refuse (fall back to spatial B&B), not blow up. Build a
        # product of several 3-term sums so the term count crosses 64.
        m = Model("big")
        xs = [m.continuous(f"x{i}", lb=1e-3, ub=1e3) for i in range(5)]
        # (x0+x1+1)*(x1+x2+1)*(x2+x3+1)*(x3+x4+1) -> 3^4 = 81 > 64 terms.
        prod = xs[0] + xs[1] + 1.0
        for i in range(1, 4):
            prod = prod * (xs[i] + xs[i + 1] + 1.0)
        m.minimize(prod)
        # Over-budget => not classified as GP (sound refusal, not a crash).
        assert classify_gp(m) is None


# ──────────────────────────────────────────────────────────────────────
# GP-4: SumOverExpression (indexed dm.sum) is recognised (#413)
# ──────────────────────────────────────────────────────────────────────


class TestGP4IndexedSum:
    """The API's own ``dm.sum(..., over=set)`` aggregation classifies as a GP.

    A ``SumOverExpression`` is a plain additive sum of its element expressions;
    before the fix it was refused outright, so the natural indexed posynomial
    never fast-pathed. Recursing into its terms is soundness-neutral.
    """

    def test_indexed_posynomial_classifies_and_solves(self):
        import discopt.modeling as dm

        m = Model("idx")
        s = m.set("s", ["a", "b", "c"])
        xv = m.continuous("xv", over=s, lb=0.5, ub=10.0)
        m.minimize(dm.sum(lambda i: 1.0 / xv[i], over=s))
        assert classify_gp(m) is not None  # fails on main
        result = solve_gp(m)
        assert result is not None
        assert result.status == "optimal"
        # each 1/xv[i] minimised at xv[i]=10 -> 3 * 0.1 = 0.3.
        assert result.objective == pytest.approx(0.3, abs=1e-4)
        assert result.gap_certified is True

    def test_indexed_sum_in_constraint_classifies(self):
        import discopt.modeling as dm

        m = Model("idx2")
        s = m.set("s", ["a", "b"])
        xv = m.continuous("xv", over=s, lb=0.5, ub=10.0)
        m.minimize(dm.sum(lambda i: 1.0 / xv[i], over=s))
        m.subject_to(dm.sum(lambda i: xv[i], over=s) <= 3.0)
        assert classify_gp(m) is not None

    def test_indexed_non_posynomial_still_refused(self):
        # Negative control: an indexed sum of a non-monomial (sin) is not a GP.
        import discopt.modeling as dm

        m = Model("idxsin")
        s = m.set("s", ["a", "b"])
        xv = m.continuous("xv", over=s, lb=0.5, ub=10.0)
        m.minimize(dm.sum(lambda i: dm.sin(xv[i]), over=s))
        assert classify_gp(m) is None

    def test_indexed_signomial_still_refused(self):
        # Negative control: a signomial written with an indexed sum stays refused.
        import discopt.modeling as dm

        m = Model("idxsig")
        s = m.set("s", ["a", "b"])
        xv = m.continuous("xv", over=s, lb=0.5, ub=10.0)
        m.minimize(dm.sum(lambda i: xv[i], over=s) - xv["a"] * xv["b"])
        assert classify_gp(m) is None
