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
from discopt.modeling.core import Model

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
