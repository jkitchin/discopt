"""Tests for McCormick relaxation bounds in the B&B loop."""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
from discopt.modeling.core import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_nonconvex_model():
    """min x*y  s.t. x in [1,4], y in [1,4], x+y >= 3, x integer."""
    m = Model()
    x = m.integer("x", lb=1, ub=4)
    y = m.continuous("y", lb=1.0, ub=4.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 3.0)
    return m


def _simple_minimize_model():
    """min x^2 + y^2  s.t. x in [0,4], y in [0,4], x integer."""
    m = Model()
    x = m.integer("x", lb=0, ub=4)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(x**2 + y**2)
    return m


def _maximize_model():
    """max -(x^2)  s.t. x in [0,4], x integer => optimal x=0, obj=0."""
    m = Model()
    x = m.integer("x", lb=0, ub=4)
    m.maximize(-(x**2))
    return m


def _convex_quadratic_model():
    """min (x-1)^2 + (y-2)^2  s.t. x in [0,3], y in [0,3], x integer."""
    m = Model()
    x = m.integer("x", lb=0, ub=3)
    y = m.continuous("y", lb=0.0, ub=3.0)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    return m


def _ge_constrained_model():
    """min x + y  s.t. x + y >= 2, x in [0,5], y in [0,5]."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(x + y)
    m.subject_to(x + y >= 2.0)
    return m


def _eq_constrained_model():
    """min x + y  s.t. x + y == 3, x in [0,5], y in [0,5]."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(x + y)
    m.subject_to(x + y == 3.0)
    return m


# ===========================================================================
# Option A: Midpoint bounds
# ===========================================================================


class TestMidpointBounds:
    """Tests for McCormick midpoint evaluation (Option A)."""

    def test_cv_at_midpoint_underestimates_f(self):
        """cv(midpoint) <= f(midpoint) by McCormick validity."""
        from discopt._jax.mccormick_nlp import evaluate_midpoint_bound
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _simple_minimize_model()
        relax_fn = compile_objective_relaxation(model)

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([4.0, 4.0])
        mc_lb = evaluate_midpoint_bound(relax_fn, lb, ub, negate=False)

        # Midpoint = [2, 2], f(2,2) = 4 + 4 = 8
        # cv(midpoint) should be <= f(midpoint)
        assert mc_lb <= 8.0 + 1e-6
        assert np.isfinite(mc_lb)

    def test_narrower_bounds_tighter_cv(self):
        """Narrower bounds should give tighter (higher) cv."""
        from discopt._jax.mccormick_nlp import evaluate_midpoint_bound
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _simple_minimize_model()
        relax_fn = compile_objective_relaxation(model)

        lb_wide = jnp.array([0.0, 0.0])
        ub_wide = jnp.array([4.0, 4.0])
        lb_narrow = jnp.array([1.0, 1.0])
        ub_narrow = jnp.array([3.0, 3.0])

        cv_wide = evaluate_midpoint_bound(relax_fn, lb_wide, ub_wide)
        cv_narrow = evaluate_midpoint_bound(relax_fn, lb_narrow, ub_narrow)

        # Both evaluate at their respective midpoints
        # Wide: mid=[2,2], Narrow: mid=[2,2] (same midpoint!)
        # But McCormick relaxation quality improves with narrower bounds
        # cv_narrow should be >= cv_wide (tighter underestimator)
        assert cv_narrow >= cv_wide - 1e-8

    def test_batch_evaluation(self):
        """Batch midpoint evaluation matches serial."""
        from discopt._jax.mccormick_nlp import (
            evaluate_midpoint_bound,
            evaluate_midpoint_bound_batch,
        )
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _simple_minimize_model()
        relax_fn = compile_objective_relaxation(model)

        lb_batch = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        ub_batch = jnp.array([[4.0, 4.0], [2.0, 2.0], [3.0, 3.0]])

        batch_result = np.asarray(evaluate_midpoint_bound_batch(relax_fn, lb_batch, ub_batch))

        for i in range(3):
            serial = evaluate_midpoint_bound(relax_fn, lb_batch[i], ub_batch[i])
            np.testing.assert_allclose(batch_result[i], serial, atol=1e-10)

    def test_maximize_sign_handling(self):
        """For maximize, lower bound uses -cc."""
        from discopt._jax.mccormick_nlp import evaluate_midpoint_bound
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _maximize_model()
        relax_fn = compile_objective_relaxation(model)

        lb = jnp.array([0.0])
        ub = jnp.array([4.0])

        # For max -(x^2), the objective expression is -(x^2).
        # negate=True means we want bound for minimizing the negated obj.
        mc_lb = evaluate_midpoint_bound(relax_fn, lb, ub, negate=True)
        assert np.isfinite(mc_lb)

        # Without negate, we get cv of -(x^2) at midpoint
        mc_cv = evaluate_midpoint_bound(relax_fn, lb, ub, negate=False)
        # cv of -(x^2) at midpoint=2 should be <= -(2^2) = -4
        assert mc_cv <= -4.0 + 1e-6

    def test_end_to_end_minlp_midpoint(self):
        """Full solve with mccormick_bounds='midpoint'."""
        model = _simple_minimize_model()
        result = model.solve(mccormick_bounds="midpoint", max_nodes=1000)
        assert result.status in ("optimal", "feasible")
        if result.status == "optimal":
            # x integer in [0,4], y continuous [0,4]: optimal at x=0, y=0
            assert result.objective is not None
            assert result.objective <= 0.0 + 1e-4


# ===========================================================================
# Option B: NLP bounds
# ===========================================================================


class TestNLPBounds:
    """Tests for McCormick NLP relaxation solving (Option B)."""

    def test_nlp_bound_is_valid_lower_bound(self):
        """NLP relaxation bound <= true optimum (since cv <= f)."""
        from discopt._jax.mccormick_nlp import solve_mccormick_relaxation_nlp
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _simple_minimize_model()
        relax_fn = compile_objective_relaxation(model)

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([4.0, 4.0])

        nlp_lb = solve_mccormick_relaxation_nlp(relax_fn, None, None, lb, ub)

        # True minimum of x^2+y^2 on [0,4]^2 is 0 at (0,0)
        # min_x cv(x) <= min_x f(x) = 0, so nlp_lb <= 0
        assert nlp_lb <= 0.0 + 1e-4

    def test_nlp_bound_finds_minimum_of_underestimator(self):
        """NLP solving finds the global min of the convex underestimator."""
        from discopt._jax.mccormick_nlp import (
            evaluate_midpoint_bound,
            solve_mccormick_relaxation_nlp,
        )
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _simple_minimize_model()
        relax_fn = compile_objective_relaxation(model)

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([4.0, 4.0])

        mp_lb = evaluate_midpoint_bound(relax_fn, lb, ub)
        nlp_lb = solve_mccormick_relaxation_nlp(relax_fn, None, None, lb, ub)

        # NLP minimizes cv over the domain, should give <= cv(midpoint)
        assert nlp_lb <= mp_lb + 1e-6

    def test_handles_ge_constraint(self):
        """NLP relaxation with >= constraints (binding)."""
        from discopt._jax.mccormick_nlp import solve_mccormick_relaxation_nlp
        from discopt._jax.relaxation_compiler import (
            compile_constraint_relaxation,
            compile_objective_relaxation,
        )

        model = _ge_constrained_model()
        c = model._constraints[0]

        obj_fn = compile_objective_relaxation(model)
        con_fns = [compile_constraint_relaxation(c, model)]
        # Use actual normalized sense from the model
        senses = [c.sense]

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([5.0, 5.0])

        nlp_lb = solve_mccormick_relaxation_nlp(obj_fn, con_fns, senses, lb, ub)
        # x+y >= 2 normalized to (2-x-y) <= 0
        # Relaxation constraint: cv of (2-x-y) <= 0 => 2-x-y <= 0 => x+y >= 2
        # True min of x+y s.t. x+y>=2 is 2
        assert nlp_lb <= 2.0 + 1e-3
        assert nlp_lb >= 1.0  # binding constraint should push > 0

    def test_handles_eq_constraint(self):
        """NLP relaxation with == constraints (one-sided relaxation)."""
        from discopt._jax.mccormick_nlp import solve_mccormick_relaxation_nlp
        from discopt._jax.relaxation_compiler import (
            compile_constraint_relaxation,
            compile_objective_relaxation,
        )

        model = _eq_constrained_model()
        c = model._constraints[0]

        obj_fn = compile_objective_relaxation(model)
        con_fns = [compile_constraint_relaxation(c, model)]
        senses = [c.sense]

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([5.0, 5.0])

        nlp_lb = solve_mccormick_relaxation_nlp(obj_fn, con_fns, senses, lb, ub)
        # x+y == 3 normalized to (3-x-y) <= 0 (one-sided)
        # cv of (3-x-y) <= 0 => 3-x-y <= 0 => x+y >= 3
        # min x+y s.t. x+y >= 3 is 3
        # This is a valid lower bound on the equality-constrained problem
        assert nlp_lb <= 3.0 + 1e-3

    def test_convex_fast_convergence(self):
        """Convex problem converges quickly with few IPM iterations."""
        from discopt._jax.mccormick_nlp import solve_mccormick_relaxation_nlp
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _convex_quadratic_model()
        relax_fn = compile_objective_relaxation(model)

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([3.0, 3.0])

        nlp_lb = solve_mccormick_relaxation_nlp(relax_fn, None, None, lb, ub, max_iter=50)
        # min (x-1)^2 + (y-2)^2 on [0,3]^2 is 0 at (1,2)
        # McCormick cv should reach near-zero minimum
        assert nlp_lb <= 0.0 + 1e-2
        assert np.isfinite(nlp_lb)

    def test_expired_deadline_skips_relaxation_solves(self):
        """Expired B&B deadlines should not start more McCormick NLP solves."""
        from discopt._jax.mccormick_nlp import (
            solve_mccormick_batch,
            solve_mccormick_relaxation_nlp,
        )

        calls = 0

        def relax_fn(x_cv, x_cc, lb, ub):
            nonlocal calls
            calls += 1
            return x_cv[0], x_cc[0]

        lb = jnp.array([0.0])
        ub = jnp.array([1.0])
        expired = time.perf_counter() - 1.0

        nlp_lb = solve_mccormick_relaxation_nlp(relax_fn, None, None, lb, ub, deadline=expired)
        assert nlp_lb == -np.inf

        lb_batch = jnp.array([[0.0], [0.0], [0.0]])
        ub_batch = jnp.array([[1.0], [1.0], [1.0]])
        batch_lbs = solve_mccormick_batch(
            relax_fn, None, None, lb_batch, ub_batch, deadline=expired
        )
        np.testing.assert_allclose(np.asarray(batch_lbs), np.full(3, -np.inf))
        assert calls == 0

    def test_end_to_end_minlp_nlp(self):
        """Full solve with mccormick_bounds='nlp'."""
        model = _simple_minimize_model()
        result = model.solve(mccormick_bounds="nlp", max_nodes=1000)
        assert result.status in ("optimal", "feasible")
        if result.status == "optimal":
            assert result.objective is not None
            assert result.objective <= 0.0 + 1e-4


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Integration tests for McCormick bounds in solver."""

    def test_coexists_with_alphabb(self):
        """McCormick bounds + alphaBB both active, takes max."""
        model = _simple_nonconvex_model()
        result = model.solve(mccormick_bounds="midpoint", max_nodes=500)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_auto_activates_for_dag_models(self):
        """'auto' mode should activate midpoint for DAG models."""
        model = _simple_minimize_model()
        result = model.solve(mccormick_bounds="auto", max_nodes=100)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_none_disables(self):
        """'none' mode should disable McCormick bounds."""
        model = _simple_minimize_model()
        result = model.solve(mccormick_bounds="none", max_nodes=100)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_global_optimality_with_bounds(self):
        """McCormick bounds should help prove global optimality."""
        model = _convex_quadratic_model()
        result = model.solve(mccormick_bounds="midpoint", max_nodes=500)
        assert result.status in ("optimal", "feasible")
        if result.status == "optimal":
            # (x-1)^2 + (y-2)^2, x integer: optimal x=1, y=2, obj=0
            assert result.objective is not None
            np.testing.assert_allclose(result.objective, 0.0, atol=1e-3)
