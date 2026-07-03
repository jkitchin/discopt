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
# C-18: the removed "midpoint" mode returned a non-bound
# ===========================================================================


class TestC18MidpointNotABound:
    """Correctness issue C-18.

    ``mccormick_bounds="midpoint"`` returned the convex underestimator's VALUE
    at the box midpoint, ``u(mid)``, and fed it into the node lower bound. But
    ``u(mid) <= f(mid)`` does NOT imply ``u(mid) <= min_box f``: the value can sit
    ABOVE the true box minimum, so it is not a valid lower bound and could fathom
    the node holding the true optimum -> false "optimal". The mode is removed and
    now rejected loudly.
    """

    def test_midpoint_value_exceeds_true_box_minimum(self):
        """Documents the non-bound: for x**2 on [1,3], u(mid=2) > min_box x**2 = 1.

        This is the mechanism that made the old ``evaluate_midpoint_bound`` return
        an invalid lower bound. We reconstruct the exact value the removed helper
        produced (cv at the midpoint) directly from the compiled relaxation and
        show it exceeds the true box minimum.
        """
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        m = Model()
        x = m.continuous("x", lb=1.0, ub=3.0)
        m.minimize(x * x)

        relax_fn = compile_objective_relaxation(m)
        lb = jnp.array([1.0])
        ub = jnp.array([3.0])
        mid = 0.5 * (lb + ub)
        cv, _cc = relax_fn(mid, mid, lb, ub)
        u_mid = float(cv)  # exactly what evaluate_midpoint_bound(...) returned

        true_box_min = 1.0  # min of x**2 on [1, 3] at x=1
        # The old "bound" is strictly ABOVE the true minimum -> NOT a valid bound.
        assert u_mid > true_box_min + 1e-6

    def test_midpoint_mode_is_rejected_loudly(self):
        """Selecting the removed mode raises ValueError (never returns a non-bound)."""
        import pytest

        model = _simple_minimize_model()
        with pytest.raises(ValueError, match="C-18"):
            model.solve(mccormick_bounds="midpoint", max_nodes=1000)

    def test_unknown_mccormick_bounds_rejected(self):
        """A typo/unknown value is rejected rather than silently ignored."""
        import pytest

        model = _simple_minimize_model()
        with pytest.raises(ValueError, match="Unknown mccormick_bounds"):
            model.solve(mccormick_bounds="bogus", max_nodes=1000)

    def test_evaluate_midpoint_helpers_are_gone(self):
        """The unsound helpers were deleted, not just left unwired."""
        import discopt._jax.mccormick_nlp as mn

        assert not hasattr(mn, "evaluate_midpoint_bound")
        assert not hasattr(mn, "evaluate_midpoint_bound_batch")


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
        """NLP solving finds the global min of the convex underestimator.

        The NLP mode minimizes cv over the box, so its bound must be <= the value
        of cv at any interior point (here the box midpoint). This is exactly why
        it is sound where the removed "midpoint" mode was not: it returns
        ``min_box cv`` rather than ``cv(mid)``.
        """
        from discopt._jax.mccormick_nlp import solve_mccormick_relaxation_nlp
        from discopt._jax.relaxation_compiler import compile_objective_relaxation

        model = _simple_minimize_model()
        relax_fn = compile_objective_relaxation(model)

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([4.0, 4.0])

        # cv evaluated at the box midpoint (what the removed midpoint mode used).
        mid = 0.5 * (lb + ub)
        cv_mid, _cc = relax_fn(mid, mid, lb, ub)
        cv_mid = float(cv_mid)

        nlp_lb = solve_mccormick_relaxation_nlp(relax_fn, None, None, lb, ub)

        # NLP minimizes cv over the domain, should give <= cv(midpoint)
        assert nlp_lb <= cv_mid + 1e-6

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
        result = model.solve(mccormick_bounds="nlp", max_nodes=500)
        assert result.status in ("optimal", "feasible", "node_limit")

    def test_auto_activates_for_dag_models(self):
        """'auto' mode should activate a valid bound path for DAG models."""
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
        result = model.solve(mccormick_bounds="nlp", max_nodes=500)
        assert result.status in ("optimal", "feasible")
        if result.status == "optimal":
            # (x-1)^2 + (y-2)^2, x integer: optimal x=1, y=2, obj=0
            assert result.objective is not None
            np.testing.assert_allclose(result.objective, 0.0, atol=1e-3)
