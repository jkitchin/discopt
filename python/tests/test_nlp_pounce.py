"""Tests for the POUNCE NLP wrapper (pure-Rust Ipopt port).

Mirrors ``test_nlp_ipopt.py``: the POUNCE backend reuses the same callback
adapter and status map as the cyipopt backend, so it must produce the same
results on the same models.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

pounce = pytest.importorskip("pounce")

from discopt._jax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt.modeling import examples  # noqa: E402
from discopt.modeling.core import Model  # noqa: E402
from discopt.solvers import NLPResult, SolveStatus  # noqa: E402
from discopt.solvers.nlp_pounce import solve_nlp, solve_nlp_from_model  # noqa: E402

pytestmark = pytest.mark.requires_pounce


# ─────────────────────────────────────────────────────────────
# Test 1: Simple unconstrained NLP
# ─────────────────────────────────────────────────────────────


class TestUnconstrainedNLP:
    def test_minimize_quadratic(self):
        """min x^2 + y^2 => optimal at (0, 0) with obj = 0."""
        m = Model("unconstrained_quad")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)

        result = solve_nlp_from_model(m, x0=np.array([5.0, 3.0]))
        assert result.status == SolveStatus.OPTIMAL
        assert np.allclose(result.x, [0.0, 0.0], atol=1e-6)
        assert abs(result.objective) < 1e-10

    def test_minimize_shifted_quadratic(self):
        """min (x-3)^2 + (y+1)^2 => optimal at (3, -1)."""
        m = Model("shifted_quad")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize((x - 3) ** 2 + (y + 1) ** 2)

        result = solve_nlp_from_model(m, x0=np.array([0.0, 0.0]))
        assert result.status == SolveStatus.OPTIMAL
        assert np.allclose(result.x, [3.0, -1.0], atol=1e-5)
        assert abs(result.objective) < 1e-8


# ─────────────────────────────────────────────────────────────
# Test 2: Constrained NLP
# ─────────────────────────────────────────────────────────────


class TestConstrainedNLP:
    def test_constrained_quadratic(self):
        """min x^2 + y^2 s.t. x + y >= 1 => optimal at (0.5, 0.5)."""
        m = Model("constrained_quad")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = solve_nlp_from_model(m, x0=np.array([5.0, 5.0]))
        assert result.status == SolveStatus.OPTIMAL
        assert np.allclose(result.x, [0.5, 0.5], atol=1e-5)
        assert abs(result.objective - 0.5) < 1e-6

    def test_equality_constraint(self):
        """min x^2 + y^2 s.t. x + y == 2 => optimal at (1, 1)."""
        m = Model("eq_constrained")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y == 2)

        result = solve_nlp_from_model(m, x0=np.array([3.0, 1.0]))
        assert result.status == SolveStatus.OPTIMAL
        assert np.allclose(result.x, [1.0, 1.0], atol=1e-5)
        assert abs(result.objective - 2.0) < 1e-6

    def test_multipliers_returned(self):
        """Verify constraint multipliers are returned."""
        m = Model("multiplier_test")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = solve_nlp_from_model(m, x0=np.array([5.0, 5.0]))
        assert result.multipliers is not None
        assert result.multipliers.shape == (1,)


# ─────────────────────────────────────────────────────────────
# Test 3: Example models (continuous relaxation)
# ─────────────────────────────────────────────────────────────

_SOLVABLE_EXAMPLES = [
    ("simple_minlp", examples.example_simple_minlp),
    ("pooling_haverly", examples.example_pooling_haverly),
]


class TestExampleModels:
    @pytest.mark.parametrize(
        "name,factory", _SOLVABLE_EXAMPLES, ids=[e[0] for e in _SOLVABLE_EXAMPLES]
    )
    def test_example_converges(self, name, factory):
        """Solve continuous relaxation of example models, verify convergence."""
        m = factory()
        result = solve_nlp_from_model(m, options={"max_iter": 3000})
        assert result.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.ITERATION_LIMIT,
        ), f"Example {name} failed with status {result.status}"
        assert result.x is not None
        assert np.all(np.isfinite(result.x))
        assert np.isfinite(result.objective)


# ─────────────────────────────────────────────────────────────
# Test 4: Infeasible problem
# ─────────────────────────────────────────────────────────────


class TestInfeasible:
    def test_contradictory_constraints(self):
        """Contradictory constraints should yield INFEASIBLE status."""
        m = Model("infeasible")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x**2)
        m.subject_to(x <= 1)
        m.subject_to(x >= 5)

        result = solve_nlp_from_model(m, x0=np.array([0.0]))
        assert result.status in (SolveStatus.INFEASIBLE, SolveStatus.ERROR)


# ─────────────────────────────────────────────────────────────
# Test 5: solve_nlp_from_model convenience
# ─────────────────────────────────────────────────────────────


class TestConvenience:
    def test_from_model_no_x0(self):
        """solve_nlp_from_model generates default x0 from bounds midpoint."""
        m = Model("convenience")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = solve_nlp_from_model(m)
        assert result.status == SolveStatus.OPTIMAL
        assert np.allclose(result.x, [0.5, 0.5], atol=1e-5)

    def test_result_type(self):
        """Result should be an NLPResult."""
        m = Model("type_test")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x**2)

        result = solve_nlp_from_model(m)
        assert isinstance(result, NLPResult)

    def test_wall_time_positive(self):
        """Wall time should be recorded."""
        m = Model("time_test")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x**2)

        result = solve_nlp_from_model(m)
        assert result.wall_time > 0.0

    def test_iterations_reported(self):
        """POUNCE exposes an iteration count (unlike the cyipopt wrapper)."""
        m = Model("iter_test")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = solve_nlp_from_model(m, x0=np.array([8.0, 8.0]))
        assert result.iterations > 0


# ─────────────────────────────────────────────────────────────
# Test 6: Custom options
# ─────────────────────────────────────────────────────────────


class TestOptions:
    def test_max_iter_respected(self):
        """Setting max_iter=1 should hit iteration limit."""
        m = Model("options_test")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        result = solve_nlp_from_model(m, x0=np.array([8.0, 8.0]), options={"max_iter": 1})
        assert result.status == SolveStatus.ITERATION_LIMIT

    def test_tol_option(self):
        """Custom tolerance should be accepted without error."""
        m = Model("tol_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)

        result = solve_nlp_from_model(m, options={"tol": 1e-12})
        assert result.status == SolveStatus.OPTIMAL


# ─────────────────────────────────────────────────────────────
# Test 7: Gradient/Hessian callbacks are used (not finite-diff)
# ─────────────────────────────────────────────────────────────


class TestCallbacksUsed:
    def test_gradient_callback_called(self):
        """Verify the gradient callback is called during solve."""
        m = Model("grad_callback")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        ev = NLPEvaluator(m)

        grad_count = [0]
        original_grad = ev.evaluate_gradient

        def counting_grad(x):
            grad_count[0] += 1
            return original_grad(x)

        ev.evaluate_gradient = counting_grad

        result = solve_nlp(ev, np.array([5.0, 5.0]))
        assert result.status == SolveStatus.OPTIMAL
        assert grad_count[0] > 0, "Gradient callback was never called"

    def test_hessian_callback_called(self):
        """Verify the Hessian callback is called during solve."""
        m = Model("hess_callback")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        ev = NLPEvaluator(m)

        hess_count = [0]
        original_hess = ev.evaluate_lagrangian_hessian

        def counting_hess(x, obj_factor, lambda_):
            hess_count[0] += 1
            return original_hess(x, obj_factor, lambda_)

        ev.evaluate_lagrangian_hessian = counting_hess

        result = solve_nlp(ev, np.array([5.0, 5.0]))
        assert result.status == SolveStatus.OPTIMAL
        assert hess_count[0] > 0, "Lagrangian Hessian callback was never called"


# ─────────────────────────────────────────────────────────────
# Test 8: solve_nlp with explicit constraint_bounds
# ─────────────────────────────────────────────────────────────


class TestExplicitBounds:
    def test_explicit_constraint_bounds(self):
        """Pass constraint_bounds explicitly instead of inferring."""
        m = Model("explicit_bounds")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 1)

        ev = NLPEvaluator(m)
        # Constraint body is (1 - x - y), sense <=, so cl=-inf, cu=0
        result = solve_nlp(
            ev,
            np.array([5.0, 5.0]),
            constraint_bounds=[(-1e20, 0.0)],
        )
        assert result.status == SolveStatus.OPTIMAL
        assert np.allclose(result.x, [0.5, 0.5], atol=1e-5)


# ─────────────────────────────────────────────────────────────
# Test 9: Maximize objective
# ─────────────────────────────────────────────────────────────


class TestMaximize:
    def test_maximize_via_negation(self):
        """NLPEvaluator negates maximize objectives; verify solve works."""
        m = Model("maximize_test")
        x = m.continuous("x", lb=0, ub=5)
        m.maximize(x)  # max x s.t. x in [0, 5] => x* = 5

        result = solve_nlp_from_model(m)
        assert result.status == SolveStatus.OPTIMAL
        assert abs(result.x[0] - 5.0) < 1e-5
