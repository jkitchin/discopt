"""Tests for the JAX NLP Evaluator."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/jaxminlp_benchmarks")
sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

from jaxminlp_api.core import Constant, Model, ObjectiveSense
from jaxminlp_api import examples

from discopt._jax.nlp_evaluator import NLPEvaluator


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _flat_size(model: Model) -> int:
    """Total size of the flat variable vector."""
    return sum(v.size for v in model._variables)


def _random_interior_point(model: Model, rng: np.random.Generator) -> np.ndarray:
    """Generate a random point strictly inside variable bounds."""
    parts = []
    for v in model._variables:
        lb = np.clip(v.lb.flatten(), -1e3, 1e3)
        ub = np.clip(v.ub.flatten(), -1e3, 1e3)
        width = np.maximum(ub - lb, 1e-6)
        vals = lb + rng.uniform(0.05, 0.95, size=v.size) * width
        parts.append(vals)
    return np.concatenate(parts).astype(np.float64)


# ─────────────────────────────────────────────────────────────
# Test 1: Objective evaluation
# ─────────────────────────────────────────────────────────────

class TestObjectiveEvaluation:
    def test_simple_minlp_objective(self):
        """For simple_minlp at a known point, verify objective matches manual calculation."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        # Variables: x1 (scalar), x2 (scalar), x3 (binary scalar)
        # Objective: x1**2 + x2**2 + x3
        x = np.array([1.0, 2.0, 0.5])
        obj = ev.evaluate_objective(x)
        expected = 1.0**2 + 2.0**2 + 0.5  # = 5.5
        assert np.isclose(obj, expected, atol=1e-10)

    def test_objective_at_zeros(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([0.0, 0.0, 0.0])
        obj = ev.evaluate_objective(x)
        assert np.isclose(obj, 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Test 2: Gradient vs finite differences
# ─────────────────────────────────────────────────────────────

class TestGradient:
    def test_gradient_vs_finite_diff(self):
        """At 50 random interior points, verify gradient matches finite differences."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        rng = np.random.default_rng(42)
        n = ev.n_variables
        eps = 1e-5

        for _ in range(50):
            x = _random_interior_point(m, rng)
            grad = ev.evaluate_gradient(x)

            fd_grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                fd_grad[i] = (ev.evaluate_objective(x_plus) - ev.evaluate_objective(x_minus)) / (2 * eps)

            assert np.allclose(grad, fd_grad, atol=1e-6), (
                f"Gradient mismatch at {x}: AD={grad}, FD={fd_grad}"
            )

    def test_gradient_shape(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        grad = ev.evaluate_gradient(x)
        assert grad.shape == (3,)

    def test_gradient_known_value(self):
        """For x1**2 + x2**2 + x3, gradient is [2*x1, 2*x2, 1]."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([3.0, 4.0, 1.0])
        grad = ev.evaluate_gradient(x)
        expected = np.array([6.0, 8.0, 1.0])
        assert np.allclose(grad, expected, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Test 3: Hessian vs finite differences
# ─────────────────────────────────────────────────────────────

class TestHessian:
    def test_hessian_vs_finite_diff(self):
        """At 20 random points, verify Hessian matches numerical Hessian."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        rng = np.random.default_rng(123)
        n = ev.n_variables
        eps = 1e-4

        for _ in range(20):
            x = _random_interior_point(m, rng)
            hess = ev.evaluate_hessian(x)

            fd_hess = np.zeros((n, n))
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                g_plus = ev.evaluate_gradient(x_plus)
                g_minus = ev.evaluate_gradient(x_minus)
                fd_hess[i, :] = (g_plus - g_minus) / (2 * eps)

            assert np.allclose(hess, fd_hess, atol=1e-4), (
                f"Hessian mismatch:\nAD:\n{hess}\nFD:\n{fd_hess}"
            )

    def test_hessian_shape(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        hess = ev.evaluate_hessian(x)
        assert hess.shape == (3, 3)

    def test_hessian_known_value(self):
        """For x1**2 + x2**2 + x3, Hessian is diag(2, 2, 0)."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 1.0, 1.0])
        hess = ev.evaluate_hessian(x)
        expected = np.diag([2.0, 2.0, 0.0])
        assert np.allclose(hess, expected, atol=1e-10)

    def test_hessian_symmetry(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        rng = np.random.default_rng(99)
        x = _random_interior_point(m, rng)
        hess = ev.evaluate_hessian(x)
        assert np.allclose(hess, hess.T, atol=1e-12)


# ─────────────────────────────────────────────────────────────
# Test 4: Constraint evaluation
# ─────────────────────────────────────────────────────────────

class TestConstraints:
    def test_constraint_evaluation(self):
        """Verify constraint bodies evaluate correctly for simple_minlp."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        # Constraints:
        #   x1 + x2 >= 1  =>  body = 1 - x1 - x2 (normalized: (1-x1-x2) <= 0)
        #   x1**2 + x2 <= 3  =>  body = x1**2 + x2 - 3
        x = np.array([1.0, 2.0, 0.5])
        cons = ev.evaluate_constraints(x)
        assert cons.shape == (2,)
        # Constraint 0: (1 - x1 - x2) - 0 = 1 - 1 - 2 = -2
        assert np.isclose(cons[0], -2.0, atol=1e-10)
        # Constraint 1: (x1**2 + x2 - 3) - 0 = 1 + 2 - 3 = 0
        assert np.isclose(cons[1], 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Test 5: Jacobian vs finite differences
# ─────────────────────────────────────────────────────────────

class TestJacobian:
    def test_jacobian_vs_finite_diff(self):
        """Verify constraint Jacobian matches numerical Jacobian."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        rng = np.random.default_rng(77)
        n = ev.n_variables
        eps = 1e-5

        for _ in range(20):
            x = _random_interior_point(m, rng)
            jac = ev.evaluate_jacobian(x)

            fd_jac = np.zeros((ev.n_constraints, n))
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                c_plus = ev.evaluate_constraints(x_plus)
                c_minus = ev.evaluate_constraints(x_minus)
                fd_jac[:, i] = (c_plus - c_minus) / (2 * eps)

            assert np.allclose(jac, fd_jac, atol=1e-5), (
                f"Jacobian mismatch:\nAD:\n{jac}\nFD:\n{fd_jac}"
            )

    def test_jacobian_shape(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        jac = ev.evaluate_jacobian(x)
        assert jac.shape == (2, 3)

    def test_jacobian_known_value(self):
        """Jacobian of simple_minlp constraints at x = [1, 2, 0.5]:
        Constraint 0 body: (1 - x1 - x2), d/dx = [-1, -1, 0]
        Constraint 1 body: (x1**2 + x2 - 3), d/dx = [2*x1, 1, 0] = [2, 1, 0]
        """
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        jac = ev.evaluate_jacobian(x)
        expected = np.array([
            [-1.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
        ])
        assert np.allclose(jac, expected, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Test 6: JIT warmup performance
# ─────────────────────────────────────────────────────────────

class TestJITPerformance:
    def test_jit_warmup_speedup(self):
        """Second call to evaluate_objective should be >=10x faster than first."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])

        # First call triggers JIT compilation
        t0 = time.perf_counter()
        ev.evaluate_objective(x)
        first_time = time.perf_counter() - t0

        # Subsequent calls use cached JIT
        n_calls = 100
        t0 = time.perf_counter()
        for _ in range(n_calls):
            ev.evaluate_objective(x)
        avg_second = (time.perf_counter() - t0) / n_calls

        # First call should be significantly slower due to JIT compilation
        assert first_time > avg_second * 10, (
            f"Expected 10x warmup overhead: first={first_time:.6f}s, "
            f"avg_subsequent={avg_second:.6f}s"
        )


# ─────────────────────────────────────────────────────────────
# Test 7: All 7 examples — no NaN, no Inf
# ─────────────────────────────────────────────────────────────

_EXAMPLE_FACTORIES = [
    examples.example_simple_minlp,
    examples.example_pooling_haverly,
    examples.example_process_synthesis,
    examples.example_portfolio,
    examples.example_reactor_design,
    examples.example_facility_location,
    examples.example_parametric,
]


class TestAllExamples:
    @pytest.mark.parametrize("factory", _EXAMPLE_FACTORIES, ids=[
        "simple_minlp", "pooling_haverly", "process_synthesis",
        "portfolio", "reactor_design", "facility_location", "parametric",
    ])
    def test_objective_no_nan_inf(self, factory):
        """Evaluate objective for each example model — no NaN, no Inf."""
        m = factory()
        ev = NLPEvaluator(m)
        rng = np.random.default_rng(42)
        x = _random_interior_point(m, rng)
        obj = ev.evaluate_objective(x)
        assert np.isfinite(obj), f"Non-finite objective: {obj}"

    @pytest.mark.parametrize("factory", _EXAMPLE_FACTORIES, ids=[
        "simple_minlp", "pooling_haverly", "process_synthesis",
        "portfolio", "reactor_design", "facility_location", "parametric",
    ])
    def test_gradient_no_nan_inf(self, factory):
        """Evaluate gradient for each example model — no NaN, no Inf."""
        m = factory()
        ev = NLPEvaluator(m)
        rng = np.random.default_rng(42)
        x = _random_interior_point(m, rng)
        grad = ev.evaluate_gradient(x)
        assert np.all(np.isfinite(grad)), f"Non-finite gradient: {grad}"


# ─────────────────────────────────────────────────────────────
# Test 8: Return types are numpy arrays
# ─────────────────────────────────────────────────────────────

class TestReturnTypes:
    def test_objective_returns_float(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        obj = ev.evaluate_objective(x)
        assert isinstance(obj, float)

    def test_gradient_returns_numpy(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        grad = ev.evaluate_gradient(x)
        assert isinstance(grad, np.ndarray)
        assert not isinstance(grad, jnp.ndarray.__class__), "Should be numpy, not jax array"

    def test_hessian_returns_numpy(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        hess = ev.evaluate_hessian(x)
        assert isinstance(hess, np.ndarray)

    def test_constraints_returns_numpy(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        cons = ev.evaluate_constraints(x)
        assert isinstance(cons, np.ndarray)

    def test_jacobian_returns_numpy(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        x = np.array([1.0, 2.0, 0.5])
        jac = ev.evaluate_jacobian(x)
        assert isinstance(jac, np.ndarray)


# ─────────────────────────────────────────────────────────────
# Test 9: Variable bounds
# ─────────────────────────────────────────────────────────────

class TestVariableBounds:
    def test_bounds_simple_minlp(self):
        """Verify variable_bounds returns correct lb/ub."""
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        lb, ub = ev.variable_bounds
        # x1: [0, 5], x2: [0, 5], x3: [0, 1] (binary)
        assert np.allclose(lb, [0.0, 0.0, 0.0])
        assert np.allclose(ub, [5.0, 5.0, 1.0])

    def test_bounds_shape(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        lb, ub = ev.variable_bounds
        assert lb.shape == (ev.n_variables,)
        assert ub.shape == (ev.n_variables,)

    def test_bounds_are_numpy(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        lb, ub = ev.variable_bounds
        assert isinstance(lb, np.ndarray)
        assert isinstance(ub, np.ndarray)


# ─────────────────────────────────────────────────────────────
# Test 10: Zero constraints
# ─────────────────────────────────────────────────────────────

class TestZeroConstraints:
    def test_no_constraints(self):
        """Model with only objective, no constraints."""
        m = Model("unconstrained")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x ** 2)
        ev = NLPEvaluator(m)

        assert ev.n_constraints == 0

        test_x = np.array([3.0])
        cons = ev.evaluate_constraints(test_x)
        assert cons.shape == (0,)
        assert isinstance(cons, np.ndarray)

        jac = ev.evaluate_jacobian(test_x)
        assert jac.shape == (0, 1)
        assert isinstance(jac, np.ndarray)

    def test_unconstrained_objective_and_gradient(self):
        """Unconstrained model still computes objective and gradient."""
        m = Model("unconstrained")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x ** 2)
        ev = NLPEvaluator(m)

        test_x = np.array([3.0])
        assert np.isclose(ev.evaluate_objective(test_x), 9.0)
        assert np.allclose(ev.evaluate_gradient(test_x), [6.0])


# ─────────────────────────────────────────────────────────────
# Test 11: Maximize objective (negation)
# ─────────────────────────────────────────────────────────────

class TestMaximize:
    def test_maximize_negates_objective(self):
        """Model with maximize => evaluator negates internally."""
        m = Model("maximize_test")
        x = m.continuous("x", lb=0, ub=10)
        m.maximize(x ** 2 + 3 * x)
        ev = NLPEvaluator(m)

        test_x = np.array([2.0])
        # Original: x^2 + 3x = 4 + 6 = 10
        # Negated for minimization: -10
        assert np.isclose(ev.evaluate_objective(test_x), -10.0)

    def test_maximize_negates_gradient(self):
        """Gradient should also be negated for maximize."""
        m = Model("maximize_test")
        x = m.continuous("x", lb=0, ub=10)
        m.maximize(x ** 2 + 3 * x)
        ev = NLPEvaluator(m)

        test_x = np.array([2.0])
        # d/dx (x^2 + 3x) = 2x + 3 = 7
        # Negated: -7
        grad = ev.evaluate_gradient(test_x)
        assert np.isclose(grad[0], -7.0)

    def test_maximize_negates_hessian(self):
        """Hessian should also be negated for maximize."""
        m = Model("maximize_test")
        x = m.continuous("x", lb=0, ub=10)
        m.maximize(x ** 2 + 3 * x)
        ev = NLPEvaluator(m)

        test_x = np.array([2.0])
        # d2/dx2 (x^2 + 3x) = 2
        # Negated: -2
        hess = ev.evaluate_hessian(test_x)
        assert np.isclose(hess[0, 0], -2.0)

    def test_minimize_does_not_negate(self):
        """Minimize should not negate."""
        m = Model("minimize_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x ** 2 + 3 * x)
        ev = NLPEvaluator(m)

        test_x = np.array([2.0])
        assert np.isclose(ev.evaluate_objective(test_x), 10.0)


# ─────────────────────────────────────────────────────────────
# Test: n_variables and n_constraints properties
# ─────────────────────────────────────────────────────────────

class TestProperties:
    def test_n_variables(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        assert ev.n_variables == 3  # x1, x2, x3

    def test_n_constraints(self):
        m = examples.example_simple_minlp()
        ev = NLPEvaluator(m)
        assert ev.n_constraints == 2

    def test_no_objective_raises(self):
        m = Model("empty")
        m.continuous("x", lb=0, ub=1)
        with pytest.raises(ValueError, match="no objective"):
            NLPEvaluator(m)
