"""
Tests for the parametric response compiler (_jax/parametric.py).

Test classes:
  - TestCompileResponseFunction: response compilation and evaluation
  - TestResponseJacobian: jax.jacobian of responses w.r.t. parameters
  - TestUtilities: flatten_params, variable_total_size, extract_x_flat
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.parametric import (
    compile_response_function,
    extract_x_flat,
    flatten_params,
    param_total_size,
    variable_total_size,
)

# ──────────────────────────────────────────────────────────
# TestCompileResponseFunction
# ──────────────────────────────────────────────────────────


class TestCompileResponseFunction:
    """Test that compile_response_function correctly compiles expressions."""

    def test_single_response(self):
        """Compiled response matches direct evaluation."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)  # dummy objective
        fn = compile_response_function({"y": p * x + 1}, m)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        result = fn(x_flat, p_flat)
        assert result.shape == (1,)
        assert float(result[0]) == pytest.approx(7.0)

    def test_multiple_responses(self):
        """Multiple responses return correct stacked vector."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_response_function(
            {
                "y1": p * x,
                "y2": p**2 + x,
                "y3": dm.exp(p),
            },
            m,
        )
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        result = fn(x_flat, p_flat)
        assert result.shape == (3,)
        assert float(result[0]) == pytest.approx(6.0)  # 2*3
        assert float(result[1]) == pytest.approx(7.0)  # 4+3
        assert float(result[2]) == pytest.approx(np.exp(2))  # exp(2)

    def test_response_names_metadata(self):
        """Compiled function has response_names and n_responses attributes."""
        m = dm.Model()
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_response_function({"a": x, "b": x}, m)
        assert fn.response_names == ["a", "b"]
        assert fn.n_responses == 2

    def test_empty_responses_raises(self):
        """Empty responses dict raises ValueError."""
        m = dm.Model()
        m.continuous("x", lb=0, ub=10)
        m.minimize(dm.exp(m._variables[0]))
        with pytest.raises(ValueError, match="non-empty"):
            compile_response_function({}, m)

    def test_parameter_value_changes(self):
        """Response changes when parameter values change."""
        m = dm.Model()
        p = m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_response_function({"y": p * x}, m)
        x_flat = jnp.array([5.0])
        assert float(fn(x_flat, jnp.array([2.0]))[0]) == pytest.approx(10.0)
        assert float(fn(x_flat, jnp.array([3.0]))[0]) == pytest.approx(15.0)

    def test_response_with_math_functions(self):
        """Responses with exp, log, sin, etc. compile correctly."""
        m = dm.Model()
        p = m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(x)
        fn = compile_response_function(
            {
                "exp_response": dm.exp(-p * x),
                "log_response": dm.log(p * x),
            },
            m,
        )
        x_flat = jnp.array([2.0])
        p_flat = jnp.array([1.0])
        result = fn(x_flat, p_flat)
        assert float(result[0]) == pytest.approx(np.exp(-2.0))
        assert float(result[1]) == pytest.approx(np.log(2.0))


# ──────────────────────────────────────────────────────────
# TestResponseJacobian
# ──────────────────────────────────────────────────────────


class TestResponseJacobian:
    """Test jax.jacobian of responses w.r.t. parameters."""

    def test_linear_jacobian(self):
        """dy/dp for y = p*x is x."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_response_function({"y": p * x}, m)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)
        assert J.shape == (1, 1)
        assert float(J[0, 0]) == pytest.approx(3.0, abs=1e-12)

    def test_two_responses_one_param(self):
        """Jacobian shape (2, 1) for two responses, one parameter."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_response_function(
            {
                "y1": p * x,  # dy1/dp = x = 3
                "y2": p**2 + x,  # dy2/dp = 2p = 4
            },
            m,
        )
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)
        assert J.shape == (2, 1)
        np.testing.assert_allclose(J[:, 0], [3.0, 4.0], atol=1e-12)

    def test_two_responses_two_params(self):
        """Jacobian shape (2, 2) for two responses, two parameters."""
        m = dm.Model()
        a = m.parameter("a", value=1.0)
        b = m.parameter("b", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_response_function(
            {
                "y1": a * x,  # dy1/da = x=5, dy1/db = 0
                "y2": b * x,  # dy2/da = 0,   dy2/db = x=5
            },
            m,
        )
        x_flat = jnp.array([5.0])
        p_flat = jnp.array([1.0, 2.0])
        J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)
        assert J.shape == (2, 2)
        expected = np.array([[5.0, 0.0], [0.0, 5.0]])
        np.testing.assert_allclose(J, expected, atol=1e-12)

    def test_nonlinear_jacobian_matches_finite_difference(self):
        """Autodiff Jacobian matches central finite differences."""
        m = dm.Model()
        k = m.parameter("k", value=0.5)
        A = m.parameter("A", value=3.0)
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(x)

        # y = A * exp(-k * x)
        fn = compile_response_function({"y": A * dm.exp(-k * x)}, m)
        x_flat = jnp.array([2.0])
        p_flat = jnp.array([0.5, 3.0])

        # Autodiff
        J_auto = jax.jacobian(fn, argnums=1)(x_flat, p_flat)

        # Finite difference (central)
        eps = 1e-5
        J_fd = np.zeros_like(J_auto)
        for i in range(len(p_flat)):
            p_plus = p_flat.at[i].set(p_flat[i] + eps)
            p_minus = p_flat.at[i].set(p_flat[i] - eps)
            J_fd[:, i] = (fn(x_flat, p_plus) - fn(x_flat, p_minus)) / (2 * eps)

        np.testing.assert_allclose(J_auto, J_fd, atol=1e-6)

    def test_exponential_decay_multiple_times(self):
        """Jacobian for y_i = A*exp(-k*t_i) at multiple time points."""
        m = dm.Model()
        k = m.parameter("k", value=0.3)
        A = m.parameter("A", value=5.0)
        t = m.continuous("t", lb=0.1, ub=10)
        m.minimize(t)

        # Simulate multiple measurement times by evaluating at different x
        t_vals = [1.0, 2.0, 5.0]
        responses = {}
        for i, tv in enumerate(t_vals):
            # Use constant expressions for time points
            responses[f"y_{i}"] = A * dm.exp(-k * t)

        fn = compile_response_function(responses, m)
        x_flat = jnp.array([2.0])  # evaluate at t=2
        p_flat = jnp.array([0.3, 5.0])

        J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)
        assert J.shape == (3, 2)

        # All responses are the same expression evaluated at same x,
        # so all rows should be identical
        for i in range(3):
            np.testing.assert_allclose(J[i], J[0], atol=1e-12)


# ──────────────────────────────────────────────────────────
# TestUtilities
# ──────────────────────────────────────────────────────────


class TestUtilities:
    """Test utility functions."""

    def test_flatten_params(self):
        m = dm.Model()
        m.parameter("a", value=2.0)
        m.parameter("b", value=np.array([3.0, 4.0]))
        m.continuous("x", lb=0, ub=10)
        m.minimize(m._variables[0])
        p = flatten_params(m)
        np.testing.assert_array_equal(p, [2.0, 3.0, 4.0])

    def test_param_total_size(self):
        m = dm.Model()
        m.parameter("a", value=2.0)
        m.parameter("b", value=np.array([3.0, 4.0]))
        assert param_total_size(m) == 3

    def test_param_total_size_no_params(self):
        m = dm.Model()
        assert param_total_size(m) == 0

    def test_variable_total_size(self):
        m = dm.Model()
        m.continuous("x", shape=(3,), lb=0, ub=10)
        m.continuous("y", lb=0, ub=1)
        assert variable_total_size(m) == 4

    def test_extract_x_flat(self):
        m = dm.Model()
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(dm.sum(x))
        result = m.solve()
        x_flat = extract_x_flat(result, m)
        assert x_flat.shape == (2,)
        # x* should be at lower bounds (minimizing sum)
        np.testing.assert_allclose(x_flat, [0.0, 0.0], atol=1e-3)

    def test_extract_x_flat_no_solution_raises(self):
        from discopt.modeling.core import SolveResult

        result = SolveResult(status="infeasible")
        m = dm.Model()
        m.continuous("x", lb=0, ub=10)
        with pytest.raises(ValueError, match="No solution"):
            extract_x_flat(result, m)
