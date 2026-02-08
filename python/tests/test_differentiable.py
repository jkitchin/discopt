"""
Tests for T22: Level 1 differentiable solving via envelope theorem.

Test classes:
  - TestParametricCompiler: parametric DAG compilation produces correct functions
  - TestDifferentiableSolve: differentiable_solve returns correct solutions and gradients
  - TestFiniteDifference: gradient matches finite-difference approximation
  - TestJaxDifferentiableSolve: JAX-native solve_fn works with jax.grad
  - TestComposability: gradient through solve embedded in larger computation
"""

import discopt.modeling as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.differentiable import (
    DiffSolveResult,
    _compile_parametric_objective,
    _flatten_params,
    _make_jax_differentiable_solve,
    _param_total_size,
    differentiable_solve,
)
from discopt.modeling.core import Constant

# ──────────────────────────────────────────────────────────
# TestParametricCompiler
# ──────────────────────────────────────────────────────────


class TestParametricCompiler:
    """Test that the parametric DAG compiler correctly handles parameters."""

    def test_constant_independent_of_params(self):
        m = dm.Model("test")
        m.parameter("p", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = _compile_parametric_objective(m)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([2.0])
        assert float(fn(x_flat, p_flat)) == pytest.approx(3.0)

    def test_parameter_appears_in_objective(self):
        m = dm.Model("test")
        p = m.parameter("p", value=5.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(p * x)
        fn = _compile_parametric_objective(m)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([5.0])
        assert float(fn(x_flat, p_flat)) == pytest.approx(15.0)

    def test_parameter_value_changes(self):
        m = dm.Model("test")
        p = m.parameter("p", value=5.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(p * x)
        fn = _compile_parametric_objective(m)
        x_flat = jnp.array([3.0])
        # Different p value
        p_flat = jnp.array([10.0])
        assert float(fn(x_flat, p_flat)) == pytest.approx(30.0)

    def test_multiple_parameters(self):
        m = dm.Model("test")
        a = m.parameter("a", value=2.0)
        b = m.parameter("b", value=3.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(a * x + b)
        fn = _compile_parametric_objective(m)
        x_flat = jnp.array([4.0])
        p_flat = jnp.array([2.0, 3.0])
        assert float(fn(x_flat, p_flat)) == pytest.approx(11.0)

    def test_flatten_params(self):
        m = dm.Model("test")
        m.parameter("a", value=2.0)
        m.parameter("b", value=np.array([3.0, 4.0]))
        m.continuous("x", lb=0, ub=10)
        m.minimize(Constant(0.0))
        p_flat = _flatten_params(m)
        np.testing.assert_array_equal(p_flat, [2.0, 3.0, 4.0])

    def test_param_total_size(self):
        m = dm.Model("test")
        m.parameter("a", value=2.0)
        m.parameter("b", value=np.array([3.0, 4.0]))
        assert _param_total_size(m) == 3

    def test_parametric_grad_wrt_p(self):
        """jax.grad w.r.t. p_flat produces correct derivatives."""
        m = dm.Model("test")
        m.parameter("p", value=5.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(m._parameters[0] * x)
        fn = _compile_parametric_objective(m)
        # df/dp = x = 3.0
        grad_fn = jax.grad(fn, argnums=1)
        x_flat = jnp.array([3.0])
        p_flat = jnp.array([5.0])
        g = grad_fn(x_flat, p_flat)
        assert float(g[0]) == pytest.approx(3.0)


# ──────────────────────────────────────────────────────────
# TestDifferentiableSolve
# ──────────────────────────────────────────────────────────


class TestDifferentiableSolve:
    """Test differentiable_solve returns correct solutions and gradients."""

    def test_simple_parametric_lp(self):
        """min p*x s.t. x >= 1, x <= 5, p > 0.

        Optimal: x* = 1 (for p > 0), obj* = p.
        d(obj*)/dp = 1 (since x* = 1 is constant w.r.t. p).
        Actually by envelope: d(obj*)/dp = x* = 1.
        """
        m = dm.Model("param_lp")
        p = m.parameter("price", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(p * x)

        result = differentiable_solve(m)
        assert isinstance(result, DiffSolveResult)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(3.0, abs=1e-4)

        grad = result.gradient(p)
        # d(obj*)/dp = x* = 1.0
        assert float(grad) == pytest.approx(1.0, abs=1e-3)

    def test_parametric_rhs(self):
        """min x s.t. x >= b, x <= 10.

        Optimal: x* = b, obj* = b.
        d(obj*)/db = 1.
        By envelope: dL/db = d(x - lambda*(x - b))/db = lambda.
        Since constraint x >= b is active, lambda = 1 (from stationarity).
        """
        m = dm.Model("param_rhs")
        b = m.parameter("b", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= b)

        result = differentiable_solve(m)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(2.0, abs=1e-3)

        grad = result.gradient(b)
        # d(obj*)/db = 1.0
        assert float(grad) == pytest.approx(1.0, abs=1e-2)

    def test_parametric_quadratic(self):
        """min (x - p)^2 s.t. x in [-10, 10].

        Optimal: x* = p, obj* = 0.
        d(obj*)/dp = 0 (at optimum, objective is 0 for all p in range).
        """
        m = dm.Model("param_quad")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve(m)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0, abs=1e-4)

        grad = result.gradient(p)
        # d(obj*)/dp = 0 (unconstrained optimum doesn't depend on p)
        assert float(grad) == pytest.approx(0.0, abs=1e-3)

    def test_parametric_cost_vector(self):
        """min c1*x1 + c2*x2 s.t. x1 + x2 >= 1, x1,x2 in [0,5].

        For c1 < c2: x1* = 1, x2* = 0, obj* = c1.
        d(obj*)/dc1 = x1* = 1, d(obj*)/dc2 = x2* = 0.
        """
        m = dm.Model("param_cost")
        c = m.parameter("c", value=np.array([1.0, 3.0]))
        x1 = m.continuous("x1", lb=0, ub=5)
        x2 = m.continuous("x2", lb=0, ub=5)
        m.minimize(c[0] * x1 + c[1] * x2)
        m.subject_to(x1 + x2 >= 1)

        result = differentiable_solve(m)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(1.0, abs=1e-3)

        grad = result.gradient(c)
        # d(obj*)/dc1 = x1* = 1, d(obj*)/dc2 = x2* = 0
        assert grad[0] == pytest.approx(1.0, abs=1e-2)
        assert grad[1] == pytest.approx(0.0, abs=1e-2)

    def test_parametric_nlp(self):
        """min x^2 + p*x s.t. x >= 0.

        For p >= 0: x* = 0, obj* = 0.
        For p < 0: x* = -p/2, obj* = -p^2/4.
        d(obj*)/dp = -p/2 when p < 0.
        """
        m = dm.Model("param_nlp")
        p = m.parameter("p", value=-4.0)
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x**2 + p * x)

        result = differentiable_solve(m)
        assert result.status == "optimal"
        # x* = 2.0, obj* = 4 - 8 = -4
        assert result.objective == pytest.approx(-4.0, abs=1e-3)

        grad = result.gradient(p)
        # d(obj*)/dp = x* = 2.0 (by envelope theorem)
        assert float(grad) == pytest.approx(2.0, abs=1e-2)

    def test_rejects_integer_variables(self):
        """differentiable_solve should raise for models with integer vars."""
        m = dm.Model("int_model")
        m.parameter("p", value=1.0)
        m.binary("y")
        m.continuous("x", lb=0, ub=10)
        m.minimize(m._variables[1])

        with pytest.raises(ValueError, match="continuous"):
            differentiable_solve(m)


# ──────────────────────────────────────────────────────────
# TestFiniteDifference
# ──────────────────────────────────────────────────────────


class TestFiniteDifference:
    """Validate gradients against finite-difference approximations."""

    @staticmethod
    def _fd_gradient(model_fn, base_val, eps=1e-5):
        """Compute finite-difference gradient for a model factory.

        model_fn(p_val) -> (model, param) with param.value = p_val.
        base_val: the parameter value around which to compute the FD gradient.
        """
        if np.ndim(base_val) == 0:
            # Scalar parameter
            m_plus, _ = model_fn(float(base_val) + eps)
            r_plus = differentiable_solve(m_plus)
            m_minus, _ = model_fn(float(base_val) - eps)
            r_minus = differentiable_solve(m_minus)
            return (r_plus.objective - r_minus.objective) / (2 * eps)
        else:
            base_arr = np.asarray(base_val, dtype=np.float64)
            grad = np.zeros_like(base_arr)
            for i in range(base_arr.size):
                val_plus = base_arr.copy()
                val_plus.flat[i] += eps
                m_plus, _ = model_fn(val_plus)
                r_plus = differentiable_solve(m_plus)

                val_minus = base_arr.copy()
                val_minus.flat[i] -= eps
                m_minus, _ = model_fn(val_minus)
                r_minus = differentiable_solve(m_minus)

                grad.flat[i] = (r_plus.objective - r_minus.objective) / (2 * eps)
            return grad

    def test_fd_simple_lp(self):
        """FD validation for min p*x s.t. x >= 1."""

        def make_model(p_val):
            m = dm.Model("fd_lp")
            p = m.parameter("p", value=p_val)
            x = m.continuous("x", lb=1, ub=5)
            m.minimize(p * x)
            return m, p

        m, p = make_model(3.0)
        result = differentiable_solve(m)
        analytic_grad = float(result.gradient(p))
        fd_grad = self._fd_gradient(make_model, 3.0)

        assert analytic_grad == pytest.approx(fd_grad, rel=1e-3)

    def test_fd_parametric_rhs(self):
        """FD validation for min x s.t. x >= b."""

        def make_model(b_val):
            m = dm.Model("fd_rhs")
            b = m.parameter("b", value=b_val)
            x = m.continuous("x", lb=0, ub=10)
            m.minimize(x)
            m.subject_to(x >= b)
            return m, b

        m, b = make_model(2.0)
        result = differentiable_solve(m)
        analytic_grad = float(result.gradient(b))
        fd_grad = self._fd_gradient(make_model, 2.0)

        assert analytic_grad == pytest.approx(fd_grad, rel=1e-2)

    def test_fd_parametric_quadratic(self):
        """FD validation for min x^2 + p*x, x >= 0."""

        def make_model(p_val):
            m = dm.Model("fd_quad")
            p = m.parameter("p", value=p_val)
            x = m.continuous("x", lb=0, ub=100)
            m.minimize(x**2 + p * x)
            return m, p

        m, p = make_model(-4.0)
        result = differentiable_solve(m)
        analytic_grad = float(result.gradient(p))
        fd_grad = self._fd_gradient(make_model, -4.0)

        assert analytic_grad == pytest.approx(fd_grad, rel=1e-2)

    def test_fd_cost_vector(self):
        """FD validation for parametric cost vector."""

        def make_model(c_val):
            m = dm.Model("fd_cost")
            c = m.parameter("c", value=np.asarray(c_val, dtype=np.float64))
            x1 = m.continuous("x1", lb=0, ub=5)
            x2 = m.continuous("x2", lb=0, ub=5)
            m.minimize(c[0] * x1 + c[1] * x2)
            m.subject_to(x1 + x2 >= 1)
            return m, c

        m, c = make_model(np.array([1.0, 3.0]))
        result = differentiable_solve(m)
        analytic_grad = result.gradient(c)
        fd_grad = self._fd_gradient(make_model, np.array([1.0, 3.0]))

        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-2, atol=1e-3)

    def test_fd_nonlinear_parametric(self):
        """FD validation for min exp(p*x) + x^2, x in [-5, 5]."""

        def make_model(p_val):
            m = dm.Model("fd_nonlinear")
            p = m.parameter("p", value=p_val)
            x = m.continuous("x", lb=-5, ub=5)
            m.minimize(dm.exp(p * x) + x**2)
            return m, p

        m, p = make_model(0.5)
        result = differentiable_solve(m)
        analytic_grad = float(result.gradient(p))
        fd_grad = self._fd_gradient(make_model, 0.5)

        assert analytic_grad == pytest.approx(fd_grad, rel=1e-2)


# ──────────────────────────────────────────────────────────
# TestJaxDifferentiableSolve
# ──────────────────────────────────────────────────────────


class TestJaxDifferentiableSolve:
    """Test the JAX-native differentiable solve via custom_jvp."""

    def test_forward_pass(self):
        """Forward pass returns correct objective."""
        m = dm.Model("jax_fwd")
        m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)
        p_flat = jnp.array([3.0])
        obj = solve_fn(p_flat)
        assert float(obj) == pytest.approx(3.0, abs=1e-3)

    def test_jvp(self):
        """JVP returns correct tangent."""
        m = dm.Model("jax_jvp")
        m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)
        p_flat = jnp.array([3.0])
        p_dot = jnp.array([1.0])

        primal, tangent = jax.jvp(solve_fn, (p_flat,), (p_dot,))
        assert float(primal) == pytest.approx(3.0, abs=1e-3)
        # d(obj*)/dp = x* = 1.0
        assert float(tangent) == pytest.approx(1.0, abs=1e-2)

    def test_grad(self):
        """jax.grad through the solve returns correct gradient."""
        m = dm.Model("jax_grad")
        m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)
        grad_fn = jax.grad(solve_fn)
        p_flat = jnp.array([3.0])
        grad = grad_fn(p_flat)
        # d(obj*)/dp = x* = 1.0
        assert float(grad[0]) == pytest.approx(1.0, abs=1e-2)

    def test_grad_quadratic(self):
        """jax.grad for quadratic parametric NLP."""
        m = dm.Model("jax_quad")
        m.parameter("p", value=-4.0)
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x**2 + m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)
        grad_fn = jax.grad(solve_fn)
        p_flat = jnp.array([-4.0])
        grad = grad_fn(p_flat)
        # d(obj*)/dp = x* = 2.0
        assert float(grad[0]) == pytest.approx(2.0, abs=1e-1)


# ──────────────────────────────────────────────────────────
# TestComposability
# ──────────────────────────────────────────────────────────


class TestComposability:
    """Test that the differentiable solve composes with other JAX operations."""

    def test_solve_in_larger_computation(self):
        """Gradient through: loss = solve(p)^2."""
        m = dm.Model("compose")
        m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)

        def loss(p_flat):
            obj_star = solve_fn(p_flat)
            return obj_star**2

        grad_fn = jax.grad(loss)
        p_flat = jnp.array([3.0])
        grad = grad_fn(p_flat)
        # obj* = p (since x*=1), loss = p^2, dloss/dp = 2p = 6
        assert float(grad[0]) == pytest.approx(6.0, abs=0.5)

    def test_solve_plus_regularization(self):
        """Gradient through: loss = solve(p) + 0.5*p^2."""
        m = dm.Model("compose_reg")
        m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)

        def loss(p_flat):
            obj_star = solve_fn(p_flat)
            return obj_star + 0.5 * p_flat[0] ** 2

        grad_fn = jax.grad(loss)
        p_flat = jnp.array([3.0])
        grad = grad_fn(p_flat)
        # obj* = p, dloss/dp = 1 + p = 4
        assert float(grad[0]) == pytest.approx(4.0, abs=0.5)

    def test_solve_with_scaling(self):
        """Gradient through: loss = alpha * solve(p)."""
        m = dm.Model("compose_scale")
        m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(m._parameters[0] * x)

        solve_fn = _make_jax_differentiable_solve(m)

        def loss(p_flat):
            return 2.0 * solve_fn(p_flat)

        grad_fn = jax.grad(loss)
        p_flat = jnp.array([3.0])
        grad = grad_fn(p_flat)
        # obj* = p, dloss/dp = 2 * 1 = 2
        assert float(grad[0]) == pytest.approx(2.0, abs=0.5)
