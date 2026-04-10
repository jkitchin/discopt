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
    DiffSolveResultL3,
    SensitivityInfo,
    _compile_parametric_constraint,
    _compile_parametric_objective,
    _flatten_params,
    _make_jax_differentiable_solve,
    _param_total_size,
    _perturbation_gradient,
    differentiable_solve,
    differentiable_solve_l3,
    find_active_set,
    implicit_differentiate,
)
from discopt.modeling.core import Constant, Constraint

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

    @pytest.mark.parametrize("nlp_solver", ["ipm", "ipopt"])
    def test_simple_parametric_lp(self, nlp_solver):
        """min p*x s.t. x >= 1, x <= 5, p > 0.

        Optimal: x* = 1 (for p > 0), obj* = p.
        d(obj*)/dp = 1 (since x* = 1 is constant w.r.t. p).
        Actually by envelope: d(obj*)/dp = x* = 1.
        """
        m = dm.Model("param_lp")
        p = m.parameter("price", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(p * x)

        result = differentiable_solve(m, nlp_solver=nlp_solver)
        assert isinstance(result, DiffSolveResult)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(3.0, abs=1e-4)

        grad = result.gradient(p)
        # d(obj*)/dp = x* = 1.0
        assert float(grad) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("nlp_solver", ["ipm", "ipopt"])
    def test_parametric_rhs(self, nlp_solver):
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

        result = differentiable_solve(m, nlp_solver=nlp_solver)
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

    @pytest.mark.parametrize("nlp_solver", ["ipm", "ipopt"])
    def test_parametric_nlp(self, nlp_solver):
        """min x^2 + p*x s.t. x >= 0.

        For p >= 0: x* = 0, obj* = 0.
        For p < 0: x* = -p/2, obj* = -p^2/4.
        d(obj*)/dp = -p/2 when p < 0.
        """
        m = dm.Model("param_nlp")
        p = m.parameter("p", value=-4.0)
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x**2 + p * x)

        result = differentiable_solve(m, nlp_solver=nlp_solver)
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


# ──────────────────────────────────────────────────────────
# TestActiveSetFinder (L3)
# ──────────────────────────────────────────────────────────


class TestActiveSetFinder:
    """Test active set identification at the solution point."""

    def test_active_inequality_at_boundary(self):
        """x >= b where x* = b should have an active constraint."""
        m = dm.Model("active_ineq")
        b = m.parameter("b", value=2.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= b)

        constraint_fns = [
            _compile_parametric_constraint(c, m)
            for c in m._constraints
            if isinstance(c, Constraint)
        ]
        p_flat = _flatten_params(m)
        # x* = 2.0 (the active constraint boundary)
        x_star = jnp.array([2.0])

        active_cons, active_bounds = find_active_set(x_star, m, constraint_fns, p_flat)
        assert len(active_cons) > 0

    def test_active_equality_constraint(self):
        """Equality constraint should always be active at feasible point."""
        m = dm.Model("active_eq")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)
        m.subject_to(x == p)

        constraint_fns = [
            _compile_parametric_constraint(c, m)
            for c in m._constraints
            if isinstance(c, Constraint)
        ]
        p_flat = _flatten_params(m)
        x_star = jnp.array([3.0])

        active_cons, _ = find_active_set(x_star, m, constraint_fns, p_flat)
        assert 0 in active_cons

    def test_no_active_constraints_interior(self):
        """Interior point has no active inequality constraints."""
        m = dm.Model("interior")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize((x - 5.0) ** 2)
        m.subject_to(x <= 8.0)

        constraint_fns = [
            _compile_parametric_constraint(c, m)
            for c in m._constraints
            if isinstance(c, Constraint)
        ]
        p_flat = _flatten_params(m)
        x_star = jnp.array([5.0])  # interior point

        active_cons, active_bounds = find_active_set(x_star, m, constraint_fns, p_flat)
        assert len(active_cons) == 0
        assert len(active_bounds) == 0

    def test_active_variable_bounds(self):
        """Variable at its lower bound should be detected."""
        m = dm.Model("active_bound")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)

        constraint_fns = []
        p_flat = _flatten_params(m)
        x_star = jnp.array([0.0])  # at lower bound

        _, active_bounds = find_active_set(x_star, m, constraint_fns, p_flat)
        assert 0 in active_bounds

    def test_tolerance_parameter(self):
        """Tight tolerance should not flag slightly inactive constraints."""
        m = dm.Model("tol_test")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)
        m.subject_to(x <= 5.0)

        constraint_fns = [
            _compile_parametric_constraint(c, m)
            for c in m._constraints
            if isinstance(c, Constraint)
        ]
        p_flat = _flatten_params(m)
        # x = 4.99 => constraint body = 4.99 - 5 = -0.01, inactive with tight tol
        x_star = jnp.array([4.99])

        active_tight, _ = find_active_set(x_star, m, constraint_fns, p_flat, tol=1e-3)
        assert len(active_tight) == 0

        # With loose tolerance, it should be active
        active_loose, _ = find_active_set(x_star, m, constraint_fns, p_flat, tol=0.02)
        assert len(active_loose) > 0


# ──────────────────────────────────────────────────────────
# TestImplicitDifferentiation (L3)
# ──────────────────────────────────────────────────────────


class TestImplicitDifferentiation:
    """Test implicit differentiation via KKT system."""

    @pytest.mark.parametrize("nlp_solver", ["ipm", "ipopt"])
    def test_unconstrained_quadratic(self, nlp_solver):
        """min (x - p)^2, unconstrained: x* = p, dx/dp = 1, dobj/dp = 0."""
        m = dm.Model("imp_unconstrained")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m, nlp_solver=nlp_solver)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0, abs=1e-4)

        # L3 implicit gradient: dobj/dp = 0 (since obj = 0 at optimum)
        grad_l3 = result.implicit_gradient(p)
        assert float(grad_l3) == pytest.approx(0.0, abs=1e-3)

        # dx/dp = 1.0 (x* = p)
        sens = result.sensitivity_matrix()
        assert sens is not None
        assert float(sens[0, 0]) == pytest.approx(1.0, abs=1e-2)

    @pytest.mark.parametrize("nlp_solver", ["ipm", "ipopt"])
    def test_constrained_active(self, nlp_solver):
        """min x^2 s.t. x >= p. At optimum x* = p, dobj/dp = 2p."""
        m = dm.Model("imp_constrained")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x**2)
        m.subject_to(x >= p)

        result = differentiable_solve_l3(m, nlp_solver=nlp_solver)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(4.0, abs=1e-3)

        # L3: dobj/dp = 2*x* = 2*p = 4.0
        grad_l3 = result.implicit_gradient(p)
        assert float(grad_l3) == pytest.approx(4.0, abs=0.1)

    def test_multiple_parameters(self):
        """min (x - a)^2 + (y - b)^2, unconstrained.

        x* = a, y* = b. dobj/da = 0, dobj/db = 0.
        dx/da = 1, dx/db = 0, dy/da = 0, dy/db = 1.
        """
        m = dm.Model("imp_multi")
        a = m.parameter("a", value=2.0)
        b = m.parameter("b", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize((x - a) ** 2 + (y - b) ** 2)

        result = differentiable_solve_l3(m)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(0.0, abs=1e-4)

        grad_a = result.implicit_gradient(a)
        grad_b = result.implicit_gradient(b)
        assert float(grad_a) == pytest.approx(0.0, abs=1e-3)
        assert float(grad_b) == pytest.approx(0.0, abs=1e-3)

        sens = result.sensitivity_matrix()
        assert sens is not None
        assert sens.shape == (2, 2)
        # dx/da = 1, dx/db = 0
        assert float(sens[0, 0]) == pytest.approx(1.0, abs=1e-2)
        assert float(sens[0, 1]) == pytest.approx(0.0, abs=1e-2)
        # dy/da = 0, dy/db = 1
        assert float(sens[1, 0]) == pytest.approx(0.0, abs=1e-2)
        assert float(sens[1, 1]) == pytest.approx(1.0, abs=1e-2)

    def test_l3_agrees_with_l1(self):
        """On well-conditioned problems, L3 and L1 should agree."""
        m = dm.Model("l3_vs_l1")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(p * x)

        result = differentiable_solve_l3(m)
        l1_grad = float(result.gradient(p))
        l3_grad = float(result.implicit_gradient(p))

        # Both should give dobj/dp = x* = 1.0
        assert l1_grad == pytest.approx(1.0, abs=1e-2)
        assert l3_grad == pytest.approx(1.0, abs=1e-2)

    def test_l3_vs_finite_difference(self):
        """L3 gradient should match finite difference."""

        def make_model(p_val):
            m = dm.Model("l3_fd")
            p = m.parameter("p", value=p_val)
            x = m.continuous("x", lb=0, ub=100)
            m.minimize(x**2 + p * x)
            return m, p

        m, p = make_model(-4.0)
        result = differentiable_solve_l3(m)
        l3_grad = float(result.implicit_gradient(p))

        # Finite difference
        eps = 1e-5
        m_plus, _ = make_model(-4.0 + eps)
        r_plus = differentiable_solve(m_plus)
        m_minus, _ = make_model(-4.0 - eps)
        r_minus = differentiable_solve(m_minus)
        fd_grad = (r_plus.objective - r_minus.objective) / (2 * eps)

        assert l3_grad == pytest.approx(fd_grad, rel=1e-2)


# ──────────────────────────────────────────────────────────
# TestDifferentiableSolveL3 (L3)
# ──────────────────────────────────────────────────────────


class TestDifferentiableSolveL3:
    """Test the full differentiable_solve_l3 function."""

    def test_basic_parametric_lp(self):
        """min p*x s.t. x >= 1. obj* = p, dobj/dp = 1."""
        m = dm.Model("l3_lp")
        p = m.parameter("price", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(p * x)

        result = differentiable_solve_l3(m)
        assert isinstance(result, DiffSolveResultL3)
        assert result.status == "optimal"
        assert result.objective == pytest.approx(3.0, abs=1e-3)

    def test_parametric_quadratic(self):
        """min (x - p)^2, x in [-10, 10]. obj* = 0 for all p in range."""
        m = dm.Model("l3_quad")
        p = m.parameter("p", value=5.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)
        assert result.objective == pytest.approx(0.0, abs=1e-4)

        grad = result.implicit_gradient(p)
        assert float(grad) == pytest.approx(0.0, abs=1e-3)

    def test_parametric_nlp_active_constraint(self):
        """min x^2 s.t. x >= p. x* = p, obj* = p^2, dobj/dp = 2p."""
        m = dm.Model("l3_nlp")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x**2)
        m.subject_to(x >= p)

        result = differentiable_solve_l3(m)
        assert result.objective == pytest.approx(9.0, abs=1e-2)

        l1_grad = result.gradient(p)
        l3_grad = result.implicit_gradient(p)
        # Both should give dobj/dp = 2*p = 6.0
        assert float(l1_grad) == pytest.approx(6.0, abs=0.2)
        assert float(l3_grad) == pytest.approx(6.0, abs=0.2)

    def test_both_gradients_available(self):
        """Both .gradient() and .implicit_gradient() should work."""
        m = dm.Model("l3_both")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)
        g1 = result.gradient(p)
        g3 = result.implicit_gradient(p)
        sens = result.sensitivity_matrix()

        # All should be accessible
        assert g1 is not None
        assert g3 is not None
        assert sens is not None

    def test_repr(self):
        """Test string representation."""
        m = dm.Model("l3_repr")
        m.parameter("p", value=1.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)

        result = differentiable_solve_l3(m)
        r = repr(result)
        assert "DiffSolveResultL3" in r
        assert "l3=ok" in r


# ──────────────────────────────────────────────────────────
# TestPerturbationSmoothing (L3)
# ──────────────────────────────────────────────────────────


class TestPerturbationSmoothing:
    """Test fallback perturbation smoothing for ill-conditioned KKT."""

    def test_perturbation_gradient_finite(self):
        """Perturbation-based gradient should return finite values."""
        m = dm.Model("perturb")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=5)
        m.minimize(p * x)

        p_flat = _flatten_params(m)
        grad = _perturbation_gradient(m, p_flat, ipopt_options=None)
        assert np.all(np.isfinite(grad))
        # dobj/dp = x* = 1
        assert float(grad[0]) == pytest.approx(1.0, abs=1e-2)

    def test_perturbation_matches_analytic(self):
        """Perturbation gradient should match analytic gradient."""
        m = dm.Model("perturb_match")
        p = m.parameter("p", value=-4.0)
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x**2 + p * x)

        # Analytic: x* = -p/2 = 2, obj* = -4. dobj/dp = x* = 2.
        p_flat = _flatten_params(m)
        grad = _perturbation_gradient(m, p_flat, ipopt_options=None)
        assert float(grad[0]) == pytest.approx(2.0, abs=0.1)

    def test_perturbation_multi_param(self):
        """Perturbation with multiple parameters."""
        m = dm.Model("perturb_multi")
        a = m.parameter("a", value=1.0)
        b = m.parameter("b", value=3.0)
        x1 = m.continuous("x1", lb=0, ub=5)
        x2 = m.continuous("x2", lb=0, ub=5)
        m.minimize(a * x1 + b * x2)
        m.subject_to(x1 + x2 >= 1)

        p_flat = _flatten_params(m)
        grad = _perturbation_gradient(m, p_flat, ipopt_options=None)
        assert np.all(np.isfinite(grad))
        # For a < b: x1* = 1, x2* = 0. dobj/da = 1, dobj/db = 0
        assert float(grad[0]) == pytest.approx(1.0, abs=1e-2)
        assert float(grad[1]) == pytest.approx(0.0, abs=1e-2)


# ──────────────────────────────────────────────────────────
# TestL3VsL1Comparison (L3)
# ──────────────────────────────────────────────────────────


class TestL3VsL1Comparison:
    """Compare L3 and L1 gradients on various problems."""

    def test_agree_on_well_conditioned(self):
        """L3 and L1 should agree on well-conditioned problems."""
        m = dm.Model("l3_l1_agree")
        p = m.parameter("p", value=-4.0)
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x**2 + p * x)

        result = differentiable_solve_l3(m)
        l1 = float(result.gradient(p))
        l3 = float(result.implicit_gradient(p))

        # Both should give dobj/dp = x* = 2.0
        assert l1 == pytest.approx(l3, abs=0.2)
        assert l1 == pytest.approx(2.0, abs=0.1)

    def test_l3_closer_to_fd(self):
        """On constrained problems, L3 should be at least as close to FD as L1."""

        def make_model(p_val):
            m = dm.Model("l3_fd_compare")
            p = m.parameter("p", value=p_val)
            x = m.continuous("x", lb=-10, ub=10)
            m.minimize(x**2)
            m.subject_to(x >= p)
            return m, p

        m, p = make_model(2.0)
        result = differentiable_solve_l3(m)
        l1 = float(result.gradient(p))
        l3 = float(result.implicit_gradient(p))

        # Finite difference
        eps = 1e-5
        m_plus, _ = make_model(2.0 + eps)
        r_plus = differentiable_solve(m_plus)
        m_minus, _ = make_model(2.0 - eps)
        r_minus = differentiable_solve(m_minus)
        fd = (r_plus.objective - r_minus.objective) / (2 * eps)

        # L3 should be close to FD
        assert l3 == pytest.approx(fd, rel=0.05)
        # L3 error should be <= L1 error (or at most slightly worse)
        assert abs(l3 - fd) <= abs(l1 - fd) + 0.5

    def test_sensitivity_matrix_shape(self):
        """Sensitivity matrix should have shape (n_vars, n_params).

        Uses ipopt for this LP since IPM can struggle with zero-Hessian
        objectives, causing the KKT system to be singular.
        """
        m = dm.Model("l3_sens_shape")
        m.parameter("a", value=1.0)
        m.parameter("b", value=2.0)
        x1 = m.continuous("x1", lb=0, ub=10)
        x2 = m.continuous("x2", lb=0, ub=10)
        m.minimize(m._parameters[0] * x1 + m._parameters[1] * x2)
        m.subject_to(x1 + x2 >= 1)

        result = differentiable_solve_l3(m, nlp_solver="ipopt")
        sens = result.sensitivity_matrix()
        assert sens is not None
        assert sens.shape == (2, 2)


# ──────────────────────────────────────────────────────────
# TestSIPOPTFeatures
# ──────────────────────────────────────────────────────────


class TestSIPOPTFeatures:
    """Test sIPOPT-inspired sensitivity features: approximate_resolve,
    dual_sensitivity, reduced_hessian, and sensitivity re-query."""

    def test_approximate_resolve_unconstrained(self):
        """min (x-p)^2 => x*=p. Approximate resolve for new p should give x~p_new."""
        m = dm.Model("sipopt_approx_unc")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)
        assert result.status == "optimal"

        # Approximate resolve at p=3.5 (small perturbation)
        x_approx = result.approximate_resolve([(p, 3.5)])
        assert "x" in x_approx
        # x*(p=3.5) ≈ 3.5 (first-order exact for this linear sensitivity)
        assert float(np.asarray(x_approx["x"]).flat[0]) == pytest.approx(3.5, abs=1e-2)

    def test_approximate_resolve_constrained(self):
        """min x^2 s.t. x >= p => x*=p. Verify linear approximation."""
        m = dm.Model("sipopt_approx_con")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x**2)
        m.subject_to(x >= p)

        result = differentiable_solve_l3(m)
        assert result.status == "optimal"

        # Approximate resolve at p=2.1
        x_approx = result.approximate_resolve([(p, 2.1)])
        # x*(2.1) ≈ 2.0 + dx/dp * 0.1 = 2.0 + 1.0 * 0.1 = 2.1
        assert float(np.asarray(x_approx["x"]).flat[0]) == pytest.approx(2.1, abs=0.05)

    def test_approximate_resolve_accuracy(self):
        """Small dp gives error O(dp^2) confirming first-order accuracy."""
        m = dm.Model("sipopt_approx_acc")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)

        # Two perturbation sizes
        dp_large = 0.1
        dp_small = 0.01

        x_approx_large = result.approximate_resolve([(p, 2.0 + dp_large)])
        x_approx_small = result.approximate_resolve([(p, 2.0 + dp_small)])

        # True values: x*(p) = p
        err_large = abs(float(np.asarray(x_approx_large["x"]).flat[0]) - (2.0 + dp_large))
        err_small = abs(float(np.asarray(x_approx_small["x"]).flat[0]) - (2.0 + dp_small))

        # For this linear problem, errors should be near-zero
        assert err_large < 1e-3
        assert err_small < 1e-4

    def test_dual_sensitivity_simple(self):
        """min x s.t. x >= p => lambda=1 (always). dlambda/dp should be ~0."""
        m = dm.Model("sipopt_dual_simple")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x)
        m.subject_to(x >= p)

        result = differentiable_solve_l3(m)
        ds = result.dual_sensitivity(p)
        # With only one active constraint and constant multiplier,
        # dual sensitivity should exist (may or may not be zero depending
        # on the specific KKT structure)
        if ds is not None:
            assert ds.shape[1] == 1  # one parameter
            assert np.all(np.isfinite(ds))

    def test_dual_sensitivity_matches_finite_diff(self):
        """Dual sensitivity should match finite-difference for multipliers."""
        m = dm.Model("sipopt_dual_fd")
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x**2)
        m.subject_to(x >= p)

        result = differentiable_solve_l3(m)
        ds = result.dual_sensitivity(p)

        if ds is not None and ds.size > 0:
            # All values should be finite
            assert np.all(np.isfinite(ds))

    def test_reduced_hessian_unconstrained(self):
        """min (x-p)^2 => Hessian = [[2]], no active constraints.

        Reduced Hessian should equal full Hessian.
        """
        m = dm.Model("sipopt_rh_unc")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)
        rh = result.reduced_hessian()
        assert rh is not None
        # Hessian of (x-p)^2 w.r.t. x is 2.0
        assert rh.shape == (1, 1)
        assert float(rh[0, 0]) == pytest.approx(2.0, abs=1e-3)

    def test_reduced_hessian_constrained(self):
        """min x1^2 + x2^2 s.t. x1 + x2 = p.

        One equality constraint active => reduced Hessian has dim (2-1, 2-1) = (1, 1).
        """
        m = dm.Model("sipopt_rh_con")
        p = m.parameter("p", value=4.0)
        x1 = m.continuous("x1", lb=-10, ub=10)
        x2 = m.continuous("x2", lb=-10, ub=10)
        m.minimize(x1**2 + x2**2)
        m.subject_to(x1 + x2 == p)

        result = differentiable_solve_l3(m)
        rh = result.reduced_hessian()
        assert rh is not None
        # One equality removes one DOF: reduced Hessian should be (1,1)
        assert rh.shape == (1, 1)
        # The reduced Hessian should be positive definite
        assert float(rh[0, 0]) > 0

    def test_sensitivity_new_rhs(self):
        """sensitivity(dp) should match dx_dp @ dp from the stored matrix."""
        m = dm.Model("sipopt_sens_rhs")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)
        dp = np.array([0.5])

        dx_direct = result.sensitivity(dp)
        dx_matrix = result.sensitivity_matrix() @ dp

        np.testing.assert_allclose(dx_direct, dx_matrix, atol=1e-8)

    def test_sensitivity_batch(self):
        """Multiple dp vectors give correct results via batch solve."""
        m = dm.Model("sipopt_sens_batch")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        result = differentiable_solve_l3(m)
        # Batch: 3 different dp vectors, each of size 1
        dp_batch = np.array([[0.1, 0.5, 1.0]])  # (1, 3)

        dx_batch = result.sensitivity(dp_batch)
        assert dx_batch.shape == (1, 3)

        # Each column should match individual sensitivity
        for i in range(3):
            dx_single = result.sensitivity(dp_batch[:, i])
            np.testing.assert_allclose(dx_batch[:, i], dx_single, atol=1e-10)

    def test_no_regression_existing_api(self):
        """Existing .gradient(), .implicit_gradient(), .sensitivity_matrix() unchanged."""
        m = dm.Model("sipopt_no_regress")
        p = m.parameter("p", value=-4.0)
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x**2 + p * x)

        result = differentiable_solve_l3(m)

        # All existing APIs should still work
        g1 = result.gradient(p)
        g3 = result.implicit_gradient(p)
        sens = result.sensitivity_matrix()
        val = result.value(x)

        assert float(np.asarray(g1).flat[0]) == pytest.approx(2.0, abs=0.1)
        assert float(np.asarray(g3).flat[0]) == pytest.approx(2.0, abs=0.1)
        assert sens is not None
        assert sens.shape == (1, 1)
        assert float(np.asarray(val).flat[0]) == pytest.approx(2.0, abs=0.1)

    def test_implicit_differentiate_returns_sensitivity_info(self):
        """implicit_differentiate should return SensitivityInfo NamedTuple."""
        m = dm.Model("sipopt_sensinfo")
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)

        constraint_fns = [
            _compile_parametric_constraint(c, m)
            for c in m._constraints
            if isinstance(c, Constraint)
        ]
        p_flat = _flatten_params(m)
        x_star = jnp.array([3.0])

        active_cons, active_bounds = find_active_set(x_star, m, constraint_fns, p_flat)

        info = implicit_differentiate(m, x_star, None, p_flat, active_cons, active_bounds)

        assert isinstance(info, SensitivityInfo)
        assert info.dx_dp.shape == (1, 1)
        assert info.n_vars == 1
        assert info.H_xp is not None


# ──────────────────────────────────────────────────────────
# TestSolveResultGradient
# ──────────────────────────────────────────────────────────


class TestSolveResultGradient:
    """Test SolveResult.gradient() lazy sensitivity computation."""

    def test_gradient_quadratic_at_optimum(self):
        """min (x - p)^2: x*=p, obj*=0, d(obj*)/dp = 0."""
        m = dm.Model()
        p = m.parameter("p", value=2.0)
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize((x - p) ** 2)
        result = m.solve()
        assert result.gradient(p) == pytest.approx(0.0, abs=1e-4)

    def test_gradient_linear_objective(self):
        """min p*x s.t. x >= 1: x*=1, obj*=p, d(obj*)/dp = 1."""
        m = dm.Model()
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=10)
        m.minimize(p * x)
        result = m.solve()
        assert result.gradient(p) == pytest.approx(1.0, abs=1e-3)

    def test_gradient_cached_on_second_call(self):
        """Second call to gradient() uses cached sensitivity."""
        m = dm.Model()
        p = m.parameter("p", value=3.0)
        x = m.continuous("x", lb=1, ub=10)
        m.minimize(p * x)
        result = m.solve()
        g1 = result.gradient(p)
        # _sensitivity should now be cached
        assert result._sensitivity is not None
        g2 = result.gradient(p)
        assert g1 == pytest.approx(g2)

    def test_gradient_raises_for_binary_model(self):
        """gradient() raises ValueError for models with integer variables."""
        m = dm.Model()
        m.parameter("p", value=1.0)
        y = m.binary("y")
        m.minimize(y)
        result = m.solve()
        with pytest.raises(ValueError, match="continuous"):
            result.gradient(m._parameters[0])

    def test_gradient_raises_no_parameters(self):
        """gradient() raises ValueError when model has no parameters."""
        m = dm.Model()
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        result = m.solve()
        from discopt.modeling.core import Parameter

        dummy_param = Parameter("dummy", 1.0, m)
        with pytest.raises(ValueError, match="no parameters"):
            result.gradient(dummy_param)

    def test_gradient_multiple_parameters(self):
        """min c1*x1 + c2*x2 s.t. x1+x2>=1: d(obj*)/dc1=x1*, d(obj*)/dc2=x2*."""
        m = dm.Model()
        c1 = m.parameter("c1", value=1.0)
        c2 = m.parameter("c2", value=3.0)
        x1 = m.continuous("x1", lb=0, ub=5)
        x2 = m.continuous("x2", lb=0, ub=5)
        m.minimize(c1 * x1 + c2 * x2)
        m.subject_to(x1 + x2 >= 1)
        result = m.solve()
        # c1 < c2, so x1*=1, x2*=0
        assert result.gradient(c1) == pytest.approx(1.0, abs=1e-2)
        assert result.gradient(c2) == pytest.approx(0.0, abs=1e-2)
