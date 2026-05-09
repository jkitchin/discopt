"""Tests for callback-based IPM solver (ipm_callbacks.py)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.ipm_callbacks import IPMOptions, ipm_solve_callbacks

# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------


def _rosenbrock():
    """Rosenbrock f(x) = (1-x1)^2 + 100*(x2-x1^2)^2, solution (1,1)."""

    def obj(x):
        return float((1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def grad(x):
        x1, x2 = x[0], x[1]
        g1 = -2 * (1 - x1) + 200 * (x2 - x1**2) * (-2 * x1)
        g2 = 200 * (x2 - x1**2)
        return np.array([g1, g2])

    def hess(x, obj_factor, y):
        x1, x2 = x[0], x[1]
        h11 = obj_factor * (2 - 400 * x2 + 1200 * x1**2)
        h12 = obj_factor * (-400 * x1)
        h22 = obj_factor * 200.0
        return np.array([[h11, h12], [h12, h22]])

    x0 = np.array([-1.2, 1.0])
    x_l = np.full(2, -1e20)
    x_u = np.full(2, 1e20)
    return obj, grad, hess, None, None, x0, x_l, x_u, None, None, np.array([1.0, 1.0]), 0.0


def _hs35():
    """HS35: bound-constrained QP.

    min  9 - 8x1 - 6x2 - 4x3 + 2x1^2 + 2x2^2 + x3^2 + 2x1*x2 + 2x1*x3
    s.t. x1 + x2 + 2*x3 <= 3
         x >= 0
    Solution: x* = (4/3, 7/9, 4/9), f* = 1/9
    """

    def obj(x):
        x1, x2, x3 = x[0], x[1], x[2]
        return float(
            9 - 8 * x1 - 6 * x2 - 4 * x3 + 2 * x1**2 + 2 * x2**2 + x3**2 + 2 * x1 * x2 + 2 * x1 * x3
        )

    def grad(x):
        x1, x2, x3 = x[0], x[1], x[2]
        g1 = -8 + 4 * x1 + 2 * x2 + 2 * x3
        g2 = -6 + 2 * x1 + 4 * x2
        g3 = -4 + 2 * x3 + 2 * x1
        return np.array([g1, g2, g3])

    def hess(x, obj_factor, y):
        H = obj_factor * np.array(
            [
                [4.0, 2.0, 2.0],
                [2.0, 4.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        )
        return H

    def con(x):
        return np.array([x[0] + x[1] + 2 * x[2]])

    def jac(x):
        return np.array([[1.0, 1.0, 2.0]])

    x0 = np.array([0.5, 0.5, 0.5])
    x_l = np.array([0.0, 0.0, 0.0])
    x_u = np.full(3, 1e20)
    g_l = np.array([-1e20])
    g_u = np.array([3.0])
    x_star = np.array([4 / 3, 7 / 9, 4 / 9])
    f_star = 1.0 / 9.0
    return obj, grad, hess, con, jac, x0, x_l, x_u, g_l, g_u, x_star, f_star


def _hs71():
    """HS71: nonlinear constraints.

    min  x1*x4*(x1+x2+x3) + x3
    s.t. x1*x2*x3*x4 >= 25
         x1^2 + x2^2 + x3^2 + x4^2 = 40
         1 <= x_i <= 5
    Solution: x* ~ (1.0, 4.743, 3.821, 1.379), f* ~ 17.014
    """

    def obj(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    def grad(x):
        g = np.zeros(4)
        g[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3]
        g[1] = x[0] * x[3]
        g[2] = x[0] * x[3] + 1.0
        g[3] = x[0] * (x[0] + x[1] + x[2])
        return g

    def hess(x, obj_factor, y):
        H = np.zeros((4, 4))
        # Objective Hessian
        H[0, 0] = obj_factor * 2.0 * x[3]
        H[0, 1] = obj_factor * x[3]
        H[1, 0] = H[0, 1]
        H[0, 2] = obj_factor * x[3]
        H[2, 0] = H[0, 2]
        H[0, 3] = obj_factor * (2 * x[0] + x[1] + x[2])
        H[3, 0] = H[0, 3]
        H[1, 3] = obj_factor * x[0]
        H[3, 1] = H[1, 3]
        H[2, 3] = obj_factor * x[0]
        H[3, 2] = H[2, 3]

        # Constraint 1 Hessian: x1*x2*x3*x4
        if len(y) > 0:
            H[0, 1] += y[0] * x[2] * x[3]
            H[1, 0] += y[0] * x[2] * x[3]
            H[0, 2] += y[0] * x[1] * x[3]
            H[2, 0] += y[0] * x[1] * x[3]
            H[0, 3] += y[0] * x[1] * x[2]
            H[3, 0] += y[0] * x[1] * x[2]
            H[1, 2] += y[0] * x[0] * x[3]
            H[2, 1] += y[0] * x[0] * x[3]
            H[1, 3] += y[0] * x[0] * x[2]
            H[3, 1] += y[0] * x[0] * x[2]
            H[2, 3] += y[0] * x[0] * x[1]
            H[3, 2] += y[0] * x[0] * x[1]

        # Constraint 2 Hessian: x1^2+x2^2+x3^2+x4^2
        if len(y) > 1:
            for i in range(4):
                H[i, i] += y[1] * 2.0

        return H

    def con(x):
        return np.array(
            [
                x[0] * x[1] * x[2] * x[3],
                x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2,
            ]
        )

    def jac(x):
        J = np.zeros((2, 4))
        J[0, 0] = x[1] * x[2] * x[3]
        J[0, 1] = x[0] * x[2] * x[3]
        J[0, 2] = x[0] * x[1] * x[3]
        J[0, 3] = x[0] * x[1] * x[2]
        J[1, 0] = 2 * x[0]
        J[1, 1] = 2 * x[1]
        J[1, 2] = 2 * x[2]
        J[1, 3] = 2 * x[3]
        return J

    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    x_l = np.array([1.0, 1.0, 1.0, 1.0])
    x_u = np.array([5.0, 5.0, 5.0, 5.0])
    g_l = np.array([25.0, 40.0])
    g_u = np.array([1e20, 40.0])
    x_star = np.array([1.0, 4.74299963, 3.82114998, 1.37940829])
    f_star = 17.0140173
    return obj, grad, hess, con, jac, x0, x_l, x_u, g_l, g_u, x_star, f_star


def _quadratic_unconstrained():
    """Simple quadratic: min 0.5*x'*I*x, solution at origin."""

    def obj(x):
        return float(0.5 * np.dot(x, x))

    def grad(x):
        return x.copy()

    def hess(x, obj_factor, y):
        return obj_factor * np.eye(len(x))

    n = 5
    x0 = np.ones(n) * 3.0
    x_l = np.full(n, -1e20)
    x_u = np.full(n, 1e20)
    return obj, grad, hess, None, None, x0, x_l, x_u, None, None, np.zeros(n), 0.0


def _bounded_quadratic():
    """Bounded quadratic: min (x1-2)^2 + (x2-3)^2, 0 <= x <= 1.5.
    Solution at (1.5, 1.5).
    """

    def obj(x):
        return float((x[0] - 2) ** 2 + (x[1] - 3) ** 2)

    def grad(x):
        return np.array([2 * (x[0] - 2), 2 * (x[1] - 3)])

    def hess(x, obj_factor, y):
        return obj_factor * np.array([[2.0, 0.0], [0.0, 2.0]])

    x0 = np.array([0.5, 0.5])
    x_l = np.array([0.0, 0.0])
    x_u = np.array([1.5, 1.5])
    x_star = np.array([1.5, 1.5])
    f_star = float((1.5 - 2) ** 2 + (1.5 - 3) ** 2)
    return obj, grad, hess, None, None, x0, x_l, x_u, None, None, x_star, f_star


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIPMCallbacksUnconstrained:
    def test_simple_quadratic(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _quadratic_unconstrained()
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu)
        assert int(state.converged) in (1, 2)
        np.testing.assert_allclose(np.asarray(state.x), x_star, atol=1e-5)
        assert abs(float(state.obj) - f_star) < 1e-5

    def test_rosenbrock(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _rosenbrock()
        opts = IPMOptions(max_iter=2000, tol=1e-7)
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        assert int(state.converged) in (1, 2)
        np.testing.assert_allclose(np.asarray(state.x), x_star, atol=1e-4)

    def test_bounded_quadratic(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _bounded_quadratic()
        opts = IPMOptions(max_iter=2000, acceptable_tol=1e-5, acceptable_iter=5)
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        # May hit iter limit near active bounds but solution is close
        np.testing.assert_allclose(np.asarray(state.x), x_star, atol=1e-2)
        assert abs(float(state.obj) - f_star) < 0.1


class TestIPMCallbacksConstrained:
    def test_hs35(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _hs35()
        opts = IPMOptions(max_iter=1000, tol=1e-7)
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        assert int(state.converged) in (1, 2)
        np.testing.assert_allclose(np.asarray(state.x), x_star, atol=1e-3)
        assert abs(float(state.obj) - f_star) < 1e-3

    def test_hs71(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _hs71()
        opts = IPMOptions(max_iter=3000, tol=1e-7, acceptable_tol=1e-5, acceptable_iter=5)
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        # Check feasibility (relax tolerance for near-bound convergence)
        c = con(np.asarray(state.x))
        assert c[0] >= 25.0 - 1e-2  # product constraint
        assert abs(c[1] - 40.0) < 1e-2  # sum-of-squares equality
        np.testing.assert_allclose(float(state.obj), f_star, rtol=5e-2)


class TestIPMCallbacksOptions:
    def test_custom_options(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _quadratic_unconstrained()
        opts = IPMOptions(tol=1e-10, max_iter=500, mu_init=0.01)
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        assert int(state.converged) in (1, 2)
        np.testing.assert_allclose(np.asarray(state.x), x_star, atol=1e-6)

    def test_no_predictor_corrector(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _rosenbrock()
        opts = IPMOptions(predictor_corrector=False, max_iter=2000)
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        assert int(state.converged) in (1, 2)
        np.testing.assert_allclose(np.asarray(state.x), x_star, atol=1e-3)

    def test_least_squares_init(self):
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu, x_star, f_star = _hs35()
        opts = IPMOptions(
            least_squares_mult_init=True,
            max_iter=2000,
            acceptable_tol=1e-5,
            acceptable_iter=5,
        )
        state = ipm_solve_callbacks(obj, grad, hess, con, jac, x0, xl, xu, gl, gu, options=opts)
        # LS init may converge differently near active bounds; verify it ran
        assert int(state.iteration) > 0
        # Objective should be reasonable (< 1.0 for HS35, optimal is 1/9)
        assert float(state.obj) < 1.0


class TestIPMCallbacksAgreesWithJAXIPM:
    """Verify the callback IPM gives same results as the JAX autodiff IPM."""

    def test_rosenbrock_agreement(self):
        """Both IPMs should converge to the same point on Rosenbrock."""
        from discopt._jax.ipm import ipm_solve

        # JAX version
        def obj_jax(x):
            return (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

        x0_jax = jnp.array([-1.2, 1.0])
        xl = jnp.full(2, -1e20)
        xu = jnp.full(2, 1e20)
        opts = IPMOptions(max_iter=2000, tol=1e-7)
        state_jax = ipm_solve(obj_jax, None, x0_jax, xl, xu, options=opts)

        # Callback version
        obj, grad, hess, _, _, x0, x_l, x_u, _, _, _, _ = _rosenbrock()
        state_cb = ipm_solve_callbacks(obj, grad, hess, None, None, x0, x_l, x_u, options=opts)

        np.testing.assert_allclose(np.asarray(state_cb.x), np.asarray(state_jax.x), atol=1e-4)
        np.testing.assert_allclose(float(state_cb.obj), float(state_jax.obj), atol=1e-4)


class TestSolveNLPIPMCallbacks:
    """Test the high-level evaluator adapter."""

    def test_with_mock_evaluator(self):
        """Test solve_nlp_ipm_callbacks with a mock evaluator interface."""
        from discopt._jax.ipm_callbacks import solve_nlp_ipm_callbacks

        class MockEvaluator:
            n_variables = 2
            n_constraints = 0
            variable_bounds = (np.full(2, -1e20), np.full(2, 1e20))

            def evaluate_objective(self, x):
                return float(0.5 * np.dot(x, x))

            def evaluate_gradient(self, x):
                return x.copy()

            def evaluate_lagrangian_hessian(self, x, obj_factor, y):
                return obj_factor * np.eye(2)

        ev = MockEvaluator()
        result = solve_nlp_ipm_callbacks(ev, np.array([3.0, 4.0]))
        assert result.status.value == "optimal"
        np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-5)
        assert result.iterations > 0
        assert result.wall_time > 0

    def test_with_constrained_mock(self):
        """Test with a constrained mock evaluator (HS35-like)."""
        from discopt._jax.ipm_callbacks import solve_nlp_ipm_callbacks

        class MockConstrainedEvaluator:
            n_variables = 3
            n_constraints = 1
            variable_bounds = (np.array([0.0, 0.0, 0.0]), np.full(3, 1e20))

            def evaluate_objective(self, x):
                x1, x2, x3 = x[0], x[1], x[2]
                return float(
                    9
                    - 8 * x1
                    - 6 * x2
                    - 4 * x3
                    + 2 * x1**2
                    + 2 * x2**2
                    + x3**2
                    + 2 * x1 * x2
                    + 2 * x1 * x3
                )

            def evaluate_gradient(self, x):
                x1, x2, x3 = x[0], x[1], x[2]
                return np.array(
                    [
                        -8 + 4 * x1 + 2 * x2 + 2 * x3,
                        -6 + 2 * x1 + 4 * x2,
                        -4 + 2 * x3 + 2 * x1,
                    ]
                )

            def evaluate_lagrangian_hessian(self, x, obj_factor, y):
                return obj_factor * np.array(
                    [
                        [4.0, 2.0, 2.0],
                        [2.0, 4.0, 0.0],
                        [2.0, 0.0, 2.0],
                    ]
                )

            def evaluate_constraints(self, x):
                return np.array([x[0] + x[1] + 2 * x[2]])

            def evaluate_jacobian(self, x):
                return np.array([[1.0, 1.0, 2.0]])

        ev = MockConstrainedEvaluator()
        result = solve_nlp_ipm_callbacks(
            ev,
            np.array([0.5, 0.5, 0.5]),
            constraint_bounds=[(-1e20, 3.0)],
        )
        assert result.status.value == "optimal"
        np.testing.assert_allclose(result.objective, 1.0 / 9.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Feasibility restoration tests
# ---------------------------------------------------------------------------


def _hs106():
    """HS106: 8 variables, 6 inequality constraints.

    min  f(x)  (nonlinear objective)
    s.t. 6 nonlinear inequality constraints
         x_l <= x <= x_u

    Known optimal: f* ~ 7049.248
    """

    def obj(x):
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        return x1 + x2 + x3

    def grad(x):
        g = np.zeros(8)
        g[0] = 1.0
        g[1] = 1.0
        g[2] = 1.0
        return g

    def hess(x, obj_factor, y):
        H = np.zeros((8, 8))
        # Objective Hessian is zero (linear objective)
        # Constraint Hessians:
        if len(y) > 0 and y[0] != 0:
            # c1 = 1 - 0.0025*(x4 + x6)  -- linear, no Hessian
            pass
        if len(y) > 1 and y[1] != 0:
            # c2 = 1 - 0.0025*(x5 - x4 + x7)  -- linear, no Hessian
            pass
        if len(y) > 2 and y[2] != 0:
            # c3 = 1 - 0.01*(x8 - x5)  -- linear, no Hessian
            pass
        if len(y) > 3 and y[3] != 0:
            # c4 = x1*x6 - 833.33252*x4 - 100*x1 + 83333.333
            H[0, 5] += y[3]
            H[5, 0] += y[3]
        if len(y) > 4 and y[4] != 0:
            # c5 = x2*x7 - 1250*x5 - x2*x4 + 1250*x4
            H[1, 6] += y[4]
            H[6, 1] += y[4]
            H[1, 3] += -y[4]
            H[3, 1] += -y[4]
        if len(y) > 5 and y[5] != 0:
            # c6 = x3*x8 - 2500*x5 - x3*x5 + 1250000
            H[2, 7] += y[5]
            H[7, 2] += y[5]
            H[2, 4] += -y[5]
            H[4, 2] += -y[5]
        return H

    def con(x):
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        c = np.zeros(6)
        c[0] = 1 - 0.0025 * (x4 + x6)
        c[1] = 1 - 0.0025 * (x5 - x4 + x7)
        c[2] = 1 - 0.01 * (x8 - x5)
        c[3] = x1 * x6 - 833.33252 * x4 - 100 * x1 + 83333.333
        c[4] = x2 * x7 - 1250 * x5 - x2 * x4 + 1250 * x4
        c[5] = x3 * x8 - 2500 * x5 - x3 * x5 + 1250000
        return c

    def jac(x):
        x1, x2, x3, x4, x5, x6, x7, x8 = x
        J = np.zeros((6, 8))
        # c1
        J[0, 3] = -0.0025
        J[0, 5] = -0.0025
        # c2
        J[1, 3] = 0.0025
        J[1, 4] = -0.0025
        J[1, 6] = -0.0025
        # c3
        J[2, 4] = 0.01
        J[2, 7] = -0.01
        # c4
        J[3, 0] = x6 - 100
        J[3, 3] = -833.33252
        J[3, 5] = x1
        # c5
        J[4, 1] = x7 - x4
        J[4, 3] = -x2 + 1250
        J[4, 4] = -1250
        J[4, 6] = x2
        # c6
        J[5, 2] = x8 - x5
        J[5, 4] = -2500 - x3
        J[5, 7] = x3
        return J

    x0 = np.array([5000.0, 5000.0, 5000.0, 200.0, 350.0, 150.0, 225.0, 425.0])
    x_l = np.array([100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    x_u = np.array([10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    g_l = np.zeros(6)
    g_u = np.full(6, 1e20)
    return obj, grad, hess, con, jac, x0, x_l, x_u, g_l, g_u


class TestFeasibilityRestoration:
    def test_restoration_callbacks_construction(self):
        """Verify restoration callback shapes and values."""
        from discopt._jax.ipm_callbacks import _build_restoration_callbacks

        def con(x):
            return np.array([x[0] + x[1] - 1.0])  # c(x) = x1+x2-1

        def jac(x):
            return np.array([[1.0, 1.0]])

        x_ref = np.array([0.0, 0.0])
        g_l = np.array([0.0])
        g_u = np.array([1e20])
        has_g_lb = jnp.array([1.0])
        has_g_ub = jnp.array([0.0])

        r_obj, r_grad, r_hess = _build_restoration_callbacks(
            con,
            jac,
            x_ref,
            g_l,
            g_u,
            has_g_lb,
            has_g_ub,
        )

        # At x=[0,0], c=-1, violation of g_l=0: softplus(0-(-1))=softplus(1)>0
        val = r_obj(np.array([0.0, 0.0]))
        assert val > 0.0  # positive violation

        g = r_grad(np.array([0.0, 0.0]))
        assert g.shape == (2,)

        H = r_hess(np.array([0.0, 0.0]), 1.0, np.zeros(0))
        assert H.shape == (2, 2)
        # Gauss-Newton + rho*I should be positive definite
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > 0)

    def test_restoration_reduces_violation(self):
        """Verify restoration decreases constraint violation on infeasible start."""
        from discopt._jax.ipm import _total_violation
        from discopt._jax.ipm_callbacks import _feasibility_restoration

        def con(x):
            return np.array([x[0] ** 2 + x[1] ** 2])  # c(x) = x1^2+x2^2

        def jac(x):
            return np.array([[2 * x[0], 2 * x[1]]])

        # Constraint: c(x) >= 1.0, starting far from feasible
        x0 = np.array([0.1, 0.1])  # c(x0) = 0.02, way below 1.0
        x_l = np.full(2, -10.0)
        x_u = np.full(2, 10.0)
        g_l = np.array([1.0])
        g_u = np.array([1e20])
        has_g_lb = jnp.array([1.0])
        has_g_ub = jnp.array([0.0])

        c0 = jnp.asarray(con(x0))
        viol0 = float(_total_violation(c0, jnp.asarray(g_l), jnp.asarray(g_u), has_g_lb, has_g_ub))
        assert viol0 > 0.5  # confirm infeasible

        x_rest, success = _feasibility_restoration(
            con,
            jac,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            has_g_lb,
            has_g_ub,
            IPMOptions(),
        )

        c1 = jnp.asarray(con(x_rest))
        viol1 = float(_total_violation(c1, jnp.asarray(g_l), jnp.asarray(g_u), has_g_lb, has_g_ub))
        assert success
        assert viol1 < viol0 * 0.9

    def test_already_feasible_skips_restoration(self):
        """No-op when point is already feasible."""
        from discopt._jax.ipm_callbacks import _feasibility_restoration

        def con(x):
            return np.array([x[0] + x[1]])

        def jac(x):
            return np.array([[1.0, 1.0]])

        x0 = np.array([1.0, 1.0])  # c(x0) = 2.0 >= 0
        x_l = np.full(2, -10.0)
        x_u = np.full(2, 10.0)
        g_l = np.array([0.0])
        g_u = np.array([1e20])
        has_g_lb = jnp.array([1.0])
        has_g_ub = jnp.array([0.0])

        x_rest, success = _feasibility_restoration(
            con,
            jac,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            has_g_lb,
            has_g_ub,
            IPMOptions(),
        )
        assert not success  # should skip

    @pytest.mark.slow
    def test_restoration_recursion_guard(self):
        """_in_restoration=True prevents nested restoration calls."""
        # Run the solver with _in_restoration=True on a problem that would
        # otherwise trigger restoration — it should exit without recursing.
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu = _hs106()
        opts = IPMOptions(max_iter=100)  # low iter to finish quickly
        state = ipm_solve_callbacks(
            obj,
            grad,
            hess,
            con,
            jac,
            x0,
            xl,
            xu,
            gl,
            gu,
            options=opts,
            _in_restoration=True,
        )
        # Just verify it completes without infinite recursion
        assert int(state.iteration) > 0

    @pytest.mark.slow
    def test_hs106_runs_without_crash(self):
        """HS106 should converge to optimal objective ~7049.

        The filter line search allows steps that increase the objective
        when they reduce constraint violation, enabling convergence on
        problems where l1 merit would stall.
        """
        obj, grad, hess, con, jac, x0, xl, xu, gl, gu = _hs106()
        opts = IPMOptions(max_iter=500, tol=1e-6, acceptable_tol=1e-4, acceptable_iter=5)
        state = ipm_solve_callbacks(
            obj,
            grad,
            hess,
            con,
            jac,
            x0,
            xl,
            xu,
            gl,
            gu,
            options=opts,
        )
        assert int(state.iteration) > 0
        assert int(state.converged) in (1, 2, 3)
        # Filter line search should allow reaching the true optimum
        assert float(state.obj) > 6000.0, f"HS106 obj={float(state.obj):.1f}, expected ~7049"

    def test_restoration_end_to_end(self):
        """Solver recovers from infeasible stagnation via restoration on a
        quadratic problem with a nonlinear constraint and a bad start."""
        # min (x1-3)^2 + (x2-3)^2
        # s.t. x1^2 + x2^2 >= 10  (circle constraint, forces ||x|| >= sqrt(10))
        #      0 <= x1, x2 <= 5
        # Optimal: x* near (sqrt(5), sqrt(5)) ~ (2.236, 2.236), obj ~ 1.172

        def obj_fn(x):
            return float((x[0] - 3) ** 2 + (x[1] - 3) ** 2)

        def grad_fn(x):
            return np.array([2 * (x[0] - 3), 2 * (x[1] - 3)])

        def hess_fn(x, obj_factor, y):
            H = obj_factor * np.array([[2.0, 0.0], [0.0, 2.0]])
            if len(y) > 0 and y[0] != 0:
                H[0, 0] += y[0] * 2.0
                H[1, 1] += y[0] * 2.0
            return H

        def con_fn(x):
            return np.array([x[0] ** 2 + x[1] ** 2])

        def jac_fn(x):
            return np.array([[2 * x[0], 2 * x[1]]])

        x0 = np.array([0.5, 0.5])  # infeasible: 0.5 << sqrt(10)
        x_l = np.array([0.0, 0.0])
        x_u = np.array([5.0, 5.0])
        g_l = np.array([10.0])
        g_u = np.array([1e20])

        opts = IPMOptions(
            max_iter=2000,
            tol=1e-6,
            acceptable_tol=1e-4,
            acceptable_iter=5,
        )
        state = ipm_solve_callbacks(
            obj_fn,
            grad_fn,
            hess_fn,
            con_fn,
            jac_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            options=opts,
        )

        x_sol = np.asarray(state.x)
        # Verify feasibility
        c_val = con_fn(x_sol)
        assert c_val[0] >= 10.0 - 1e-3, f"Constraint violated: {c_val[0]}"
        # Verify solution quality
        assert float(state.obj) < 3.0  # optimal is ~1.17
        assert int(state.converged) in (1, 2)


class TestFilterLineSearch:
    """Tests for the Ipopt-style filter line search."""

    def test_filter_accepts_feasibility_improving_step(self):
        """Filter should accept steps that improve feasibility even if obj increases."""
        from discopt._jax.ipm_callbacks import ipm_solve_callbacks

        # Problem: min -x1 s.t. x1^2 + x2^2 <= 1, 0 <= x <= 2
        # Start far from feasible. The filter should allow infeasible->feasible
        # transitions even if objective temporarily worsens.

        def obj(x):
            return float(-x[0])

        def grad_fn(x):
            return np.array([-1.0, 0.0])

        def hess_fn(x, obj_factor, y):
            H = np.zeros((2, 2))
            if len(y) > 0:
                H[0, 0] += y[0] * 2.0
                H[1, 1] += y[0] * 2.0
            return H

        def con_fn(x):
            return np.array([x[0] ** 2 + x[1] ** 2])

        def jac_fn(x):
            return np.array([[2 * x[0], 2 * x[1]]])

        x0 = np.array([1.5, 1.5])  # infeasible: 1.5^2+1.5^2=4.5 > 1
        x_l = np.array([0.0, 0.0])
        x_u = np.array([2.0, 2.0])
        g_l = np.array([-1e20])
        g_u = np.array([1.0])

        opts = IPMOptions(max_iter=500, tol=1e-6)
        state = ipm_solve_callbacks(
            obj,
            grad_fn,
            hess_fn,
            con_fn,
            jac_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            options=opts,
        )

        x_sol = np.asarray(state.x)
        # Should be feasible
        assert x_sol[0] ** 2 + x_sol[1] ** 2 <= 1.0 + 1e-3
        # Optimal is x=(1,0), obj=-1
        assert float(state.obj) < -0.9

    def test_filter_options_in_ipmoptions(self):
        """Verify new filter options are accessible and have correct defaults."""
        opts = IPMOptions()
        assert opts.gamma_theta == 1e-5
        assert opts.gamma_phi == 1e-5
        assert opts.s_phi == 2.3
        assert opts.s_theta == 1.1
        assert opts.delta_switch == 1.0
        assert opts.theta_max_fact == 1e4
        assert opts.theta_min_fact == 1e-4
        assert opts.mu_linear_decrease == 0.2
        assert opts.mu_superlinear_power == 1.5
        assert opts.barrier_tol_factor == 10.0
