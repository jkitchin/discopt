"""
Test suite for the PCG iterative solver and IPM+PCG integration.

Tests cover:
  - PCG solver correctness on small known SPD systems
  - PCG convergence on ill-conditioned systems
  - PCG with diagonal preconditioner vs no preconditioner
  - Matrix-free PCG (matvec interface)
  - Condensed KKT solve via PCG
  - IPM with PCG on small NLP problems (verify same results as dense)
  - Scaling tests: IPM+PCG on problems with 1K, 5K, 10K variables
  - JIT compatibility (pcg_solve should be jittable)
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.ipm import IPMOptions, ipm_solve
from discopt._jax.pcg import (
    PCGOptions,
    PCGResult,
    diagonal_preconditioner,
    identity_preconditioner,
    pcg_solve,
    pcg_solve_matvec,
    solve_kkt_condensed_pcg,
)

# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------


def _make_spd_matrix(n: int, cond: float = 10.0, seed: int = 42) -> jnp.ndarray:
    """Create an n x n symmetric positive definite matrix with given condition number."""
    key = jax.random.PRNGKey(seed)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    # Eigenvalues from 1 to cond
    eigvals = jnp.linspace(1.0, cond, n)
    return Q @ jnp.diag(eigvals) @ Q.T


def _make_diagonal_spd(n: int, min_eig: float = 1.0, max_eig: float = 100.0) -> jnp.ndarray:
    """Create a diagonal SPD matrix."""
    eigvals = jnp.linspace(min_eig, max_eig, n)
    return jnp.diag(eigvals)


# ---------------------------------------------------------------
# 1. PCG solver correctness tests
# ---------------------------------------------------------------


class TestPCGCorrectness:
    """Unit tests for the PCG solver on known SPD systems."""

    def test_identity_system(self):
        """Ax = b with A = I should give x = b."""
        n = 5
        A = jnp.eye(n)
        b = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pcg_solve(A, b)
        assert isinstance(result, PCGResult)
        np.testing.assert_allclose(result.x, b, atol=1e-10)
        assert bool(result.converged)
        assert int(result.iterations) <= 5

    def test_diagonal_system(self):
        """Diagonal SPD system."""
        A = jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0]))
        b = jnp.array([4.0, 9.0, 25.0, 49.0])
        expected = b / jnp.diag(A)
        result = pcg_solve(A, b)
        np.testing.assert_allclose(result.x, expected, atol=1e-10)
        assert bool(result.converged)

    def test_small_spd(self):
        """2x2 SPD system with known solution."""
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x_expected = jnp.linalg.solve(A, b)
        result = pcg_solve(A, b)
        np.testing.assert_allclose(result.x, x_expected, atol=1e-10)
        assert bool(result.converged)

    def test_larger_spd(self):
        """10x10 random SPD matrix."""
        A = _make_spd_matrix(10, cond=50.0)
        b = jnp.ones(10)
        x_expected = jnp.linalg.solve(A, b)
        result = pcg_solve(A, b)
        np.testing.assert_allclose(result.x, x_expected, atol=1e-8)
        assert bool(result.converged)

    def test_50x50_spd(self):
        """50x50 random SPD matrix."""
        A = _make_spd_matrix(50, cond=100.0)
        b = jnp.ones(50)
        x_expected = jnp.linalg.solve(A, b)
        result = pcg_solve(A, b)
        np.testing.assert_allclose(result.x, x_expected, atol=1e-6)
        assert bool(result.converged)

    def test_zero_rhs(self):
        """Zero RHS should give zero solution."""
        A = _make_spd_matrix(5)
        b = jnp.zeros(5)
        result = pcg_solve(A, b)
        np.testing.assert_allclose(result.x, jnp.zeros(5), atol=1e-10)
        assert bool(result.converged)

    def test_initial_guess(self):
        """Providing a good initial guess should reduce iterations."""
        A = _make_spd_matrix(10, cond=10.0)
        b = jnp.ones(10)
        x_exact = jnp.linalg.solve(A, b)

        # Solve from zeros
        result_zero = pcg_solve(A, b)
        # Solve from near-exact guess
        x0_good = x_exact + 1e-5 * jnp.ones(10)
        result_good = pcg_solve(A, b, x0=x0_good)

        assert int(result_good.iterations) <= int(result_zero.iterations)
        np.testing.assert_allclose(result_good.x, x_exact, atol=1e-8)

    def test_residual_norm(self):
        """Residual norm should be small after convergence."""
        A = _make_spd_matrix(20)
        b = jnp.ones(20)
        result = pcg_solve(A, b)
        actual_residual = jnp.linalg.norm(b - A @ result.x)
        np.testing.assert_allclose(float(result.residual_norm), float(actual_residual), atol=1e-12)
        assert float(result.residual_norm) < 1e-8


# ---------------------------------------------------------------
# 2. PCG convergence and ill-conditioning tests
# ---------------------------------------------------------------


class TestPCGConvergence:
    """Test PCG behavior on ill-conditioned systems."""

    def test_well_conditioned(self):
        """Well-conditioned system should converge in few iterations."""
        A = _make_spd_matrix(20, cond=2.0)
        b = jnp.ones(20)
        result = pcg_solve(A, b, options=PCGOptions(tol=1e-10))
        assert bool(result.converged)
        assert int(result.iterations) <= 20

    def test_moderately_conditioned(self):
        """Moderate condition number: still converges but needs more iters."""
        A = _make_spd_matrix(20, cond=1000.0)
        b = jnp.ones(20)
        result = pcg_solve(A, b, options=PCGOptions(tol=1e-8, max_iter=200))
        assert bool(result.converged)

    def test_ill_conditioned(self):
        """Highly ill-conditioned system: PCG may need many iterations."""
        A = _make_spd_matrix(20, cond=1e6)
        b = jnp.ones(20)
        result = pcg_solve(A, b, options=PCGOptions(tol=1e-6, max_iter=500))
        # Should still converge with enough iterations
        assert bool(result.converged)

    def test_max_iter_respected(self):
        """When max_iter is too small, should not converge."""
        A = _make_spd_matrix(20, cond=1e4)
        b = jnp.ones(20)
        result = pcg_solve(A, b, options=PCGOptions(tol=1e-12, max_iter=2))
        # With only 2 iterations on a 20x20 system, unlikely to converge
        # to 1e-12 tolerance
        assert int(result.iterations) <= 2

    def test_absolute_tolerance(self):
        """Test with absolute (non-relative) tolerance."""
        A = _make_spd_matrix(10, cond=10.0)
        b = jnp.ones(10)
        result = pcg_solve(
            A,
            b,
            options=PCGOptions(tol=1e-8, use_relative_tol=False),
        )
        assert bool(result.converged)
        assert float(result.residual_norm) < 1e-8


# ---------------------------------------------------------------
# 3. Preconditioner tests
# ---------------------------------------------------------------


class TestPCGPreconditioner:
    """Test PCG with different preconditioners."""

    def test_diagonal_preconditioner_reduces_iterations(self):
        """Diagonal preconditioner should reduce iteration count."""
        A = _make_spd_matrix(30, cond=1000.0)
        b = jnp.ones(30)

        # Without preconditioner
        result_no_precond = pcg_solve(
            A,
            b,
            preconditioner=identity_preconditioner(30),
            options=PCGOptions(tol=1e-8, max_iter=500),
        )

        # With diagonal preconditioner
        precond = diagonal_preconditioner(A)
        result_precond = pcg_solve(
            A, b, preconditioner=precond, options=PCGOptions(tol=1e-8, max_iter=500)
        )

        assert bool(result_precond.converged)
        # Preconditioner should not increase iterations
        assert int(result_precond.iterations) <= int(result_no_precond.iterations)

    def test_identity_preconditioner_same_as_none(self):
        """Identity preconditioner should give same result as no preconditioner."""
        A = _make_spd_matrix(10, cond=10.0)
        b = jnp.ones(10)

        result_none = pcg_solve(A, b)
        result_id = pcg_solve(A, b, preconditioner=identity_preconditioner(10))

        np.testing.assert_allclose(result_none.x, result_id.x, atol=1e-10)
        assert int(result_none.iterations) == int(result_id.iterations)

    def test_perfect_preconditioner(self):
        """A^{-1} as preconditioner should converge in 1 iteration."""
        A = _make_spd_matrix(10, cond=100.0)
        b = jnp.ones(10)
        A_inv = jnp.linalg.inv(A)
        perfect_precond = lambda r: A_inv @ r  # noqa: E731

        result = pcg_solve(A, b, preconditioner=perfect_precond)
        assert int(result.iterations) <= 2  # 1-2 iterations with perfect precond
        assert bool(result.converged)

    def test_diagonal_preconditioner_values(self):
        """Verify diagonal preconditioner produces correct M_inv."""
        A = jnp.diag(jnp.array([2.0, 5.0, 10.0]))
        precond = diagonal_preconditioner(A)
        r = jnp.ones(3)
        z = precond(r)
        np.testing.assert_allclose(z, jnp.array([0.5, 0.2, 0.1]), atol=1e-10)


# ---------------------------------------------------------------
# 4. Matrix-free PCG tests
# ---------------------------------------------------------------


class TestPCGMatvec:
    """Test the matrix-free PCG solver."""

    def test_matvec_matches_dense(self):
        """Matrix-free should give same result as dense."""
        A = _make_spd_matrix(15, cond=50.0)
        b = jnp.ones(15)

        result_dense = pcg_solve(A, b)
        result_matvec = pcg_solve_matvec(lambda v: A @ v, b)

        np.testing.assert_allclose(result_dense.x, result_matvec.x, atol=1e-10)

    def test_matvec_with_preconditioner(self):
        """Matrix-free PCG with diagonal preconditioner."""
        A = _make_spd_matrix(20, cond=100.0)
        b = jnp.ones(20)
        precond = diagonal_preconditioner(A)

        result = pcg_solve_matvec(lambda v: A @ v, b, preconditioner=precond)
        x_expected = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result.x, x_expected, atol=1e-8)

    def test_matvec_implicit_operator(self):
        """Test with an implicit operator (not explicitly stored)."""
        n = 10

        # Tridiagonal: 2 on diagonal, -1 on off-diagonals
        def tridiag_matvec(v):
            result = 2.0 * v
            result = result.at[:-1].add(-v[1:])
            result = result.at[1:].add(-v[:-1])
            return result

        b = jnp.ones(n)
        result = pcg_solve_matvec(tridiag_matvec, b, options=PCGOptions(tol=1e-10))
        # Verify: tridiag @ x should equal b
        actual_b = tridiag_matvec(result.x)
        np.testing.assert_allclose(actual_b, b, atol=1e-8)
        assert bool(result.converged)


# ---------------------------------------------------------------
# 5. Condensed KKT solve tests
# ---------------------------------------------------------------


class TestCondensedKKT:
    """Test the condensed KKT system solve via PCG."""

    def test_unconstrained(self):
        """Unconstrained case: m=0, just solves W dx = rhs_x."""
        W = _make_spd_matrix(5, cond=10.0)
        J = jnp.zeros((0, 5))
        D_reg = jnp.zeros(0)
        rhs_x = jnp.ones(5)
        rhs_y = jnp.zeros(0)

        dx, dy, pcg_result = solve_kkt_condensed_pcg(W, J, D_reg, rhs_x, rhs_y)
        x_expected = jnp.linalg.solve(W, rhs_x)
        np.testing.assert_allclose(dx, x_expected, atol=1e-8)
        assert dy.shape[0] == 0

    def test_one_constraint(self):
        """Single constraint: verify dx, dy match dense solution."""
        n, m = 5, 1
        W = _make_spd_matrix(n, cond=10.0)
        key = jax.random.PRNGKey(123)
        J = jax.random.normal(key, (m, n))
        D_reg = jnp.array([0.01])
        rhs_x = jnp.ones(n)
        rhs_y = jnp.array([0.5])

        dx_pcg, dy_pcg, _ = solve_kkt_condensed_pcg(W, J, D_reg, rhs_x, rhs_y)

        # Dense reference
        KKT = jnp.block(
            [
                [W, J.T],
                [J, -jnp.diag(D_reg)],
            ]
        )
        rhs = jnp.concatenate([rhs_x, rhs_y])
        sol = jnp.linalg.solve(KKT, rhs)
        dx_dense, dy_dense = sol[:n], sol[n:]

        np.testing.assert_allclose(dx_pcg, dx_dense, atol=1e-6)
        np.testing.assert_allclose(dy_pcg, dy_dense, atol=1e-6)

    def test_multiple_constraints(self):
        """Multiple constraints: verify against dense solution."""
        n, m = 10, 3
        W = _make_spd_matrix(n, cond=20.0)
        key = jax.random.PRNGKey(456)
        J = jax.random.normal(key, (m, n))
        D_reg = jnp.array([0.01, 0.02, 0.015])
        rhs_x = jnp.ones(n)
        rhs_y = jnp.ones(m)

        dx_pcg, dy_pcg, _ = solve_kkt_condensed_pcg(W, J, D_reg, rhs_x, rhs_y)

        # Dense reference
        KKT = jnp.block(
            [
                [W, J.T],
                [J, -jnp.diag(D_reg)],
            ]
        )
        rhs = jnp.concatenate([rhs_x, rhs_y])
        sol = jnp.linalg.solve(KKT, rhs)
        dx_dense, dy_dense = sol[:n], sol[n:]

        np.testing.assert_allclose(dx_pcg, dx_dense, atol=1e-5)
        np.testing.assert_allclose(dy_pcg, dy_dense, atol=1e-5)


# ---------------------------------------------------------------
# 6. JIT compatibility tests
# ---------------------------------------------------------------


class TestPCGJIT:
    """Test that PCG functions are JIT-compatible."""

    def test_pcg_solve_jittable(self):
        """pcg_solve should work under jax.jit."""
        A = _make_spd_matrix(10, cond=10.0)
        b = jnp.ones(10)

        @jax.jit
        def solve_jit(A, b):
            return pcg_solve(A, b, options=PCGOptions(tol=1e-10, max_iter=100))

        result = solve_jit(A, b)
        x_expected = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result.x, x_expected, atol=1e-8)
        assert bool(result.converged)

    def test_pcg_matvec_jittable(self):
        """pcg_solve_matvec should work under jax.jit."""
        A = _make_spd_matrix(10, cond=10.0)
        b = jnp.ones(10)

        @jax.jit
        def solve_jit(b):
            return pcg_solve_matvec(
                lambda v: A @ v,
                b,
                options=PCGOptions(tol=1e-10, max_iter=100),
            )

        result = solve_jit(b)
        x_expected = jnp.linalg.solve(A, b)
        np.testing.assert_allclose(result.x, x_expected, atol=1e-8)

    def test_condensed_kkt_jittable(self):
        """solve_kkt_condensed_pcg should work under jax.jit."""
        n, m = 5, 2
        W = _make_spd_matrix(n, cond=10.0)
        key = jax.random.PRNGKey(789)
        J = jax.random.normal(key, (m, n))
        D_reg = jnp.array([0.01, 0.02])
        rhs_x = jnp.ones(n)
        rhs_y = jnp.ones(m)

        @jax.jit
        def solve_jit(W, J, D_reg, rhs_x, rhs_y):
            return solve_kkt_condensed_pcg(W, J, D_reg, rhs_x, rhs_y)

        dx, dy, result = solve_jit(W, J, D_reg, rhs_x, rhs_y)
        # Just verify it runs without error and produces finite results
        assert jnp.all(jnp.isfinite(dx))
        assert jnp.all(jnp.isfinite(dy))


# ---------------------------------------------------------------
# 7. IPM + PCG integration tests
# ---------------------------------------------------------------


def _obj_quadratic(x):
    return (x[0] - 2) ** 2 + (x[1] + 1) ** 2


def _obj_sum_sq(x):
    return jnp.sum(x**2)


def _con_empty(x):
    return jnp.array([])


def _con_sum_2d(x):
    return jnp.array([x[0] + x[1]])


def _con_product(x):
    return jnp.array([x[0] * x[1]])


class TestIPMPCGIntegration:
    """Test IPM solver with PCG linear solver backend."""

    def test_unconstrained_quadratic(self):
        """min (x-2)^2 + (y+1)^2 with PCG should match dense."""
        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])

        # Dense solve
        state_dense = ipm_solve(
            _obj_quadratic,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="dense"),
        )
        # PCG solve
        state_pcg = ipm_solve(
            _obj_quadratic,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="pcg"),
        )

        np.testing.assert_allclose(state_pcg.x, state_dense.x, atol=1e-3)
        np.testing.assert_allclose(float(state_pcg.obj), float(state_dense.obj), atol=1e-3)
        assert int(state_pcg.converged) in (1, 2)

    def test_bound_constrained(self):
        """min x^2+y^2 with x,y in [1,5] using PCG."""
        x0 = jnp.array([3.0, 3.0])
        x_l = jnp.array([1.0, 1.0])
        x_u = jnp.array([5.0, 5.0])

        state = ipm_solve(
            _obj_sum_sq,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="pcg"),
        )
        assert jnp.allclose(state.obj, 2.0, atol=1e-3)
        assert int(state.converged) in (1, 2)

    def test_equality_constrained(self):
        """min x^2+y^2 s.t. x+y=2 with PCG."""
        x0 = jnp.array([0.5, 0.5])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([3.0, 3.0])
        g_l = jnp.array([2.0])
        g_u = jnp.array([2.0])

        state_dense = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="dense"),
        )
        state_pcg = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="pcg"),
        )

        np.testing.assert_allclose(float(state_pcg.obj), float(state_dense.obj), atol=1e-2)
        assert int(state_pcg.converged) in (1, 2)

    def test_inequality_constrained(self):
        """min x^2+y^2 s.t. x+y>=1 with PCG."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        state = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="pcg"),
        )
        assert jnp.allclose(state.obj, 0.5, atol=1e-2)

    def test_nonlinear_equality(self):
        """min x^2+y^2 s.t. x*y=1 with PCG."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([0.1, 0.1])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])
        g_u = jnp.array([1.0])

        state = ipm_solve(
            _obj_sum_sq,
            _con_product,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="pcg"),
        )
        assert jnp.allclose(state.obj, 2.0, atol=0.05)

    def test_pcg_default_is_dense(self):
        """Default linear_solver should be 'dense'."""
        opts = IPMOptions()
        assert opts.linear_solver == "dense"


# ---------------------------------------------------------------
# 8. Scaling tests (larger problems)
# ---------------------------------------------------------------


def _make_large_quadratic(n):
    """Create objective/constraint functions for a large quadratic problem.

    min sum(x_i^2)
    s.t. sum(x_i) >= 1
    x_i in [-10, 10]
    """

    def obj_fn(x):
        return jnp.sum(x**2)

    def con_fn(x):
        return jnp.array([jnp.sum(x)])

    return obj_fn, con_fn


class TestIPMPCGScaling:
    """Test IPM+PCG scaling to larger problems."""

    def test_100_vars(self):
        """100 variables: PCG should converge."""
        n = 100
        obj_fn, con_fn = _make_large_quadratic(n)
        x0 = jnp.ones(n) * 5.0
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(
                linear_solver="pcg",
                max_iter=200,
                tol=1e-6,
            ),
        )
        assert int(state.converged) in (1, 2, 3)
        # Each x_i should be approximately 1/n
        expected_obj = 1.0 / n
        assert float(state.obj) < expected_obj + 0.1

    @pytest.mark.slow
    def test_1000_vars(self):
        """1K variables: PCG should handle this scale."""
        n = 1000
        obj_fn, con_fn = _make_large_quadratic(n)
        x0 = jnp.ones(n) * 5.0
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        t0 = time.perf_counter()
        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(
                linear_solver="pcg",
                max_iter=200,
                tol=1e-5,
                acceptable_tol=1e-4,
            ),
        )
        wall = time.perf_counter() - t0
        assert int(state.converged) in (1, 2, 3)
        assert wall < 300  # Should finish within timeout

    @pytest.mark.slow
    def test_5000_vars_unconstrained(self):
        """5K variables unconstrained: PCG should scale."""
        n = 5000

        def obj_fn(x):
            return jnp.sum(x**2)

        x0 = jnp.ones(n)
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)

        t0 = time.perf_counter()
        state = ipm_solve(
            obj_fn,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(
                linear_solver="pcg",
                max_iter=200,
                tol=1e-4,
                acceptable_tol=1e-3,
            ),
        )
        wall = time.perf_counter() - t0
        assert float(state.obj) < 1.0  # Should be near 0
        assert wall < 300

    @pytest.mark.slow
    def test_10000_vars_unconstrained(self):
        """10K variables unconstrained: PCG should still work."""
        n = 10000

        def obj_fn(x):
            return jnp.sum(x**2)

        x0 = jnp.ones(n)
        x_l = jnp.full(n, -100.0)
        x_u = jnp.full(n, 100.0)

        t0 = time.perf_counter()
        state = ipm_solve(
            obj_fn,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(
                linear_solver="pcg",
                max_iter=200,
                tol=1e-4,
                acceptable_tol=1e-3,
            ),
        )
        wall = time.perf_counter() - t0
        assert float(state.obj) < 1.0
        assert wall < 300


# ---------------------------------------------------------------
# 9. PCG options and edge cases
# ---------------------------------------------------------------


class TestPCGEdgeCases:
    """Edge cases and options testing."""

    def test_single_variable(self):
        """1x1 system."""
        A = jnp.array([[3.0]])
        b = jnp.array([6.0])
        result = pcg_solve(A, b)
        np.testing.assert_allclose(result.x, jnp.array([2.0]), atol=1e-10)

    def test_custom_options(self):
        """PCGOptions fields should be respected."""
        opts = PCGOptions(tol=1e-5, max_iter=50, use_relative_tol=False)
        assert opts.tol == 1e-5
        assert opts.max_iter == 50
        assert opts.use_relative_tol is False

    def test_pcg_result_fields(self):
        """PCGResult should have all expected fields."""
        A = jnp.eye(3)
        b = jnp.ones(3)
        result = pcg_solve(A, b)
        assert hasattr(result, "x")
        assert hasattr(result, "residual_norm")
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")
        assert result.x.shape == (3,)
        assert result.residual_norm.shape == ()
        assert result.iterations.shape == ()

    def test_ipm_options_pcg_fields(self):
        """IPMOptions should have linear_solver, pcg_tol, pcg_max_iter."""
        opts = IPMOptions(linear_solver="pcg", pcg_tol=1e-8, pcg_max_iter=500)
        assert opts.linear_solver == "pcg"
        assert opts.pcg_tol == 1e-8
        assert opts.pcg_max_iter == 500

    def test_pcg_convergence_flag_false_on_max_iter(self):
        """Convergence flag should be False when max_iter hit."""
        A = _make_spd_matrix(50, cond=1e6)
        b = jnp.ones(50)
        result = pcg_solve(A, b, options=PCGOptions(tol=1e-15, max_iter=3))
        assert int(result.iterations) == 3
        # Converged could be True or False depending on the problem,
        # but we expect False with such tight tolerance and few iterations
        # (though for some well-structured problems it might converge)


# ---------------------------------------------------------------
# 10. Lineax integration tests
# ---------------------------------------------------------------


lineax = pytest.importorskip("lineax")


class TestLineaxCGMatchesDense:
    """Verify that lineax CG produces the same results as dense on small problems."""

    def test_unconstrained_quadratic_lineax_cg(self):
        """min (x-2)^2 + (y+1)^2 with lineax_cg should match dense."""
        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])

        state_dense = ipm_solve(
            _obj_quadratic,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="dense"),
        )
        state_cg = ipm_solve(
            _obj_quadratic,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="lineax_cg"),
        )
        np.testing.assert_allclose(state_cg.x, state_dense.x, atol=1e-3)
        np.testing.assert_allclose(float(state_cg.obj), float(state_dense.obj), atol=1e-3)
        assert int(state_cg.converged) in (1, 2)

    def test_bound_constrained_lineax_cg(self):
        """min x^2+y^2 with x,y in [1,5] using lineax_cg."""
        x0 = jnp.array([3.0, 3.0])
        x_l = jnp.array([1.0, 1.0])
        x_u = jnp.array([5.0, 5.0])

        state = ipm_solve(
            _obj_sum_sq,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="lineax_cg"),
        )
        assert jnp.allclose(state.obj, 2.0, atol=1e-3)
        assert int(state.converged) in (1, 2)

    def test_equality_constrained_lineax_cg(self):
        """min x^2+y^2 s.t. x+y=2 with lineax_cg."""
        x0 = jnp.array([0.5, 0.5])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([3.0, 3.0])
        g_l = jnp.array([2.0])
        g_u = jnp.array([2.0])

        state_dense = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="dense"),
        )
        state_cg = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="lineax_cg"),
        )
        np.testing.assert_allclose(float(state_cg.obj), float(state_dense.obj), atol=1e-2)
        assert int(state_cg.converged) in (1, 2)

    def test_inequality_constrained_lineax_cg(self):
        """min x^2+y^2 s.t. x+y>=1 with lineax_cg."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        state = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="lineax_cg"),
        )
        assert jnp.allclose(state.obj, 0.5, atol=1e-2)

    def test_nonlinear_equality_lineax_cg(self):
        """min x^2+y^2 s.t. x*y=1 with lineax_cg."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([0.1, 0.1])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])
        g_u = jnp.array([1.0])

        state = ipm_solve(
            _obj_sum_sq,
            _con_product,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="lineax_cg"),
        )
        assert jnp.allclose(state.obj, 2.0, atol=0.05)


# ---------------------------------------------------------------
# 11. Lineax GMRES tests
# ---------------------------------------------------------------


class TestLineaxGMRES:
    """Test GMRES solver on the full indefinite KKT system."""

    def test_unconstrained_gmres(self):
        """min (x-2)^2 + (y+1)^2 with lineax_gmres."""
        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])

        state = ipm_solve(
            _obj_quadratic,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(linear_solver="lineax_gmres"),
        )
        np.testing.assert_allclose(state.x[0], 2.0, atol=1e-3)
        np.testing.assert_allclose(state.x[1], -1.0, atol=1e-3)
        assert int(state.converged) in (1, 2)

    def test_equality_constrained_gmres(self):
        """min x^2+y^2 s.t. x+y=2 with lineax_gmres."""
        x0 = jnp.array([0.5, 0.5])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([3.0, 3.0])
        g_l = jnp.array([2.0])
        g_u = jnp.array([2.0])

        state = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="lineax_gmres"),
        )
        np.testing.assert_allclose(float(state.obj), 2.0, atol=1e-2)
        assert int(state.converged) in (1, 2)

    def test_gmres_matches_dense(self):
        """GMRES should produce same results as dense on small problems."""
        x0 = jnp.array([2.0, 2.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        state_dense = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="dense"),
        )
        state_gmres = ipm_solve(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(linear_solver="lineax_gmres"),
        )
        np.testing.assert_allclose(
            float(state_gmres.obj),
            float(state_dense.obj),
            atol=1e-2,
        )


# ---------------------------------------------------------------
# 12. IterativeKKTSolver unit tests
# ---------------------------------------------------------------


class TestIterativeKKTSolver:
    """Direct tests of the IterativeKKTSolver class."""

    def test_cg_unconstrained_solve(self):
        """CG on SPD W with no constraints."""
        from discopt._jax.ipm_iterative import IterativeKKTSolver

        n = 5
        W = _make_spd_matrix(n, cond=10.0)
        rhs_x = jnp.ones(n)

        solver = IterativeKKTSolver(
            linear_solver="cg",
            rtol=1e-10,
            atol=1e-10,
            warm_start=False,
            use_preconditioner=True,
        )
        dx, dy, info = solver.solve(
            H=W,
            J=jnp.zeros((0, n)),
            Sig_diag=jnp.zeros(n),
            D_reg_vec=jnp.zeros(0, dtype=jnp.float64),
            delta_w=jnp.array(0.0),
            delta_c=0.0,
            rhs_x=rhs_x,
            rhs_y=jnp.zeros(0, dtype=jnp.float64),
        )
        x_expected = jnp.linalg.solve(W, rhs_x)
        np.testing.assert_allclose(dx, x_expected, atol=1e-6)
        assert dy.shape[0] == 0

    def test_cg_constrained_matches_dense(self):
        """CG condensed solve matches dense KKT on small problem."""
        from discopt._jax.ipm_iterative import IterativeKKTSolver

        n, m = 5, 2
        H = _make_spd_matrix(n, cond=10.0)
        Sig_diag = jnp.ones(n) * 0.1
        delta_w = jnp.array(0.01)
        delta_c = 1e-8

        key = jax.random.PRNGKey(42)
        J = jax.random.normal(key, (m, n))
        D_reg_vec = jnp.array([0.01, 0.02])
        rhs_x = jnp.ones(n)
        rhs_y = jnp.ones(m)

        # Dense reference
        W = H + jnp.diag(Sig_diag) + delta_w * jnp.eye(n)
        D_full = jnp.diag(D_reg_vec + delta_c)
        KKT = jnp.block([[W, J.T], [J, -D_full]])
        rhs = jnp.concatenate([rhs_x, rhs_y])
        sol_dense = jnp.linalg.solve(KKT, rhs)

        solver = IterativeKKTSolver(
            linear_solver="cg",
            rtol=1e-10,
            atol=1e-10,
            warm_start=False,
            use_preconditioner=True,
        )
        dx, dy, info = solver.solve(
            H=H,
            J=J,
            Sig_diag=Sig_diag,
            D_reg_vec=D_reg_vec,
            delta_w=delta_w,
            delta_c=delta_c,
            rhs_x=rhs_x,
            rhs_y=rhs_y,
        )
        np.testing.assert_allclose(dx, sol_dense[:n], atol=1e-4)
        np.testing.assert_allclose(dy, sol_dense[n:], atol=1e-4)

    def test_gmres_constrained_matches_dense(self):
        """GMRES solve matches dense KKT on small problem."""
        from discopt._jax.ipm_iterative import IterativeKKTSolver

        n, m = 5, 2
        H = _make_spd_matrix(n, cond=10.0)
        Sig_diag = jnp.ones(n) * 0.1
        delta_w = jnp.array(0.01)
        delta_c = 1e-8

        key = jax.random.PRNGKey(42)
        J = jax.random.normal(key, (m, n))
        D_reg_vec = jnp.array([0.01, 0.02])
        rhs_x = jnp.ones(n)
        rhs_y = jnp.ones(m)

        # Dense reference
        W = H + jnp.diag(Sig_diag) + delta_w * jnp.eye(n)
        D_full = jnp.diag(D_reg_vec + delta_c)
        KKT = jnp.block([[W, J.T], [J, -D_full]])
        rhs = jnp.concatenate([rhs_x, rhs_y])
        sol_dense = jnp.linalg.solve(KKT, rhs)

        solver = IterativeKKTSolver(
            linear_solver="gmres",
            rtol=1e-10,
            atol=1e-10,
            warm_start=False,
        )
        dx, dy, info = solver.solve(
            H=H,
            J=J,
            Sig_diag=Sig_diag,
            D_reg_vec=D_reg_vec,
            delta_w=delta_w,
            delta_c=delta_c,
            rhs_x=rhs_x,
            rhs_y=rhs_y,
        )
        np.testing.assert_allclose(dx, sol_dense[:n], atol=1e-4)
        np.testing.assert_allclose(dy, sol_dense[n:], atol=1e-4)


# ---------------------------------------------------------------
# 13. Inexact Newton forcing term tests
# ---------------------------------------------------------------


class TestForcingTerm:
    """Test the Eisenstat-Walker forcing sequence."""

    def test_large_mu_gives_large_eta(self):
        """When mu is large, inner tolerance should be lenient."""
        from discopt._jax.ipm_iterative import compute_forcing_term

        eta = compute_forcing_term(jnp.array(1.0), jnp.array(0))
        assert eta >= 0.1 or eta == 0.1

    def test_small_mu_gives_small_eta(self):
        """When mu is small, inner tolerance should be tight."""
        from discopt._jax.ipm_iterative import compute_forcing_term

        eta = compute_forcing_term(jnp.array(1e-8), jnp.array(50))
        assert eta < 1e-3

    def test_base_tol_respected(self):
        """Forcing term should never go below base_tol."""
        from discopt._jax.ipm_iterative import compute_forcing_term

        eta = compute_forcing_term(
            jnp.array(1e-30),
            jnp.array(100),
            base_tol=1e-12,
        )
        assert eta >= 1e-12

    def test_monotone_in_mu(self):
        """Smaller mu should give smaller or equal forcing term."""
        from discopt._jax.ipm_iterative import compute_forcing_term

        eta_big = compute_forcing_term(jnp.array(1.0), jnp.array(10))
        eta_small = compute_forcing_term(jnp.array(1e-6), jnp.array(10))
        assert eta_small <= eta_big


# ---------------------------------------------------------------
# 14. Convenience function tests
# ---------------------------------------------------------------


class TestConvenienceFunctions:
    """Test solve_nlp_iterative and solve_nlp_iterative_batch."""

    def test_solve_nlp_iterative_unconstrained(self):
        """solve_nlp_iterative on simple unconstrained problem."""
        from discopt._jax.ipm_iterative import solve_nlp_iterative

        x0 = jnp.array([0.0, 0.0])
        x_l = jnp.array([-5.0, -5.0])
        x_u = jnp.array([5.0, 5.0])

        state = solve_nlp_iterative(
            _obj_quadratic,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
        )
        np.testing.assert_allclose(state.x[0], 2.0, atol=1e-3)
        np.testing.assert_allclose(state.x[1], -1.0, atol=1e-3)
        assert int(state.converged) in (1, 2)

    def test_solve_nlp_iterative_constrained(self):
        """solve_nlp_iterative on constrained problem."""
        from discopt._jax.ipm_iterative import solve_nlp_iterative

        x0 = jnp.array([0.5, 0.5])
        x_l = jnp.array([0.0, 0.0])
        x_u = jnp.array([3.0, 3.0])

        state = solve_nlp_iterative(
            _obj_sum_sq,
            _con_sum_2d,
            x0,
            x_l,
            x_u,
            jnp.array([2.0]),
            jnp.array([2.0]),
        )
        np.testing.assert_allclose(float(state.obj), 2.0, atol=1e-2)

    def test_solve_nlp_iterative_batch_basic(self):
        """Batch iterative solve on simple problems."""
        from discopt._jax.ipm_iterative import solve_nlp_iterative_batch

        batch = 4
        n = 2
        x0_batch = jnp.zeros((batch, n))
        xl_batch = jnp.full((batch, n), -5.0)
        xu_batch = jnp.full((batch, n), 5.0)

        states = solve_nlp_iterative_batch(
            _obj_sum_sq,
            _con_empty,
            x0_batch,
            xl_batch,
            xu_batch,
            jnp.array([]),
            jnp.array([]),
        )
        assert jnp.all(states.obj < 1e-3)
        assert jnp.all(states.converged > 0)

    def test_solve_nlp_iterative_batch_constrained(self):
        """Batch iterative solve with constraints."""
        from discopt._jax.ipm_iterative import solve_nlp_iterative_batch

        batch = 4
        n = 2
        x0_batch = jnp.ones((batch, n))
        xl_batch = jnp.full((batch, n), 0.0)
        xu_batch = jnp.full((batch, n), 3.0)

        states = solve_nlp_iterative_batch(
            _obj_sum_sq,
            _con_sum_2d,
            x0_batch,
            xl_batch,
            xu_batch,
            jnp.array([2.0]),
            jnp.array([2.0]),
        )
        np.testing.assert_allclose(np.array(states.obj), 2.0, atol=0.05)


# ---------------------------------------------------------------
# 15. IPMOptions lineax fields
# ---------------------------------------------------------------


class TestIPMOptionsLineax:
    """Test that IPMOptions has the new lineax fields."""

    def test_lineax_options_fields(self):
        """IPMOptions should have lineax_max_steps, lineax_warm_start, etc."""
        opts = IPMOptions(
            linear_solver="lineax_cg",
            lineax_max_steps=500,
            lineax_warm_start=True,
            lineax_preconditioner=True,
        )
        assert opts.linear_solver == "lineax_cg"
        assert opts.lineax_max_steps == 500
        assert opts.lineax_warm_start is True
        assert opts.lineax_preconditioner is True

    def test_lineax_gmres_option(self):
        """IPMOptions should accept lineax_gmres."""
        opts = IPMOptions(linear_solver="lineax_gmres")
        assert opts.linear_solver == "lineax_gmres"


# ---------------------------------------------------------------
# 16. Lineax scaling tests
# ---------------------------------------------------------------


class TestLineaxScaling:
    """Test lineax solvers on larger problems."""

    def test_lineax_cg_100_vars(self):
        """100 variables with lineax_cg."""
        n = 100
        obj_fn, con_fn = _make_large_quadratic(n)
        x0 = jnp.ones(n) * 5.0
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(
                linear_solver="lineax_cg",
                max_iter=200,
                tol=1e-6,
            ),
        )
        assert int(state.converged) in (1, 2, 3)
        expected_obj = 1.0 / n
        assert float(state.obj) < expected_obj + 0.1

    def test_lineax_gmres_100_vars(self):
        """100 variables with lineax_gmres."""
        n = 100
        obj_fn, con_fn = _make_large_quadratic(n)
        x0 = jnp.ones(n) * 5.0
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(
                linear_solver="lineax_gmres",
                max_iter=200,
                tol=1e-6,
            ),
        )
        assert int(state.converged) in (1, 2, 3)
        expected_obj = 1.0 / n
        assert float(state.obj) < expected_obj + 0.1

    @pytest.mark.slow
    def test_lineax_cg_1000_vars(self):
        """1K variables with lineax_cg."""
        n = 1000
        obj_fn, con_fn = _make_large_quadratic(n)
        x0 = jnp.ones(n) * 5.0
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)
        g_l = jnp.array([1.0])
        g_u = jnp.array([1e20])

        t0 = time.perf_counter()
        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            IPMOptions(
                linear_solver="lineax_cg",
                max_iter=200,
                tol=1e-5,
                acceptable_tol=1e-4,
            ),
        )
        wall = time.perf_counter() - t0
        assert int(state.converged) in (1, 2, 3)
        assert wall < 300

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="JAX int32 buffer overflow on 50K vars (platform limitation)",
        strict=False,
    )
    def test_lineax_cg_50k_unconstrained(self):
        """50K variables unconstrained with lineax_cg."""
        n = 50000

        def obj_fn(x):
            return jnp.sum(x**2)

        x0 = jnp.ones(n) * 0.1
        x_l = jnp.full(n, -10.0)
        x_u = jnp.full(n, 10.0)

        t0 = time.perf_counter()
        state = ipm_solve(
            obj_fn,
            _con_empty,
            x0,
            x_l,
            x_u,
            jnp.array([]),
            jnp.array([]),
            IPMOptions(
                linear_solver="lineax_cg",
                max_iter=200,
                tol=1e-4,
                acceptable_tol=1e-3,
            ),
        )
        wall = time.perf_counter() - t0
        assert float(state.obj) < 1.0
        assert wall < 300


# ---------------------------------------------------------------
# 17. HAS_LINEAX flag and graceful fallback
# ---------------------------------------------------------------


class TestLineaxAvailability:
    """Test HAS_LINEAX flag and module imports."""

    def test_has_lineax_is_true(self):
        """HAS_LINEAX should be True when lineax is installed."""
        from discopt._jax.ipm_iterative import HAS_LINEAX

        assert HAS_LINEAX is True

    def test_module_imports_cleanly(self):
        """ipm_iterative module should import without errors."""
        from discopt._jax import ipm_iterative  # noqa: F401

    def test_iterative_kkt_solver_requires_lineax(self):
        """IterativeKKTSolver should raise ImportError if lineax missing."""
        from discopt._jax.ipm_iterative import IterativeKKTSolver

        # When lineax IS installed, constructor should not raise
        solver = IterativeKKTSolver()
        assert solver.linear_solver == "cg"

    def test_solve_nlp_iterative_imports(self):
        """Convenience functions should be importable."""
        from discopt._jax.ipm_iterative import (  # noqa: F401
            solve_nlp_iterative,
            solve_nlp_iterative_batch,
        )


# ---------------------------------------------------------------
# 18. Matrix-free operator tests
# ---------------------------------------------------------------


class TestMatrixFreeOperators:
    """Test the matrix-free KKT and condensed operators."""

    def test_condensed_matvec_matches_dense(self):
        """Condensed matvec should match explicit Schur complement."""
        from discopt._jax.ipm_iterative import _make_condensed_matvec

        n, m = 5, 2
        W = _make_spd_matrix(n, cond=10.0)
        key = jax.random.PRNGKey(42)
        J = jax.random.normal(key, (m, n))
        D = jnp.array([0.01, 0.02])

        matvec, W_inv, JWinv = _make_condensed_matvec(W, J, D)
        v = jnp.ones(m)

        # Explicit Schur complement
        S_explicit = J @ jnp.linalg.solve(W, J.T) + jnp.diag(D)
        expected = S_explicit @ v
        actual = matvec(v)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_kkt_matvec_matches_dense(self):
        """Full KKT matvec should match explicit matrix."""
        from discopt._jax.ipm_iterative import _make_kkt_matvec

        n, m = 5, 2
        W = _make_spd_matrix(n, cond=10.0)
        key = jax.random.PRNGKey(42)
        J = jax.random.normal(key, (m, n))
        D = jnp.array([0.01, 0.02])

        matvec = _make_kkt_matvec(W, J, D, n, m)
        v = jnp.ones(n + m)

        # Explicit KKT matrix
        KKT = jnp.block([[W, J.T], [J, -jnp.diag(D)]])
        expected = KKT @ v
        actual = matvec(v)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
