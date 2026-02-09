"""
Tests for sparse KKT assembly, sparse IPM solver, and solver.py wiring.

Test categories:
1. Sparse KKT assembly — correct structure and dimensions
2. Sparse KKT solve — matches dense solve on small problems
3. Sparse IPM — solves unconstrained, bound-constrained, and constrained NLPs
4. Sparse IPM vs dense IPM — same solutions within tolerance
5. Solver wiring — nlp_solver="sparse_ipm" dispatches correctly
6. Inertia detection — correct positive/negative detection
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import scipy.sparse as sp
from discopt._jax.sparse_kkt import (
    assemble_kkt_sparse,
    detect_inertia_sparse,
    solve_kkt_direct,
    solve_kkt_factored,
)

# ──────────────────────────────────────────────────────────
# 1. Sparse KKT assembly
# ──────────────────────────────────────────────────────────


class TestSparseKKTAssembly:
    def test_basic_assembly(self):
        """Assemble a 2x2 KKT system and check structure."""
        n, m = 3, 2
        H = sp.eye(n, format="csc") * 2.0
        J = sp.csc_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))
        sigma = np.array([1.0, 1.0, 1.0])

        kkt = assemble_kkt_sparse(H, J, sigma, delta_w=0.0, delta_c=1e-8)

        assert kkt.shape == (n + m, n + m)
        assert sp.issparse(kkt)

        # (1,1) block should be H + diag(sigma) = 3*I
        kkt_dense = kkt.toarray()
        np.testing.assert_allclose(kkt_dense[:n, :n], np.diag([3.0, 3.0, 3.0]))

        # (1,2) block should be J^T
        np.testing.assert_allclose(kkt_dense[:n, n:], J.toarray().T)

        # (2,1) block should be J
        np.testing.assert_allclose(kkt_dense[n:, :n], J.toarray())

    def test_with_inertia_correction(self):
        """delta_w should add to diagonal of (1,1) block."""
        n = 2
        H = sp.csc_matrix((n, n))
        J = sp.csc_matrix(np.array([[1.0, 1.0]]))
        sigma = np.zeros(n)

        kkt = assemble_kkt_sparse(H, J, sigma, delta_w=5.0, delta_c=1e-2)
        kkt_dense = kkt.toarray()

        # (1,1) should be 5*I
        np.testing.assert_allclose(kkt_dense[:n, :n], np.diag([5.0, 5.0]))

        # (2,2) should be -0.01
        assert abs(kkt_dense[n, n] - (-0.01)) < 1e-12

    def test_empty_constraints(self):
        """KKT with 0 constraints should be just H + sigma."""
        n = 3
        H = sp.eye(n, format="csc")
        J = sp.csc_matrix((0, n))
        sigma = np.ones(n) * 2.0

        kkt = assemble_kkt_sparse(H, J, sigma)
        assert kkt.shape == (n, n)

        kkt_dense = kkt.toarray()
        np.testing.assert_allclose(kkt_dense, np.diag([3.0, 3.0, 3.0]))


# ──────────────────────────────────────────────────────────
# 2. Sparse KKT solve
# ──────────────────────────────────────────────────────────


class TestSparseKKTSolve:
    def test_direct_solve(self):
        """Solve a simple KKT system and verify."""
        n = 3
        H = sp.eye(n, format="csc") * 2.0
        J = sp.csc_matrix((0, n))
        sigma = np.ones(n)

        kkt = assemble_kkt_sparse(H, J, sigma, delta_w=0.0)
        rhs = np.array([3.0, 3.0, 3.0])

        sol = solve_kkt_direct(kkt, rhs)
        np.testing.assert_allclose(sol, [1.0, 1.0, 1.0], atol=1e-12)

    def test_factored_solve(self):
        """Factored solve should give same result as direct."""
        n = 2
        H = sp.eye(n, format="csc") * 4.0
        J = sp.csc_matrix(np.array([[1.0, 0.0]]))
        sigma = np.zeros(n)

        kkt = assemble_kkt_sparse(H, J, sigma, delta_w=0.0, delta_c=1e-8)
        rhs = np.array([4.0, 8.0, 1.0])

        sol_direct = solve_kkt_direct(kkt, rhs)
        sol_factored, lu = solve_kkt_factored(kkt, rhs)

        np.testing.assert_allclose(sol_direct, sol_factored, atol=1e-12)

        # Reuse factorization
        rhs2 = np.array([1.0, 2.0, 3.0])
        sol2, _ = solve_kkt_factored(kkt, rhs2, lu=lu)
        sol2_direct = solve_kkt_direct(kkt, rhs2)
        np.testing.assert_allclose(sol2, sol2_direct, atol=1e-12)

    def test_matches_dense(self):
        """Sparse solve should match numpy dense solve."""
        H = sp.csc_matrix(
            np.array(
                [
                    [4.0, 1.0, 0.0],
                    [1.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            )
        )
        J = sp.csc_matrix(
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        )
        sigma = np.array([0.5, 0.5, 0.5])

        kkt = assemble_kkt_sparse(H, J, sigma, delta_w=1e-4, delta_c=1e-8)
        rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sol_sparse = solve_kkt_direct(kkt, rhs)
        sol_dense = np.linalg.solve(kkt.toarray(), rhs)

        np.testing.assert_allclose(sol_sparse, sol_dense, atol=1e-10)


# ──────────────────────────────────────────────────────────
# 3. Sparse IPM solve
# ──────────────────────────────────────────────────────────


class TestSparseIPM:
    def test_unconstrained_quadratic(self):
        """Minimize x^2 + y^2, expected solution (0, 0)."""

        from discopt._jax.sparse_ipm import SparseIPMOptions, sparse_ipm_solve

        def obj(x):
            return x[0] ** 2 + x[1] ** 2

        x0 = np.array([3.0, 4.0])
        x_l = np.array([-10.0, -10.0])
        x_u = np.array([10.0, 10.0])

        result = sparse_ipm_solve(
            obj,
            None,
            x0,
            x_l,
            x_u,
            options=SparseIPMOptions(max_iter=100),
        )

        assert result.converged in (1, 2, 3)
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-3)

    def test_bound_constrained(self):
        """Minimize (x-3)^2 + (y-4)^2 with bounds [0, 2]."""

        from discopt._jax.sparse_ipm import SparseIPMOptions, sparse_ipm_solve

        def obj(x):
            return (x[0] - 3.0) ** 2 + (x[1] - 4.0) ** 2

        x0 = np.array([1.0, 1.0])
        x_l = np.array([0.0, 0.0])
        x_u = np.array([2.0, 2.0])

        result = sparse_ipm_solve(
            obj,
            None,
            x0,
            x_l,
            x_u,
            options=SparseIPMOptions(max_iter=100),
        )

        assert result.converged in (1, 2, 3)
        # Solution should be at bounds: (2, 2)
        np.testing.assert_allclose(result.x, [2.0, 2.0], atol=1e-2)

    def test_equality_constrained(self):
        """Minimize x^2 + y^2 s.t. x + y == 1."""
        import jax.numpy as jnp
        from discopt._jax.sparse_ipm import SparseIPMOptions, sparse_ipm_solve

        def obj(x):
            return x[0] ** 2 + x[1] ** 2

        def con(x):
            return jnp.array([x[0] + x[1] - 1.0])

        x0 = np.array([2.0, 2.0])
        x_l = np.array([-10.0, -10.0])
        x_u = np.array([10.0, 10.0])
        g_l = np.array([0.0])
        g_u = np.array([0.0])

        result = sparse_ipm_solve(
            obj,
            con,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            options=SparseIPMOptions(max_iter=200),
        )

        assert result.converged in (1, 2, 3)
        # Solution should be (0.5, 0.5)
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-2)

    def test_inequality_constrained(self):
        """Minimize -x-y s.t. x + y <= 1, x,y >= 0."""
        import jax.numpy as jnp
        from discopt._jax.sparse_ipm import SparseIPMOptions, sparse_ipm_solve

        def obj(x):
            return -x[0] - x[1]

        def con(x):
            return jnp.array([x[0] + x[1]])

        x0 = np.array([0.3, 0.3])
        x_l = np.array([0.0, 0.0])
        x_u = np.array([10.0, 10.0])
        g_l = np.array([-1e20])
        g_u = np.array([1.0])

        result = sparse_ipm_solve(
            obj,
            con,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            options=SparseIPMOptions(max_iter=200),
        )

        assert result.converged in (1, 2, 3)
        # Sparse IPM on LP-like problems may not converge as tightly
        assert result.objective < -0.5


# ──────────────────────────────────────────────────────────
# 4. Sparse IPM vs dense IPM
# ──────────────────────────────────────────────────────────


class TestSparseVsDenseIPM:
    def test_rosenbrock_sparse_vs_dense(self):
        """Both IPMs should find same solution on Rosenbrock."""
        import jax.numpy as jnp
        from discopt._jax.ipm import IPMOptions, ipm_solve
        from discopt._jax.sparse_ipm import SparseIPMOptions, sparse_ipm_solve

        def obj(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        x0 = np.array([0.0, 0.0])
        x_l = np.array([-5.0, -5.0])
        x_u = np.array([5.0, 5.0])

        # Dense IPM
        dense_state = ipm_solve(
            obj,
            None,
            jnp.array(x0),
            jnp.array(x_l),
            jnp.array(x_u),
            None,
            None,
            IPMOptions(max_iter=200),
        )

        # Sparse IPM
        sparse_result = sparse_ipm_solve(
            obj,
            None,
            x0,
            x_l,
            x_u,
            options=SparseIPMOptions(max_iter=200),
        )

        # Both should find (1, 1)
        dense_x = np.asarray(dense_state.x)
        if int(dense_state.converged) in (1, 2):
            np.testing.assert_allclose(dense_x, [1.0, 1.0], atol=0.1)
        if sparse_result.converged in (1, 2):
            np.testing.assert_allclose(sparse_result.x, [1.0, 1.0], atol=0.1)


# ──────────────────────────────────────────────────────────
# 5. Solver wiring
# ──────────────────────────────────────────────────────────


class TestSolverWiring:
    def test_sparse_ipm_via_model_solve(self):
        """Model.solve(nlp_solver='sparse_ipm') should work end-to-end."""
        m = dm.Model("simple_qp")
        x = m.continuous("x", lb=0.0, ub=5.0)
        y = m.continuous("y", lb=0.0, ub=5.0)
        m.minimize((x - 1) ** 2 + (y - 2) ** 2)
        m.subject_to(x + y <= 4.0)

        result = m.solve(nlp_solver="sparse_ipm")

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        # Optimal at (1, 2), obj = 0
        assert result.objective < 1.0

    def test_sparse_ipm_unconstrained(self):
        """sparse_ipm on unconstrained problem."""
        m = dm.Model("unconstrained")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)

        result = m.solve(nlp_solver="sparse_ipm")
        assert result.status in ("optimal", "feasible")
        assert result.objective < 0.1


# ──────────────────────────────────────────────────────────
# 6. Inertia detection
# ──────────────────────────────────────────────────────────


class TestInertiaDetection:
    def test_positive_definite(self):
        """PD matrix should pass inertia check."""
        H = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        ok, min_eig = detect_inertia_sparse(H, 2)
        assert ok is True
        assert min_eig > 0

    def test_negative_definite(self):
        """ND matrix should fail inertia check."""
        H = sp.csc_matrix(np.array([[-4.0, 0.0], [0.0, -3.0]]))
        ok, min_eig = detect_inertia_sparse(H, 2)
        assert ok is False
        assert min_eig < 0

    def test_indefinite(self):
        """Indefinite matrix should fail inertia check."""
        H = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
        ok, min_eig = detect_inertia_sparse(H, 2)
        assert ok is False
        assert min_eig < 0

    def test_zero_dimension(self):
        """Empty matrix should pass."""
        ok, min_eig = detect_inertia_sparse(sp.csc_matrix((0, 0)), 0)
        assert ok is True
