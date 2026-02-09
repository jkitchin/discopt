"""
Tests for sparsity detection, graph coloring, and sparse Jacobian evaluation.

Test categories:
1. Sparsity detection correctness — detected pattern matches actual Jacobian nonzeros
2. Graph coloring validity — no two adjacent columns share a color
3. Seed matrix structure — matches coloring
4. Sparse vs dense Jacobian — elementwise comparison within atol=1e-10
5. should_use_sparse threshold logic
6. Performance — sparse faster than dense for large sparse problems
7. All correctness problems — sparse Jacobian matches dense
8. Auto-gate — small problems routed to dense
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.sparse_jacobian import make_sparse_jac_fn, sparse_jacobian_jvp
from discopt._jax.sparsity import (
    SparsityPattern,
    compute_coloring,
    detect_and_color,
    detect_sparsity_dag,
    make_seed_matrix,
    should_use_sparse,
)

# ──────────────────────────────────────────────────────────
# Helper: build models with known sparsity structure
# ──────────────────────────────────────────────────────────


def _make_diagonal_model(n: int):
    """Model where constraint i depends only on variable i.

    Jacobian is diagonal (identity-like), density = 1/n.
    """
    m = dm.Model("diagonal")
    x = m.continuous("x", shape=(n,), lb=-10, ub=10)
    m.minimize(dm.sum(x))
    for i in range(n):
        m.subject_to(x[i] <= 5.0)
    return m


def _make_tridiagonal_model(n: int):
    """Model where constraint i depends on x[i-1], x[i], x[i+1].

    Jacobian is tridiagonal, density ~ 3/n.
    """
    m = dm.Model("tridiag")
    x = m.continuous("x", shape=(n,), lb=-10, ub=10)
    m.minimize(dm.sum(x))
    for i in range(n):
        terms = x[i]
        if i > 0:
            terms = terms + x[i - 1]
        if i < n - 1:
            terms = terms + x[i + 1]
        m.subject_to(terms <= 10.0)
    return m


def _make_dense_model(n: int):
    """Model where every constraint depends on all variables.

    Jacobian is dense, density = 1.0.
    """
    m = dm.Model("dense")
    x = m.continuous("x", shape=(n,), lb=-10, ub=10)
    m.minimize(dm.sum(x))
    for i in range(n):
        m.subject_to(dm.sum(x) + x[i] <= 100.0)
    return m


def _make_block_diagonal_model(n_blocks: int, block_size: int):
    """Model with block-diagonal Jacobian structure.

    n_blocks * block_size variables, block_size constraints per block.
    Each block's constraints reference only that block's variables.
    """
    n = n_blocks * block_size
    m = dm.Model("block_diag")
    x = m.continuous("x", shape=(n,), lb=-10, ub=10)
    m.minimize(dm.sum(x))
    for b in range(n_blocks):
        start = b * block_size
        for i in range(block_size):
            idx = start + i
            terms = x[idx]
            for j in range(block_size):
                if j != i:
                    terms = terms + 0.1 * x[start + j]
            m.subject_to(terms <= 10.0)
    return m


def _make_nonlinear_sparse_model():
    """Model with nonlinear terms and sparse Jacobian."""
    m = dm.Model("nonlinear_sparse")
    x = m.continuous("x", shape=(5,), lb=0.1, ub=5.0)
    m.minimize(x[0] ** 2 + x[2] ** 2)
    m.subject_to(x[0] * x[1] <= 4.0)  # depends on x[0], x[1]
    m.subject_to(dm.exp(x[2]) <= 10.0)  # depends on x[2]
    m.subject_to(x[3] + x[4] <= 8.0)  # depends on x[3], x[4]
    return m


# ──────────────────────────────────────────────────────────
# 1. Sparsity detection correctness
# ──────────────────────────────────────────────────────────


class TestSparsityDetection:
    def test_diagonal_jacobian_pattern(self):
        """Diagonal model should have identity-like Jacobian pattern."""
        model = _make_diagonal_model(10)
        pattern = detect_sparsity_dag(model)

        assert pattern.n_vars == 10
        assert pattern.n_cons == 10
        assert pattern.jacobian_nnz == 10

        # Each row should have exactly one nonzero
        jac = pattern.jacobian_sparsity.toarray()
        for i in range(10):
            assert bool(jac[i, i])
            row_nnz = np.sum(jac[i, :])
            assert row_nnz == 1

    def test_tridiagonal_jacobian_pattern(self):
        """Tridiagonal model should have 3 nonzeros per row (except boundary)."""
        model = _make_tridiagonal_model(10)
        pattern = detect_sparsity_dag(model)

        assert pattern.n_vars == 10
        assert pattern.n_cons == 10

        jac = pattern.jacobian_sparsity.toarray()
        # Interior rows should have 3 nonzeros
        for i in range(1, 9):
            row_nnz = int(np.sum(jac[i, :]))
            assert row_nnz == 3, f"Row {i} has {row_nnz} nonzeros, expected 3"
            assert bool(jac[i, i - 1])
            assert bool(jac[i, i])
            assert bool(jac[i, i + 1])

    def test_dense_jacobian_pattern(self):
        """Dense model should have all-ones Jacobian pattern."""
        model = _make_dense_model(5)
        pattern = detect_sparsity_dag(model)

        assert pattern.n_vars == 5
        assert pattern.n_cons == 5
        # Every constraint references all variables through dm.sum(x)
        assert pattern.jacobian_nnz == 25

    def test_block_diagonal_pattern(self):
        """Block-diagonal model should have isolated blocks."""
        model = _make_block_diagonal_model(3, 4)
        pattern = detect_sparsity_dag(model)

        assert pattern.n_vars == 12
        assert pattern.n_cons == 12

        jac = pattern.jacobian_sparsity.toarray()
        # Constraints 0-3 should reference variables 0-3 only
        for i in range(4):
            for j in range(12):
                if j < 4:
                    assert bool(jac[i, j])
                else:
                    assert not bool(jac[i, j]), f"Row {i}, col {j} should be zero"

    def test_nonlinear_sparse_jacobian(self):
        """Nonlinear sparse model should detect correct incidence."""
        model = _make_nonlinear_sparse_model()
        pattern = detect_sparsity_dag(model)

        assert pattern.n_vars == 5
        assert pattern.n_cons == 3

        jac = pattern.jacobian_sparsity.toarray()
        # Constraint 0: x[0]*x[1] => columns 0, 1
        assert bool(jac[0, 0])
        assert bool(jac[0, 1])
        assert not bool(jac[0, 2])

        # Constraint 1: exp(x[2]) => column 2 only
        assert bool(jac[1, 2])
        assert not bool(jac[1, 0])

        # Constraint 2: x[3] + x[4] => columns 3, 4
        assert bool(jac[2, 3])
        assert bool(jac[2, 4])
        assert not bool(jac[2, 0])

    def test_nonlinear_hessian_pairs(self):
        """Nonlinear sparse model should detect Hessian interactions."""
        model = _make_nonlinear_sparse_model()
        pattern = detect_sparsity_dag(model)

        hess = pattern.hessian_sparsity.toarray()
        # x[0]*x[1] creates bilinear term => H[0,1] and H[1,0]
        assert bool(hess[0, 1])
        assert bool(hess[1, 0])
        # x[0]**2 in objective => H[0,0]
        assert bool(hess[0, 0])
        # x[2]**2 in objective => H[2,2]
        assert bool(hess[2, 2])

    def test_density_calculation(self):
        """Density should be computed correctly."""
        model = _make_diagonal_model(10)
        pattern = detect_sparsity_dag(model)
        assert abs(pattern.jacobian_density - 0.1) < 1e-10

        model = _make_dense_model(5)
        pattern = detect_sparsity_dag(model)
        assert abs(pattern.jacobian_density - 1.0) < 1e-10

    def test_empty_model(self):
        """Model with no constraints should have empty pattern."""
        m = dm.Model("empty")
        x = m.continuous("x", shape=(3,), lb=0, ub=1)
        m.minimize(dm.sum(x))
        pattern = detect_sparsity_dag(m)
        assert pattern.n_cons == 0
        assert pattern.jacobian_nnz == 0


# ──────────────────────────────────────────────────────────
# 2. Graph coloring validity
# ──────────────────────────────────────────────────────────


class TestGraphColoring:
    def test_diagonal_coloring(self):
        """Diagonal Jacobian needs only 1 color (no column conflicts)."""
        model = _make_diagonal_model(10)
        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)

        assert n_colors == 1
        assert np.all(colors == 0)

    def test_tridiagonal_coloring(self):
        """Tridiagonal Jacobian needs few colors (much less than n)."""
        model = _make_tridiagonal_model(20)
        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)

        # Column intersection graph for tridiagonal J has bandwidth ~5,
        # so chromatic number is bounded. Much less than n=20.
        assert n_colors <= 6

    def test_coloring_validity(self):
        """No two adjacent columns should share a color."""
        model = _make_tridiagonal_model(20)
        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)

        jac = pattern.jacobian_sparsity
        jtj = (jac.T @ jac).tocsr()
        rows, cols = jtj.nonzero()

        for r, c in zip(rows, cols):
            if r != c:
                assert colors[r] != colors[c], (
                    f"Adjacent columns {r} and {c} share color {colors[r]}"
                )

    def test_block_diagonal_coloring(self):
        """Block-diagonal: colors within each block, not across blocks."""
        model = _make_block_diagonal_model(5, 3)
        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)

        # Within-block columns are all adjacent (dense blocks),
        # so need block_size colors. Blocks don't interfere.
        assert n_colors <= 3  # block_size = 3

    def test_dense_coloring(self):
        """Dense Jacobian needs n colors (all columns adjacent)."""
        model = _make_dense_model(5)
        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)

        assert n_colors == 5

    def test_empty_coloring(self):
        """Empty model should produce empty coloring."""
        m = dm.Model("empty")
        x = m.continuous("x", shape=(3,), lb=0, ub=1)
        m.minimize(dm.sum(x))
        pattern = detect_sparsity_dag(m)
        colors, n_colors = compute_coloring(pattern)

        assert len(colors) == 3
        # With no constraints, J^T @ J is zero => 1 color suffices
        assert n_colors <= 1


# ──────────────────────────────────────────────────────────
# 3. Seed matrix structure
# ──────────────────────────────────────────────────────────


class TestSeedMatrix:
    def test_seed_shape(self):
        """Seed matrix should have shape (n, n_colors)."""
        colors = np.array([0, 1, 0, 2, 1])
        seed = make_seed_matrix(colors, 3, 5)
        assert seed.shape == (5, 3)

    def test_seed_matches_coloring(self):
        """Each row should have exactly one 1.0, at the column matching its color."""
        colors = np.array([0, 1, 0, 2, 1])
        seed = make_seed_matrix(colors, 3, 5)

        for j in range(5):
            assert seed[j, colors[j]] == 1.0
            row_nnz = np.sum(seed[j, :] != 0)
            assert row_nnz == 1

    def test_seed_column_sums(self):
        """Each seed column should have entries = number of variables with that color."""
        colors = np.array([0, 1, 0, 2, 1, 0])
        seed = make_seed_matrix(colors, 3, 6)

        col_sums = np.sum(seed, axis=0)
        assert col_sums[0] == 3  # vars 0, 2, 5
        assert col_sums[1] == 2  # vars 1, 4
        assert col_sums[2] == 1  # var 3


# ──────────────────────────────────────────────────────────
# 4. Sparse vs dense Jacobian comparison
# ──────────────────────────────────────────────────────────


class TestSparseVsDenseJacobian:
    def _compare_jacobians(self, model, atol=1e-10):
        """Compare sparse and dense Jacobian for a model."""
        evaluator = NLPEvaluator(model)
        n = evaluator.n_variables

        # Get dense Jacobian
        x = np.random.RandomState(42).uniform(0.1, 5.0, size=n)
        dense_jac = evaluator.evaluate_jacobian(x)

        # Get sparsity info
        pattern = detect_sparsity_dag(model)
        if pattern.n_cons == 0:
            return

        colors, n_colors = compute_coloring(pattern)
        seed = make_seed_matrix(colors, n_colors, n)

        # Compute sparse Jacobian via JVP
        sparse_jac = sparse_jacobian_jvp(
            evaluator._cons_fn, jnp.array(x, dtype=jnp.float64), seed, pattern, colors
        )
        sparse_dense = sparse_jac.toarray()

        np.testing.assert_allclose(sparse_dense, dense_jac, atol=atol)

    def test_diagonal_model(self):
        self._compare_jacobians(_make_diagonal_model(10))

    def test_tridiagonal_model(self):
        self._compare_jacobians(_make_tridiagonal_model(10))

    def test_block_diagonal_model(self):
        self._compare_jacobians(_make_block_diagonal_model(3, 4))

    def test_nonlinear_sparse_model(self):
        self._compare_jacobians(_make_nonlinear_sparse_model())

    def test_dense_model(self):
        self._compare_jacobians(_make_dense_model(5))

    def test_make_sparse_jac_fn(self):
        """Test the factory function for sparse Jacobian evaluation."""
        model = _make_tridiagonal_model(10)
        evaluator = NLPEvaluator(model)

        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)
        seed = make_seed_matrix(colors, n_colors, pattern.n_vars)

        jac_fn = make_sparse_jac_fn(evaluator._cons_fn, pattern, colors, seed)

        x = np.random.RandomState(42).uniform(-5, 5, size=10)
        sparse_jac = jac_fn(x)
        dense_jac = evaluator.evaluate_jacobian(x)

        np.testing.assert_allclose(sparse_jac.toarray(), dense_jac, atol=1e-10)

    def test_evaluator_sparse_jacobian_method(self):
        """Test NLPEvaluator.evaluate_sparse_jacobian method."""
        model = _make_nonlinear_sparse_model()
        evaluator = NLPEvaluator(model)

        x = np.array([1.0, 2.0, 1.0, 3.0, 4.0])
        sparse_result = evaluator.evaluate_sparse_jacobian(x)
        dense_result = evaluator.evaluate_jacobian(x)

        # For small problems, evaluate_sparse_jacobian falls back to dense
        if isinstance(sparse_result, np.ndarray):
            np.testing.assert_allclose(sparse_result, dense_result, atol=1e-10)
        else:
            np.testing.assert_allclose(sparse_result.toarray(), dense_result, atol=1e-10)


# ──────────────────────────────────────────────────────────
# 5. should_use_sparse threshold logic
# ──────────────────────────────────────────────────────────


class TestShouldUseSparse:
    def test_small_problem_returns_false(self):
        """Problems with n < 50 should use dense."""
        model = _make_diagonal_model(10)
        pattern = detect_sparsity_dag(model)
        assert should_use_sparse(pattern, min_vars=50) is False

    def test_large_sparse_returns_true(self):
        """Large sparse problem should use sparse."""
        model = _make_diagonal_model(100)
        pattern = detect_sparsity_dag(model)
        # density = 1/100 = 0.01 < 0.15
        assert should_use_sparse(pattern, min_vars=50) is True

    def test_large_dense_returns_false(self):
        """Large dense problem should use dense."""
        model = _make_dense_model(100)
        pattern = detect_sparsity_dag(model)
        # density = 1.0 > 0.15
        assert should_use_sparse(pattern, min_vars=50) is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        model = _make_tridiagonal_model(100)
        pattern = detect_sparsity_dag(model)
        # density ~ 3/100 = 0.03

        assert should_use_sparse(pattern, density_threshold=0.05) is True
        assert should_use_sparse(pattern, density_threshold=0.01) is False

    def test_no_constraints(self):
        """Model with no constraints should return False."""
        m = dm.Model("unconstrained")
        x = m.continuous("x", shape=(100,), lb=0, ub=1)
        m.minimize(dm.sum(x))
        pattern = detect_sparsity_dag(m)
        assert should_use_sparse(pattern) is False


# ──────────────────────────────────────────────────────────
# 6. Performance: sparse vs dense
# ──────────────────────────────────────────────────────────


class TestPerformance:
    @pytest.mark.slow
    def test_sparse_faster_for_large_sparse(self):
        """Sparse Jacobian should be faster than dense for n >= 200, density <= 10%."""
        n = 200
        model = _make_diagonal_model(n)
        evaluator = NLPEvaluator(model)
        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)
        seed = make_seed_matrix(colors, n_colors, n)

        jac_fn = make_sparse_jac_fn(evaluator._cons_fn, pattern, colors, seed)

        x = np.random.RandomState(42).uniform(-5, 5, size=n)

        # Warm up
        _ = jac_fn(x)
        _ = evaluator.evaluate_jacobian(x)

        n_reps = 5

        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = jac_fn(x)
        sparse_time = (time.perf_counter() - t0) / n_reps

        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = evaluator.evaluate_jacobian(x)
        dense_time = (time.perf_counter() - t0) / n_reps

        # Sparse should be significantly faster (at least 2x for diagonal)
        # n_colors=1 vs n JVPs for dense
        assert sparse_time < dense_time, (
            f"Sparse ({sparse_time:.4f}s) not faster than dense ({dense_time:.4f}s)"
        )


# ──────────────────────────────────────────────────────────
# 7. Correctness problems: sparse matches dense
# ──────────────────────────────────────────────────────────


def _make_rosenbrock_constrained():
    m = dm.Model("rosenbrock")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)
    m.subject_to(x**2 + y**2 <= 2.0)
    return m


def _make_portfolio():
    m = dm.Model("portfolio")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    returns = np.array([0.1, 0.2, 0.15])
    m.maximize(returns[0] * x[0] + returns[1] * x[1] + returns[2] * x[2])
    m.subject_to(x[0] + x[1] + x[2] == 1.0)
    m.subject_to(x[0] * x[1] + x[1] * x[2] <= 0.5)
    return m


class TestCorrectnessProblems:
    """Verify sparse Jacobian matches dense for models from test_correctness."""

    @pytest.mark.parametrize(
        "make_model",
        [
            _make_rosenbrock_constrained,
            _make_portfolio,
            _make_nonlinear_sparse_model,
        ],
        ids=["rosenbrock", "portfolio", "nonlinear_sparse"],
    )
    def test_sparse_matches_dense(self, make_model):
        model = make_model()
        evaluator = NLPEvaluator(model)
        n = evaluator.n_variables

        if evaluator.n_constraints == 0:
            return

        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)
        seed = make_seed_matrix(colors, n_colors, n)

        rng = np.random.RandomState(42)
        lb, ub = evaluator.variable_bounds
        lb_c = np.clip(lb, -10, 10)
        ub_c = np.clip(ub, -10, 10)

        for _ in range(5):
            x = lb_c + rng.uniform(size=n) * (ub_c - lb_c)
            x = np.clip(x, 0.1, 4.9)  # avoid log/div-by-zero

            sparse_jac = sparse_jacobian_jvp(
                evaluator._cons_fn,
                jnp.array(x, dtype=jnp.float64),
                seed,
                pattern,
                colors,
            )
            dense_jac = evaluator.evaluate_jacobian(x)

            np.testing.assert_allclose(sparse_jac.toarray(), dense_jac, atol=1e-8, rtol=1e-6)


# ──────────────────────────────────────────────────────────
# 8. detect_and_color convenience function
# ──────────────────────────────────────────────────────────


class TestDetectAndColor:
    def test_small_problem_returns_none(self):
        """Small problems should return None (use dense)."""
        model = _make_diagonal_model(10)
        result = detect_and_color(model, min_vars=50)
        assert result is None

    def test_large_sparse_returns_tuple(self):
        """Large sparse problems should return full tuple."""
        model = _make_diagonal_model(100)
        result = detect_and_color(model, min_vars=50)
        assert result is not None

        pattern, colors, n_colors, seed = result
        assert isinstance(pattern, SparsityPattern)
        assert len(colors) == 100
        assert n_colors >= 1
        assert seed.shape == (100, n_colors)

    def test_large_dense_returns_none(self):
        """Large dense problems should return None."""
        model = _make_dense_model(100)
        result = detect_and_color(model, min_vars=50)
        assert result is None


# ──────────────────────────────────────────────────────────
# 9. Evaluator sparsity_pattern property
# ──────────────────────────────────────────────────────────


class TestEvaluatorSparsityProperty:
    def test_sparsity_pattern_property(self):
        """NLPEvaluator.sparsity_pattern should return a SparsityPattern."""
        model = _make_nonlinear_sparse_model()
        evaluator = NLPEvaluator(model)
        pattern = evaluator.sparsity_pattern
        assert isinstance(pattern, SparsityPattern)
        assert pattern.n_vars == 5
        assert pattern.n_cons == 3

    def test_sparsity_pattern_cached(self):
        """sparsity_pattern should be cached after first access."""
        model = _make_nonlinear_sparse_model()
        evaluator = NLPEvaluator(model)
        p1 = evaluator.sparsity_pattern
        p2 = evaluator.sparsity_pattern
        assert p1 is p2


# ──────────────────────────────────────────────────────────
# 10. Edge cases
# ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_variable_model(self):
        """Model with a single variable."""
        m = dm.Model("single")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)
        m.subject_to(x <= 5.0)

        pattern = detect_sparsity_dag(m)
        assert pattern.n_vars == 1
        assert pattern.n_cons == 1
        assert pattern.jacobian_nnz == 1

        colors, n_colors = compute_coloring(pattern)
        assert n_colors == 1

    def test_model_with_parameters(self):
        """Parameters should not appear in sparsity pattern."""
        m = dm.Model("param")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        p = m.parameter("p", value=np.array([1.0, 2.0, 3.0]))
        m.minimize(dm.sum(p * x))
        m.subject_to(x[0] + x[1] <= 5.0)
        m.subject_to(x[2] <= 3.0)

        pattern = detect_sparsity_dag(m)
        assert pattern.n_vars == 3
        assert pattern.n_cons == 2

        # Constraint 0: x[0]+x[1], constraint 1: x[2]
        jac = pattern.jacobian_sparsity.toarray()
        assert bool(jac[0, 0])
        assert bool(jac[0, 1])
        assert not bool(jac[0, 2])
        assert bool(jac[1, 2])
        assert not bool(jac[1, 0])

    def test_model_with_sum_expression(self):
        """SumExpression should be handled correctly."""
        m = dm.Model("sum_expr")
        x = m.continuous("x", shape=(5,), lb=0, ub=10)
        m.minimize(dm.sum(x))
        m.subject_to(dm.sum(x) <= 10.0)

        pattern = detect_sparsity_dag(m)
        assert pattern.n_cons == 1
        # sum(x) depends on all 5 variables
        assert pattern.jacobian_nnz == 5

    def test_multiple_random_points(self):
        """Sparse Jacobian should match dense at multiple random points."""
        model = _make_tridiagonal_model(20)
        evaluator = NLPEvaluator(model)
        n = 20

        pattern = detect_sparsity_dag(model)
        colors, n_colors = compute_coloring(pattern)
        seed = make_seed_matrix(colors, n_colors, n)

        rng = np.random.RandomState(123)
        for _ in range(10):
            x = rng.uniform(-5, 5, size=n)
            sparse_jac = sparse_jacobian_jvp(
                evaluator._cons_fn,
                jnp.array(x, dtype=jnp.float64),
                seed,
                pattern,
                colors,
            )
            dense_jac = evaluator.evaluate_jacobian(x)
            np.testing.assert_allclose(sparse_jac.toarray(), dense_jac, atol=1e-10)
