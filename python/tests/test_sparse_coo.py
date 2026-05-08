"""
Tests for sparse COO Jacobian/Hessian protocol on all evaluator types.

Validates that:
1. jacobian_structure/hessian_structure return correct COO indices
2. evaluate_jacobian_values/evaluate_hessian_values match dense at COO positions
3. Dense fallback works for small/dense problems
4. validate_sparse_values catches mismatches
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator, validate_sparse_values

pytestmark = pytest.mark.unit

# ──────────────────────────────────────────────────────────
# Test helpers
# ───────────────���──────────────────────────────────────────


def _make_small_model():
    """Small dense model (below sparse threshold)."""
    m = dm.Model("small")
    x = m.continuous("x", shape=(3,), lb=-10, ub=10)
    m.minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    m.subject_to(x[0] + x[1] + x[2] <= 10)
    m.subject_to(x[0] * x[1] <= 5)
    return m


def _make_sparse_model(n: int = 60):
    """Sparse model with diagonal Jacobian (above threshold)."""
    m = dm.Model("sparse_diag")
    x = m.continuous("x", shape=(n,), lb=-10, ub=10)
    m.minimize(dm.sum(x**2))
    for i in range(n):
        m.subject_to(x[i] <= 5.0)
    return m


def _make_sparse_nonlinear_model(n: int = 60):
    """Sparse model with nonlinear constraints for Hessian sparsity."""
    m = dm.Model("sparse_nl")
    x = m.continuous("x", shape=(n,), lb=0.1, ub=10)
    m.minimize(dm.sum(x**2))
    for i in range(n):
        m.subject_to(x[i] ** 2 <= 25.0)
    return m


# ──────────────────────────────────────────────────────────
# NLPEvaluator tests
# ──────────────────────────────────────────────────────────


class TestNLPEvaluatorSmall:
    """Small model: sparsity pattern is still reported even though the
    problem is below the compressed-evaluation threshold. Values are
    evaluated densely and projected onto the pattern."""

    def test_has_sparse_structure_true_when_pattern_exists(self):
        """A nonempty model has a sparsity pattern available."""
        ev = NLPEvaluator(_make_small_model())
        assert ev.has_sparse_structure()

    def test_small_model_does_not_use_compressed_eval(self):
        """But the compressed-JVP evaluation path is not triggered for
        small/moderately-dense problems — the density threshold gates it."""
        ev = NLPEvaluator(_make_small_model())
        assert not ev._use_compressed_eval()

    def test_jacobian_structure_matches_pattern(self):
        """Structure is the true nonzero pattern: 5 entries (not 6).
        Constraints: x0+x1+x2<=10 (3 nnz), x0*x1<=5 (2 nnz, x2 absent)."""
        ev = NLPEvaluator(_make_small_model())
        rows, cols = ev.jacobian_structure()
        assert len(rows) == 5
        assert len(cols) == 5

    def test_hessian_structure_matches_pattern(self):
        """Lower-triangle Hessian nonzeros for
        obj=x0^2+x1^2+x2^2 with x0*x1 bilinear in a constraint:
        diagonal (0,0),(1,1),(2,2) plus bilinear (1,0)."""
        ev = NLPEvaluator(_make_small_model())
        rows, cols = ev.hessian_structure()
        assert len(rows) == 4

    def test_jacobian_values_match_dense(self):
        ev = NLPEvaluator(_make_small_model())
        x = np.array([1.0, 2.0, 3.0])
        vals = ev.evaluate_jacobian_values(x)
        jac = ev.evaluate_jacobian(x)
        rows, cols = ev.jacobian_structure()
        np.testing.assert_allclose(vals, jac[rows, cols], atol=1e-12)

    def test_hessian_values_match_dense(self):
        ev = NLPEvaluator(_make_small_model())
        x = np.array([1.0, 2.0, 3.0])
        lam = np.ones(ev.n_constraints)
        vals = ev.evaluate_hessian_values(x, 1.0, lam)
        h = ev.evaluate_lagrangian_hessian(x, 1.0, lam)
        rows, cols = ev.hessian_structure()
        np.testing.assert_allclose(vals, h[rows, cols], atol=1e-12)


class TestNLPEvaluatorSparse:
    """Sparse model: should detect and use sparse structure."""

    def test_has_sparse_structure_true(self):
        ev = NLPEvaluator(_make_sparse_model())
        assert ev.has_sparse_structure()

    def test_jacobian_structure_sparse(self):
        ev = NLPEvaluator(_make_sparse_model())
        rows, cols = ev.jacobian_structure()
        n, m = ev.n_variables, ev.n_constraints
        # Diagonal Jacobian: exactly m nonzeros (one per constraint)
        assert len(rows) == m
        assert len(cols) == m
        # Much less than dense
        assert len(rows) < m * n

    def test_hessian_structure_sparse(self):
        ev = NLPEvaluator(_make_sparse_nonlinear_model())
        rows, cols = ev.hessian_structure()
        n = ev.n_variables
        # Diagonal Hessian: n nonzeros in lower triangle
        assert len(rows) == n
        # Much less than dense lower triangle
        assert len(rows) < n * (n + 1) // 2

    def test_jacobian_values_match_dense(self):
        ev = NLPEvaluator(_make_sparse_model())
        x = np.random.RandomState(0).randn(ev.n_variables)
        vals = ev.evaluate_jacobian_values(x)
        jac = ev.evaluate_jacobian(x)
        rows, cols = ev.jacobian_structure()
        np.testing.assert_allclose(vals, jac[rows, cols], atol=1e-10)

    def test_hessian_values_match_dense(self):
        ev = NLPEvaluator(_make_sparse_nonlinear_model())
        x = np.full(ev.n_variables, 1.0)
        lam = np.ones(ev.n_constraints)
        vals = ev.evaluate_hessian_values(x, 1.0, lam)
        h = ev.evaluate_lagrangian_hessian(x, 1.0, lam)
        rows, cols = ev.hessian_structure()
        np.testing.assert_allclose(vals, h[rows, cols], atol=1e-10)


# ��────────────────────────────────────��────────────────────
# Validation utility tests
# ────────────────────────────────────────────────���─────────


class TestValidation:
    def test_validate_passes_small(self):
        ev = NLPEvaluator(_make_small_model())
        x = np.array([1.0, 2.0, 3.0])
        assert validate_sparse_values(ev, x)

    def test_validate_passes_sparse(self):
        ev = NLPEvaluator(_make_sparse_model())
        x = np.random.RandomState(1).randn(ev.n_variables)
        assert validate_sparse_values(ev, x)


# ───��──────────────────────────────��───────────────────────
# Unconstrained problem
# ─────────────────────────────��────────────────────────────


class TestUnconstrained:
    def test_no_constraints(self):
        m = dm.Model("unconstrained")
        x = m.continuous("x", shape=(3,), lb=-10, ub=10)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        ev = NLPEvaluator(m)

        rows, cols = ev.jacobian_structure()
        assert len(rows) == 0

        hrows, hcols = ev.hessian_structure()
        n = ev.n_variables
        assert len(hrows) == n * (n + 1) // 2

        x0 = np.array([1.0, 2.0, 3.0])
        vals = ev.evaluate_jacobian_values(x0)
        assert len(vals) == 0

        lam = np.array([], dtype=np.float64)
        hvals = ev.evaluate_hessian_values(x0, 1.0, lam)
        assert len(hvals) == n * (n + 1) // 2


# ───��─────────────────────────────────��────────────────────
# Structure caching
# ───────────────────────────��──────────────────────────────


class TestCaching:
    def test_structure_is_cached(self):
        ev = NLPEvaluator(_make_small_model())
        r1, c1 = ev.jacobian_structure()
        r2, c2 = ev.jacobian_structure()
        assert r1 is r2
        assert c1 is c2

    def test_hessian_structure_cached(self):
        ev = NLPEvaluator(_make_small_model())
        r1, c1 = ev.hessian_structure()
        r2, c2 = ev.hessian_structure()
        assert r1 is r2
        assert c1 is c2
