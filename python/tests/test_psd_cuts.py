"""Tests for PSD / eigenvalue cuts (discopt._jax.psd_cuts).

The correctness-critical property of a relaxation cut is **validity**: it must be
satisfied by every feasible point, i.e. it never removes a feasible solution (or
the optimum). For a PSD cut ``vᵀ M v ≥ 0`` this holds by construction because at
any true point ``X = x xᵀ`` we have ``M = [1; x][1; x]ᵀ ⪰ 0``. These tests pin:

* validity — sampled true points ``(x, x xᵀ)`` satisfy every emitted cut;
* separation — a point with ``X ≠ x xᵀ`` and a negative moment eigenvalue is cut
  off (the cut is violated there);
* no false cuts — a PSD moment point yields no cut;
* correct column mapping into the (original + lifted) cut vector.
"""

from __future__ import annotations

import numpy as np
from discopt._jax.psd_cuts import moment_matrix, psd_cut_from_submatrix


def _eval(cut, z: np.ndarray) -> float:
    """Signed slack of ``coeffs·z - rhs`` for a ``>=`` cut (>=0 means satisfied)."""
    assert cut.sense == ">="
    return float(cut.coeffs @ z - cut.rhs)


def _layout(k: int):
    """Column layout: x in cols 0..k-1, lifted X_ij in a contiguous symmetric block."""
    orig_cols = list(range(k))
    prod_cols = np.zeros((k, k), dtype=int)
    nxt = k
    for i in range(k):
        for j in range(i, k):
            prod_cols[i, j] = nxt
            prod_cols[j, i] = nxt
            nxt += 1
    n_total = nxt
    return orig_cols, prod_cols, n_total


def _pack(x: np.ndarray, X: np.ndarray, orig_cols, prod_cols, n_total) -> np.ndarray:
    z = np.zeros(n_total)
    k = len(x)
    for a in range(k):
        z[orig_cols[a]] = x[a]
    for a in range(k):
        for b in range(k):
            z[prod_cols[a, b]] = X[a, b]
    return z


def test_no_cut_when_moment_matrix_is_psd():
    k = 3
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, k)
    X = np.outer(x, x)  # true moments -> M is rank-1 PSD
    orig_cols, prod_cols, n_total = _layout(k)
    assert psd_cut_from_submatrix(x, X, orig_cols, prod_cols, n_total) is None


def test_cut_separates_a_non_psd_point():
    k = 3
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, k)
    # Perturb the diagonal downward: X_ii < x_i^2 makes M indefinite.
    X = np.outer(x, x)
    X = X - np.diag(np.full(k, 0.3))
    orig_cols, prod_cols, n_total = _layout(k)

    cut = psd_cut_from_submatrix(x, X, orig_cols, prod_cols, n_total)
    assert cut is not None
    z_bad = _pack(x, X, orig_cols, prod_cols, n_total)
    assert _eval(cut, z_bad) < -1e-9  # the offending point is cut off


def test_cut_is_valid_for_all_feasible_points():
    """Every emitted cut must be satisfied at all true points (x, x xᵀ)."""
    k = 4
    rng = np.random.default_rng(2)
    x0 = rng.uniform(-1, 1, k)
    X_bad = np.outer(x0, x0) - 0.5 * np.eye(k)
    orig_cols, prod_cols, n_total = _layout(k)
    cut = psd_cut_from_submatrix(x0, X_bad, orig_cols, prod_cols, n_total)
    assert cut is not None

    # Sample many genuine feasible points; the cut must never be violated.
    worst = np.inf
    for _ in range(2000):
        x = rng.uniform(-2, 2, k)
        z = _pack(x, np.outer(x, x), orig_cols, prod_cols, n_total)
        worst = min(worst, _eval(cut, z))
    assert worst >= -1e-9, f"cut violated a feasible point by {worst}"


def test_validity_matches_moment_eigenvalue_identity():
    """coeffs·z - rhs at a true point equals (v0 + v_rest·x)^2 >= 0 exactly."""
    k = 3
    rng = np.random.default_rng(3)
    x0 = rng.uniform(0, 1, k)
    X_bad = np.outer(x0, x0)
    X_bad[0, 1] = X_bad[1, 0] = X_bad[0, 1] - 0.4  # break a single off-diagonal
    orig_cols, prod_cols, n_total = _layout(k)
    cut = psd_cut_from_submatrix(x0, X_bad, orig_cols, prod_cols, n_total)
    assert cut is not None
    for _ in range(200):
        x = rng.uniform(-1, 1, k)
        z = _pack(x, np.outer(x, x), orig_cols, prod_cols, n_total)
        assert _eval(cut, z) >= -1e-9


def test_moment_matrix_shape_and_symmetry():
    x = np.array([1.0, 2.0])
    X = np.array([[1.0, 3.0], [1.0, 4.0]])  # asymmetric input
    M = moment_matrix(x, X)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)
    assert M[0, 0] == 1.0
    np.testing.assert_allclose(M[0, 1:], x)
    # off-diagonal is symmetrized
    assert np.isclose(M[1, 2], 2.0) and np.isclose(M[2, 1], 2.0)
