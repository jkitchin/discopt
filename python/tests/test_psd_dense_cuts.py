"""Tests for dense (k>=3) moment PSD cuts (Wave 2, W2d).

Pairwise 2x2 moment minors cannot capture multi-variable coupling: a moment
submatrix can have every 2x2 principal minor PSD while the full k x k block is
indefinite. Dense clique cuts close that gap. These tests prove dense cuts are
(a) strictly stronger than pairwise and (b) still valid (never cut a feasible
point).
"""

from __future__ import annotations

import numpy as np
from discopt._jax.psd_cuts import (
    _lifted_cliques,
    separate_psd_cuts_on_relaxation,
)

# 3 variables with all squares and all cross-products lifted.
_INFO = {
    "original": {0: 0, 1: 1, 2: 2},
    "monomial": {(0, 2): 3, (1, 2): 4, (2, 2): 5},
    "bilinear": {(0, 1): 6, (0, 2): 7, (1, 2): 8},
}
_N = 9


def _correlation_point(r: float) -> np.ndarray:
    """Point with x = 0, X_ii = 1, X_ij = r (a 3x3 correlation matrix block)."""
    z = np.zeros(_N)
    z[3] = z[4] = z[5] = 1.0
    z[6] = z[7] = z[8] = r
    return z


def test_dense_separates_where_pairwise_cannot():
    # r=-0.6: every 2x2 minor 1-r^2=0.64>0 (PSD) but the 3x3 has eig 1+2r=-0.2<0.
    z = _correlation_point(-0.6)
    pairwise = separate_psd_cuts_on_relaxation(_INFO, z, _N, max_dim=2)
    dense = separate_psd_cuts_on_relaxation(_INFO, z, _N, max_dim=3)
    assert len(pairwise) == 0
    assert len(dense) >= 1


def test_dense_cut_is_valid_for_feasible_points():
    z = _correlation_point(-0.6)
    cuts = separate_psd_cuts_on_relaxation(_INFO, z, _N, max_dim=3)
    assert cuts
    cut = cuts[0]
    rng = np.random.default_rng(0)
    worst = np.inf
    for _ in range(3000):
        x = rng.uniform(-2, 2, 3)
        true = np.zeros(_N)
        true[0:3] = x
        true[3], true[4], true[5] = x[0] ** 2, x[1] ** 2, x[2] ** 2
        true[6], true[7], true[8] = x[0] * x[1], x[0] * x[2], x[1] * x[2]
        worst = min(worst, float(cut.coeffs @ true - cut.rhs))
    assert worst >= -1e-9, f"dense cut violated a feasible point by {worst}"


def test_consistent_point_yields_no_dense_cut():
    # A genuine rank-1 moment point (X = x x^T) is PSD -> no cut.
    x = np.array([0.3, 0.7, 0.5])
    z = np.zeros(_N)
    z[0:3] = x
    z[3], z[4], z[5] = x[0] ** 2, x[1] ** 2, x[2] ** 2
    z[6], z[7], z[8] = x[0] * x[1], x[0] * x[2], x[1] * x[2]
    assert separate_psd_cuts_on_relaxation(_INFO, z, _N, max_dim=3) == []


def test_clique_detection_full_and_partial():
    # Fully lifted -> one 3-clique.
    cliques = _lifted_cliques(_INFO, max_dim=6)
    assert (0, 1, 2) in cliques

    # Drop one cross-product -> no 3-clique, only pairs survive.
    partial = {
        "original": {0: 0, 1: 1, 2: 2},
        "monomial": {(0, 2): 3, (1, 2): 4, (2, 2): 5},
        "bilinear": {(0, 1): 6, (1, 2): 8},  # (0,2) missing
    }
    cliques2 = _lifted_cliques(partial, max_dim=6)
    assert all(len(c) <= 2 for c in cliques2)
    assert (0, 1) in cliques2 and (1, 2) in cliques2
