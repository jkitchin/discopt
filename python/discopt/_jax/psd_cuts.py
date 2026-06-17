"""PSD / eigenvalue cuts for the lifted (moment) relaxation of QCQP.

Term-wise McCormick / RLT relaxations lift each product ``x_i x_j`` to an
independent variable ``X_ij`` but never enforce that the moment matrix

    M(x, X) = [[1, xᵀ], [x, X]]

is positive semidefinite — yet ``M ⪰ 0`` holds for *every* feasible point
(where ``X = x xᵀ``, so ``M = [1; x][1; x]ᵀ ⪰ 0``). That missing condition is the
dominant source of the relaxation gap on nonconvex QCQP.

This module separates it dynamically: at a relaxation point ``(x*, X*)`` where the
moment matrix has a negative eigenvalue ``λ_min`` with eigenvector ``v``, the
inequality

    vᵀ M(x, X) v ≥ 0

is **linear** in ``(x, X)`` and valid for the whole feasible region (it equals
``(v₀ + v_restᵀ x)² ≥ 0`` at any true point), while it is *violated* at ``(x*, X*)``
— so it is a sound cut that tightens the relaxation toward the SDP bound, with no
SDP solver: a single dense ``eigh`` on the (small) moment submatrix.

The separator is purely numeric and indexing-driven so it plugs into whatever
column layout the relaxation uses (original variables + lifted product columns).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from discopt._jax.cutting_planes import LinearCut

__all__ = ["psd_cut_from_submatrix", "moment_matrix"]


def moment_matrix(x_vals: np.ndarray, X_vals: np.ndarray) -> np.ndarray:
    """Assemble the (k+1)×(k+1) moment matrix ``[[1, xᵀ], [x, X]]`` (symmetrized)."""
    x = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    X = np.asarray(X_vals, dtype=np.float64)
    k = x.size
    if X.shape != (k, k):
        raise ValueError(f"X_vals must be ({k}, {k}); got {X.shape}")
    Xs = 0.5 * (X + X.T)
    M = np.empty((k + 1, k + 1), dtype=np.float64)
    M[0, 0] = 1.0
    M[0, 1:] = x
    M[1:, 0] = x
    M[1:, 1:] = Xs
    return M


def psd_cut_from_submatrix(
    x_vals: np.ndarray,
    X_vals: np.ndarray,
    orig_cols: Sequence[int],
    prod_cols: np.ndarray,
    n_total: int,
    *,
    tol: float = 1e-7,
) -> Optional[LinearCut]:
    """Separate one PSD cut from a moment submatrix, or ``None`` if PSD enough.

    Parameters
    ----------
    x_vals : (k,) array
        Values of the ``k`` original variables in this submatrix at the point.
    X_vals : (k, k) array
        Values of the lifted products ``X_ij ≈ x_i x_j`` at the point.
    orig_cols : (k,) sequence of int
        Column index in the relaxation LP of each original variable.
    prod_cols : (k, k) int array
        Column index of the lifted variable representing ``X_ij`` (symmetric; the
        same column may back both ``(i, j)`` and ``(j, i)``).
    n_total : int
        Total number of columns in the relaxation LP (length of the cut vector).
    tol : float
        Only emit a cut when ``λ_min(M) < -tol`` (the point violates PSD).

    Returns
    -------
    LinearCut or None
        A valid inequality ``coeffs · z ≥ rhs`` violated at ``(x*, X*)``. The cut
        is valid for every feasible point because ``vᵀ M v = (v₀ + v_restᵀ x)² ≥ 0``
        whenever ``X = x xᵀ`` — it never removes a feasible point.
    """
    x = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    k = x.size
    prod_cols = np.asarray(prod_cols)
    if prod_cols.shape != (k, k):
        raise ValueError(f"prod_cols must be ({k}, {k}); got {prod_cols.shape}")

    M = moment_matrix(x, X_vals)
    eigvals, eigvecs = np.linalg.eigh(M)  # ascending; symmetric
    lam = float(eigvals[0])
    if lam >= -tol:
        return None  # already PSD to tolerance: nothing to separate

    v = eigvecs[:, 0]
    v0 = float(v[0])
    vr = v[1:]

    # vᵀ M v = v0² + 2 v0 Σ_a vr_a x_a + Σ_{a,b} vr_a vr_b X_ab ≥ 0
    # → coeffs·z ≥ -v0², with coeffs on the (x, X) columns.
    coeffs = np.zeros(n_total, dtype=np.float64)
    for a in range(k):
        coeffs[int(orig_cols[a])] += 2.0 * v0 * vr[a]
    for a in range(k):
        for b in range(k):
            coeffs[int(prod_cols[a, b])] += vr[a] * vr[b]

    rhs = -(v0 * v0)
    return LinearCut(coeffs=coeffs, rhs=float(rhs), sense=">=")
