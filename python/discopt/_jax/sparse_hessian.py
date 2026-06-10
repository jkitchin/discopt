"""
Compressed Lagrangian-Hessian evaluation via colored Hessian-vector products.

The dense ``jax.hessian`` of the Lagrangian materializes a full ``n x n``
matrix every iteration; for discretized DAE / orthogonal-collocation NLPs
(``n`` in the tens of thousands, but block-banded with ``O(n)`` nonzeros)
this is ``O(n^2)`` in both XLA compile time and per-iteration work, and the
solve effectively hangs (issue #95).

This module recovers only the structural nonzeros. Two columns ``i`` and
``j`` of the symmetric Hessian that never share a nonzero row are
*structurally orthogonal*: a Hessian-vector product ``H @ s`` with a seed
``s`` that is one on a whole color class returns, in row ``i``, exactly
``H[i, j]`` (no other same-color column contributes to that row). This is the
Curtis-Powell-Reid direct (column-substitution) method. Each HVP is a
forward-over-reverse autodiff pass; a single ``vmap`` evaluates all colors in
``O(n_colors * cost of grad)`` rather than ``O(n^2)``.

**Dense-row separation.** Parameter-estimation collocation Hessians are
*arrowhead*: a handful of shared parameters couple nonlinearly to every
state, producing a few dense rows/columns. A dense column shares a row with
all others, so naive column coloring degenerates to ``n`` colors. We split
the columns into a dense set ``D`` (nnz above a ``sqrt(n)`` threshold) and a
sparse set ``S``. Each dense column gets its own seed (one HVP recovers that
whole column exactly, and by symmetry its row too); the sparse columns are
colored ignoring dense rows, which makes the block-banded part collapse to a
small constant number of colors. Total seeds ``= |D| + chromatic(S)``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax.sparsity import SparsityPattern, compute_coloring


def build_hessian_coloring(
    pattern: SparsityPattern,
    hess_rows: np.ndarray,
    hess_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build seeds and a recovery map for direct Hessian compression.

    Args:
        pattern: SparsityPattern whose ``hessian_sparsity`` is the full
            symmetric ``n x n`` boolean pattern.
        hess_rows, hess_cols: lower-triangle COO indices (``row >= col``) in
            the order the evaluator's ``hessian_structure()`` reports.

    Returns:
        ``(seed_matrix, coo_seed_idx, coo_lookup_row)`` where
        ``seed_matrix`` is ``(n, n_seeds)`` float64, and each lower-triangle
        entry ``k`` is recovered as ``W[coo_seed_idx[k], coo_lookup_row[k]]``
        from ``W = (H @ seed_matrix).T`` of shape ``(n_seeds, n)``.
    """
    n = pattern.n_vars
    hess = pattern.hessian_sparsity.tocsc()
    col_nnz = np.diff(hess.indptr)

    # A column is "dense" when its nonzero count scales with n (an arrowhead
    # parameter column), not when it is merely a wide band. sqrt(n) separates
    # the two: band columns are O(1), arrowhead columns are O(n).
    threshold = max(32, int(np.ceil(np.sqrt(n))))
    dense_mask = col_nnz >= threshold
    dense_cols = np.nonzero(dense_mask)[0]
    sparse_cols = np.nonzero(~dense_mask)[0]
    n_dense = len(dense_cols)
    n_sparse = len(sparse_cols)

    # Color the sparse columns by "share a sparse row". Restricting the
    # intersection graph to sparse rows drops the dense-row conflicts, so the
    # block-banded structure colors with a small constant number of colors.
    if n_sparse > 0:
        hcsr = pattern.hessian_sparsity.tocsr()
        sub = hcsr[sparse_cols][:, sparse_cols]
        shim = SparsityPattern(
            jacobian_sparsity=sub,
            hessian_sparsity=sub,
            n_vars=n_sparse,
            n_cons=n_sparse,
            jacobian_nnz=int(sub.nnz),
            hessian_nnz=int(sub.nnz),
        )
        local_colors, n_sparse_colors = compute_coloring(shim)
    else:
        local_colors = np.array([], dtype=np.int32)
        n_sparse_colors = 0

    n_seeds = n_sparse_colors + n_dense
    seed = np.zeros((n, n_seeds), dtype=np.float64)

    sparse_color_of = np.full(n, -1, dtype=np.intp)
    for local, j in enumerate(sparse_cols):
        c = int(local_colors[local])
        seed[j, c] = 1.0
        sparse_color_of[j] = c

    dense_seed_of = np.full(n, -1, dtype=np.intp)
    for t, d in enumerate(dense_cols):
        idx = n_sparse_colors + t
        seed[d, idx] = 1.0
        dense_seed_of[d] = idx

    # Per-entry recovery indices. For lower-triangle (i >= j):
    #   - j dense          -> read dense-column-j seed at row i
    #   - i dense (j sparse)-> read dense-column-i seed at row j (symmetry)
    #   - both sparse      -> read sparse color of j at row i
    row = np.asarray(hess_rows, dtype=np.intp)
    col = np.asarray(hess_cols, dtype=np.intp)
    col_is_dense = dense_mask[col]
    row_is_dense = dense_mask[row]
    coo_seed_idx = np.where(
        col_is_dense,
        dense_seed_of[col],
        np.where(row_is_dense, dense_seed_of[row], sparse_color_of[col]),
    ).astype(np.intp)
    coo_lookup_row = np.where(col_is_dense, row, np.where(row_is_dense, col, row)).astype(np.intp)

    return seed, coo_seed_idx, coo_lookup_row


def make_sparse_hess_values_fn(
    hvp_fn: Callable,
    seed_matrix: np.ndarray,
    coo_seed_idx: np.ndarray,
    coo_lookup_row: np.ndarray,
) -> Callable:
    """Create a function returning Lagrangian-Hessian values in COO order.

    The returned values are aligned with the lower-triangle COO order used to
    build the recovery map, so the NLP backend's Hessian callback can use them
    directly.

    Args:
        hvp_fn: Hessian-vector product ``fn(x, obj_factor, lam, params, v) ->
            (n,)``, JAX-traceable (forward-over-reverse on the Lagrangian).
        seed_matrix: ``(n, n_seeds)`` float64 seed matrix.
        coo_seed_idx: per-entry seed (color) index from
            :func:`build_hessian_coloring`.
        coo_lookup_row: per-entry row to read in the HVP result.

    Returns:
        Callable ``fn(x, obj_factor, lam, params) -> np.ndarray`` (1-D float64).
    """
    seeds_jax = jnp.asarray(seed_matrix.T, dtype=jnp.float64)  # (n_seeds, n)
    seed_idx = np.asarray(coo_seed_idx, dtype=np.intp)
    lookup_row = np.asarray(coo_lookup_row, dtype=np.intp)

    @jax.jit
    def batch_hvp(x, obj_factor, lam, params):
        def one(seed):
            return hvp_fn(x, obj_factor, lam, params, seed)

        # (n_seeds, n): row r of seed s is (H @ s)[r]
        return jax.vmap(one)(seeds_jax)

    def sparse_hess_values(x, obj_factor, lam, params) -> np.ndarray:
        w = np.asarray(batch_hvp(x, obj_factor, lam, params))  # (n_seeds, n)
        values: np.ndarray = w[seed_idx, lookup_row]
        return values.astype(np.float64, copy=False)

    return sparse_hess_values
