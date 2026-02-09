"""
Compressed forward-mode Jacobian evaluation using sparsity and graph coloring.

Uses JVPs (Jacobian-vector products) with seed vectors derived from graph
coloring to compute the full sparse Jacobian in O(p) JVP evaluations,
where p is the chromatic number of the column intersection graph. For
typical sparse problems p = 5-20, a dramatic reduction from O(n) for dense
evaluation.

The key insight: if two columns share no nonzero rows in the Jacobian, their
contributions to a JVP do not interfere, so they can share a seed color.
After computing p JVP results, we recover individual entries using the known
sparsity pattern.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from discopt._jax.sparsity import SparsityPattern


def sparse_jacobian_jvp(
    fn: Callable,
    x: jnp.ndarray,
    seed_matrix: np.ndarray,
    pattern: SparsityPattern,
    colors: np.ndarray,
) -> sp.csc_matrix:
    """Compute the sparse Jacobian via compressed forward-mode JVPs.

    Args:
        fn: Constraint function f(x) -> (m,) array, must be JAX-traceable.
        x: Point at which to evaluate, shape (n,).
        seed_matrix: (n, p) seed matrix from make_seed_matrix.
        pattern: SparsityPattern with Jacobian sparsity.
        colors: (n,) int array of column colors.

    Returns:
        Sparse Jacobian as scipy CSC matrix, shape (m, n).
    """
    n = pattern.n_vars
    m = pattern.n_cons
    n_colors = seed_matrix.shape[1]

    # Compute JVPs for each color column
    jvp_results = np.zeros((m, n_colors), dtype=np.float64)
    for c in range(n_colors):
        seed_col = jnp.array(seed_matrix[:, c], dtype=x.dtype)
        _, tangent_out = jax.jvp(fn, (x,), (seed_col,))
        jvp_results[:, c] = np.asarray(tangent_out)

    # Recover individual Jacobian entries from compressed JVP results
    jac_coo = pattern.jacobian_sparsity.tocoo()
    rows = jac_coo.row
    cols = jac_coo.col

    values = np.empty(len(rows), dtype=np.float64)
    for k in range(len(rows)):
        row_idx = rows[k]
        col_idx = cols[k]
        color = colors[col_idx]
        values[k] = jvp_results[row_idx, color]

    return sp.csc_matrix((values, (rows, cols)), shape=(m, n))


def make_sparse_jac_fn(
    con_fn: Callable,
    pattern: SparsityPattern,
    colors: np.ndarray,
    seed_matrix: np.ndarray,
) -> Callable:
    """Create a function that computes the sparse Jacobian at any point.

    The returned function accepts x (numpy or JAX array) and returns a
    scipy sparse CSC matrix. The JVP evaluations are JIT-compiled.

    Args:
        con_fn: JIT-compiled constraint function f(x) -> (m,).
        pattern: SparsityPattern with Jacobian sparsity.
        colors: (n,) int array of column colors.
        seed_matrix: (n, p) seed matrix.

    Returns:
        Callable that maps x -> scipy.sparse.csc_matrix Jacobian.
    """
    n_colors = seed_matrix.shape[1]

    # Pre-convert seed columns to JAX arrays
    seed_cols = [jnp.array(seed_matrix[:, c], dtype=jnp.float64) for c in range(n_colors)]

    # Pre-extract COO structure
    jac_coo = pattern.jacobian_sparsity.tocoo()
    coo_rows = jac_coo.row
    coo_cols = jac_coo.col
    coo_colors = colors[coo_cols]

    m = pattern.n_cons
    n = pattern.n_vars

    def sparse_jac(x: np.ndarray) -> sp.csc_matrix:
        x_jax = jnp.array(x, dtype=jnp.float64)

        # Compute JVPs for each color
        jvp_results = np.zeros((m, n_colors), dtype=np.float64)
        for c in range(n_colors):
            _, tangent_out = jax.jvp(con_fn, (x_jax,), (seed_cols[c],))
            jvp_results[:, c] = np.asarray(tangent_out)

        # Recover entries
        values = jvp_results[coo_rows, coo_colors]
        return sp.csc_matrix((values, (coo_rows, coo_cols)), shape=(m, n))

    return sparse_jac
