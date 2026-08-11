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

from discopt._relax.sparsity import SparsityPattern


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
    scipy sparse CSC matrix. The color JVPs are fused into a single
    jit-compiled vmapped call, so each jacobian evaluation dispatches
    exactly once into XLA regardless of chromatic number.

    Args:
        con_fn: JIT-compiled constraint function f(x) -> (m,).
        pattern: SparsityPattern with Jacobian sparsity.
        colors: (n,) int array of column colors.
        seed_matrix: (n, p) seed matrix.

    Returns:
        Callable that maps x -> scipy.sparse.csc_matrix Jacobian.
    """
    # Pre-convert the full seed matrix once and transpose to (n_colors, n)
    # so vmap walks the leading axis.
    seeds_jax = jnp.asarray(seed_matrix.T, dtype=jnp.float64)

    jac_coo = pattern.jacobian_sparsity.tocoo()
    coo_rows = jac_coo.row
    coo_cols = jac_coo.col
    coo_colors = colors[coo_cols]

    m = pattern.n_cons
    n = pattern.n_vars

    @jax.jit
    def batch_jvp(x):
        def one_jvp(seed):
            _, t = jax.jvp(con_fn, (x,), (seed,))
            return t

        # (n_colors, m) — one JVP per color, all fused into one XLA kernel
        return jax.vmap(one_jvp)(seeds_jax)

    def sparse_jac(x: np.ndarray) -> sp.csc_matrix:
        jvp_results = np.asarray(batch_jvp(x))  # (n_colors, m)
        values = jvp_results[coo_colors, coo_rows]
        return sp.csc_matrix((values, (coo_rows, coo_cols)), shape=(m, n))

    return sparse_jac


def make_sparse_jac_values_fn(
    con_fn_xp: Callable,
    pattern: SparsityPattern,
    colors: np.ndarray,
    seed_matrix: np.ndarray,
) -> Callable:
    """Create a function returning Jacobian values in COO order.

    Unlike :func:`make_sparse_jac_fn`, this skips the scipy.sparse wrapper
    entirely and returns a flat ``float64`` array aligned with
    ``pattern.jacobian_sparsity.tocoo()`` ordering. This is the hot path
    cyipopt's jacobian callback cares about — it already has the structure.

    The constraint function must take ``(x, params)`` so the NLP
    evaluator can thread ``Parameter`` values through without retracing.

    Returns:
        Callable ``fn(x, params) -> np.ndarray`` (1-D, ``float64``).
    """
    seeds_jax = jnp.asarray(seed_matrix.T, dtype=jnp.float64)

    jac_coo = pattern.jacobian_sparsity.tocoo()
    coo_rows = jac_coo.row.astype(np.intp)
    coo_cols = jac_coo.col.astype(np.intp)
    coo_colors = colors[coo_cols].astype(np.intp)

    @jax.jit
    def batch_jvp(x, params):
        def one_jvp(seed):
            _, t = jax.jvp(lambda xp: con_fn_xp(xp, params), (x,), (seed,))
            return t

        return jax.vmap(one_jvp)(seeds_jax)

    def sparse_jac_values(x, params) -> np.ndarray:
        jvp_results = np.asarray(batch_jvp(x, params))  # (n_colors, m)
        values: np.ndarray = jvp_results[coo_colors, coo_rows]
        return values.astype(np.float64, copy=False)

    return sparse_jac_values
