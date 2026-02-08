"""
alphaBB Convex Underestimators.

Implements the alphaBB method (Adjiman, Androulakis, Floudas, 1998) for
constructing convex underestimators and concave overestimators of general
nonconvex C2 functions.

For a function f(x) on a box [lb, ub], the alphaBB underestimator is:

    L(x) = f(x) + sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

where alpha_i >= max(0, -lambda_min_i / 2) ensures L is convex. Here
lambda_min_i is the minimum eigenvalue of the Hessian restricted to the
i-th diagonal block.

The perturbation term alpha_i * (x_i - lb_i) * (ub_i - x_i) is:
  - Non-negative on [lb_i, ub_i] (zero at boundaries, positive inside)
  - Concave, so subtracting it makes the result more convex

All functions are pure JAX and compatible with jax.jit and jax.vmap.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def estimate_alpha(f, lb, ub, n_samples=100):
    """Estimate sufficient alpha values to make f convex on [lb, ub].

    Computes the Hessian of f at n_samples random points in [lb, ub],
    finds the minimum eigenvalue across all samples, and returns alpha
    values that ensure the underestimator is convex.

    Args:
        f: A scalar-valued function of a 1D array x.
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        n_samples: Number of random sample points for Hessian estimation.

    Returns:
        alpha: Array of shape (n,) with alpha_i >= max(0, -lambda_min / 2).
    """
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)
    n = lb.shape[0]

    hessian_fn = jax.hessian(f)

    key = jax.random.PRNGKey(42)
    # Generate random sample points in [lb, ub]
    random_pts = lb + (ub - lb) * jax.random.uniform(key, shape=(n_samples, n), dtype=jnp.float64)

    # Also include a grid along edges for better coverage.
    # For each dimension, sample points where one coordinate is at lb or ub.
    n_edge = min(20, n_samples)
    key2 = jax.random.PRNGKey(43)
    edge_pts = lb + (ub - lb) * jax.random.uniform(
        key2, shape=(n_edge * 2 * n, n), dtype=jnp.float64
    )
    # Snap some coordinates to boundaries for edge coverage
    for dim in range(n):
        edge_pts = edge_pts.at[dim * n_edge : (dim + 1) * n_edge, dim].set(lb[dim])
        edge_pts = edge_pts.at[(n + dim) * n_edge : (n + dim + 1) * n_edge, dim].set(ub[dim])

    samples = jnp.concatenate([random_pts, edge_pts], axis=0)

    def _min_eigenvalue(x):
        H = hessian_fn(x)
        eigvals = jnp.linalg.eigvalsh(H)
        return jnp.min(eigvals)

    # Compute minimum eigenvalue at each sample point
    min_eigs = jax.vmap(_min_eigenvalue)(samples)

    # Global minimum eigenvalue across all samples
    global_min_eig = jnp.min(min_eigs)

    # alpha = max(0, -lambda_min / 2) with a safety factor of 1.5
    # to account for Hessian variation at unsampled points.
    # Conservative overestimation is critical for guaranteed convexity.
    alpha_scalar = jnp.maximum(0.0, -global_min_eig / 2.0 * 1.5 + 1e-6)
    alpha = jnp.full(n, alpha_scalar)

    return alpha


def alphabb_underestimator(f, x, lb, ub, alpha):
    """Compute the alphaBB convex underestimator of f at x.

    L(x) = f(x) + sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

    Since (x_i - lb_i) * (ub_i - x_i) >= 0 for x_i in [lb_i, ub_i],
    the perturbation is non-negative, so L(x) >= f(x) is NOT true.
    Actually, the perturbation is SUBTRACTED to get an underestimator:

    L(x) = f(x) - sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

    This gives L(x) <= f(x) on [lb, ub], and L is convex when alpha is
    sufficiently large.

    Args:
        f: Scalar-valued function.
        x: Point at which to evaluate, shape (n,).
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        alpha: Convexification parameters, shape (n,).

    Returns:
        Scalar value of the underestimator at x.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)

    perturbation = jnp.sum(alpha * (x - lb) * (ub - x))
    return f(x) - perturbation


def alphabb_overestimator(f, x, lb, ub, alpha_neg=None):
    """Compute the alphaBB concave overestimator of f at x.

    U(x) = -L_{-f}(x) where L_{-f} is the underestimator of -f.

    U(x) = f(x) + sum_i alpha_neg_i * (x_i - lb_i) * (ub_i - x_i)

    This gives U(x) >= f(x) on [lb, ub], and U is concave.

    Args:
        f: Scalar-valued function.
        x: Point at which to evaluate, shape (n,).
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        alpha_neg: Convexification parameters for -f. If None, will be
            estimated (expensive). Shape (n,).

    Returns:
        Scalar value of the overestimator at x.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)

    if alpha_neg is None:

        def neg_f(z):
            return -f(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub)

    alpha_neg = jnp.asarray(alpha_neg, dtype=jnp.float64)
    perturbation = jnp.sum(alpha_neg * (x - lb) * (ub - x))
    return f(x) + perturbation


def make_alphabb_relaxation(f, lb, ub, n_samples=100):
    """Create alphaBB underestimator and overestimator functions for f.

    Pre-computes alpha values for both f and -f, then returns efficient
    jit-compatible functions.

    Args:
        f: Scalar-valued function of a 1D array.
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        n_samples: Number of samples for Hessian estimation.

    Returns:
        (under_fn, over_fn, alpha, alpha_neg) where:
          - under_fn(x) returns the convex underestimator at x
          - over_fn(x) returns the concave overestimator at x
          - alpha: convexification params for f
          - alpha_neg: convexification params for -f
    """
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)

    alpha = estimate_alpha(f, lb, ub, n_samples)

    def neg_f(z):
        return -f(z)

    alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples)

    @functools.wraps(f)
    def under_fn(x):
        return alphabb_underestimator(f, x, lb, ub, alpha)

    @functools.wraps(f)
    def over_fn(x):
        return alphabb_overestimator(f, x, lb, ub, alpha_neg)

    return under_fn, over_fn, alpha, alpha_neg
