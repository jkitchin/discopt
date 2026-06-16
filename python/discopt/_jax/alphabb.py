"""
alphaBB Convex Underestimators.

Implements the alphaBB method (Adjiman, Androulakis, Floudas, 1998) for
constructing convex underestimators and concave overestimators of general
nonconvex C2 functions.

For a function f(x) on a box [lb, ub], the alphaBB underestimator is:

    L(x) = f(x) - sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

where alpha_i >= max(0, -lambda_min / 2) ensures L is convex. Here
lambda_min is the minimum eigenvalue of the Hessian of f over [lb, ub].

The perturbation term alpha_i * (x_i - lb_i) * (ub_i - x_i) is:
  - Non-negative on [lb_i, ub_i] (zero at boundaries, positive inside)
  - Concave, so subtracting it makes the result more convex

The concave overestimator is:

    U(x) = f(x) + sum_i beta_i * (x_i - lb_i) * (ub_i - x_i)

where beta_i >= max(0, lambda_max / 2) for lambda_max of the Hessian of f,
or equivalently beta = alpha for -f.

All functions are pure JAX and compatible with jax.jit and jax.vmap.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def _eigenvalue_method(H):
    """Compute minimum eigenvalue of a symmetric matrix via eigvalsh."""
    eigvals = jnp.linalg.eigvalsh(H)
    return jnp.min(eigvals)


def _gershgorin_method(H):
    """Conservative lower bound on minimum eigenvalue using Gershgorin circles.

    For each row i, eigenvalues lie in a disk centered at H[i,i] with
    radius sum_{j != i} |H[i,j]|. The minimum eigenvalue is bounded below
    by min_i(H[i,i] - R_i).
    """
    diag = jnp.diag(H)
    off_diag_sum = jnp.sum(jnp.abs(H), axis=1) - jnp.abs(diag)
    return jnp.min(diag - off_diag_sum)


def estimate_alpha(f, lb, ub, n_samples=100, method="eigenvalue"):
    """Estimate sufficient alpha values to make f convex on [lb, ub].

    Computes the Hessian of f at n_samples random points in [lb, ub],
    finds the minimum eigenvalue across all samples, and returns alpha
    values that ensure the underestimator is convex.

    Args:
        f: A scalar-valued function of a 1D array x.
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        n_samples: Number of random sample points for Hessian estimation.
        method: "eigenvalue" for exact eigvalsh, "gershgorin" for conservative bound.

    Returns:
        alpha: Array of shape (n,) with alpha_i >= max(0, -lambda_min / 2).
    """
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)
    n = lb.shape[0]

    hessian_fn = jax.hessian(f)
    eig_fn = _eigenvalue_method if method == "eigenvalue" else _gershgorin_method

    key = jax.random.PRNGKey(42)
    random_pts = lb + (ub - lb) * jax.random.uniform(key, shape=(n_samples, n), dtype=jnp.float64)

    n_edge = min(20, n_samples)
    key2 = jax.random.PRNGKey(43)
    edge_pts = lb + (ub - lb) * jax.random.uniform(
        key2, shape=(n_edge * 2 * n, n), dtype=jnp.float64
    )
    for dim in range(n):
        edge_pts = edge_pts.at[dim * n_edge : (dim + 1) * n_edge, dim].set(lb[dim])
        edge_pts = edge_pts.at[(n + dim) * n_edge : (n + dim + 1) * n_edge, dim].set(ub[dim])

    samples = jnp.concatenate([random_pts, edge_pts], axis=0)

    def _min_eigenvalue(x):
        H = hessian_fn(x)
        return eig_fn(H)

    min_eigs = jax.vmap(_min_eigenvalue)(samples)
    global_min_eig = jnp.min(min_eigs)

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


def relax_alphabb(f, x, lb, ub, alpha=None, alpha_neg=None, n_samples=100, method="eigenvalue"):
    """Compute alphaBB relaxation of f at x, returning (cv, cc).

    Matches the project convention: cv <= f(x) <= cc for all x in [lb, ub].

    Args:
        f: Scalar-valued function of a 1D array x.
        x: Evaluation point, shape (n,).
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        alpha: Pre-computed alpha for underestimator. If None, computed.
        alpha_neg: Pre-computed alpha for overestimator. If None, computed.
        n_samples: Number of Hessian samples (only used if alpha/alpha_neg is None).
        method: "eigenvalue" or "gershgorin".

    Returns:
        (cv, cc) where cv <= f(x) <= cc for all x in [lb, ub].
    """
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)

    if alpha is None:
        alpha = estimate_alpha(f, lb, ub, n_samples, method=method)
    if alpha_neg is None:

        def neg_f(z):
            return -f(z)

        alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples, method=method)

    fval = f(x)
    perturbation = jnp.sum(alpha * (x - lb) * (ub - x))
    perturbation_neg = jnp.sum(alpha_neg * (x - lb) * (ub - x))

    cv = fval - perturbation
    cc = fval + perturbation_neg
    return cv, cc


def make_alphabb_relaxation(f, lb, ub, n_samples=100, method="eigenvalue"):
    """Create alphaBB underestimator and overestimator functions for f.

    Pre-computes alpha values for both f and -f, then returns efficient
    jit-compatible functions.

    Args:
        f: Scalar-valued function of a 1D array.
        lb: Lower bounds, shape (n,).
        ub: Upper bounds, shape (n,).
        n_samples: Number of samples for Hessian estimation.
        method: "eigenvalue" or "gershgorin".

    Returns:
        (under_fn, over_fn, alpha, alpha_neg) where:
          - under_fn(x) returns the convex underestimator at x
          - over_fn(x) returns the concave overestimator at x
          - alpha: convexification params for f
          - alpha_neg: convexification params for -f
    """
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)

    alpha = estimate_alpha(f, lb, ub, n_samples, method=method)

    def neg_f(z):
        return -f(z)

    alpha_neg = estimate_alpha(neg_f, lb, ub, n_samples, method=method)

    @functools.wraps(f)
    def under_fn(x):
        return alphabb_underestimator(f, x, lb, ub, alpha)

    @functools.wraps(f)
    def over_fn(x):
        return alphabb_overestimator(f, x, lb, ub, alpha_neg)

    return under_fn, over_fn, alpha, alpha_neg


# ──────────────────────────────────────────────────────────────────────
# Rigorous alphaBB from a SOUND interval Hessian (safe for certification)
# ──────────────────────────────────────────────────────────────────────
#
# ``estimate_alpha`` above samples the Hessian at random/edge points; the
# resulting alpha is a heuristic, NOT a guaranteed lower bound on the
# minimum eigenvalue, so it must never be used to certify a global bound.
# The functions below instead derive alpha from a *sound* interval
# enclosure of the Hessian over the box (``convexity.interval_ad``) and a
# rigorous interval-Gershgorin eigenvalue bound, so the underestimator is
# a valid relaxation. They abstain (raise) whenever the interval Hessian is
# unbounded, leaving the caller's existing (McCormick) handling in place.


def rigorous_alpha(expr, model, box=None):
    """Guaranteed per-variable alphaBB parameters for ``expr`` over its box.

    Uses a sound interval enclosure of the Hessian and a per-row interval
    Gershgorin bound. For row ``i``,

        lambda_min >= H[i,i].lo - sum_{j != i} max(|H[i,j].lo|, |H[i,j].hi|)

    is a valid lower bound on the smallest eigenvalue contribution, giving

        alpha_i = max(0, -0.5 * gershgorin_lo_i).

    Variables absent from ``expr`` (or appearing linearly) have a
    zero Hessian row, hence ``alpha_i = 0`` — the perturbation is applied
    only to the nonlinear variables, keeping the relaxation as tight as the
    diagonal-dominance bound allows.

    Args:
        expr: Scalar :class:`~discopt.modeling.core.Expression`.
        model: Model defining the flat variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared bounds.

    Returns:
        ``np.ndarray`` of shape ``(n,)``. Entries are ``+inf`` wherever the
        interval Hessian abstained (unbounded), signalling that no useful
        alphaBB relaxation exists for this box.
    """
    import numpy as np

    from discopt._jax.convexity.interval_ad import interval_hessian

    iad = interval_hessian(expr, model, box)
    h_lo = np.asarray(iad.hess.lo, dtype=float)
    h_hi = np.asarray(iad.hess.hi, dtype=float)
    abs_max = np.maximum(np.abs(h_lo), np.abs(h_hi))
    # Per-row off-diagonal radius = sum of |.| over the row minus the diagonal.
    row_radius = abs_max.sum(axis=1) - np.abs(np.diag(abs_max))
    with np.errstate(invalid="ignore"):
        gershgorin_lo = np.diag(h_lo) - row_radius
        alpha = np.maximum(0.0, -0.5 * gershgorin_lo)
    # NaN arises from inf - inf at abstaining nodes; treat as unbounded.
    alpha = np.where(np.isnan(alpha), np.inf, alpha)
    return alpha


def compile_alphabb_relaxation(expr, model):
    """Compile a rigorous alphaBB relaxation node for ``expr``.

    Returns ``fn(x_cv, x_cc, lb, ub) -> (cv, cc)`` matching the McCormick
    relaxation-compiler node contract, with ``cv`` a convex underestimator
    and ``cc`` a concave overestimator of ``expr`` over the per-call box.

    Raises:
        ValueError: if the sound interval Hessian is unbounded (abstains),
            so the caller can fall back to its existing handling.
    """
    import numpy as np

    from discopt._jax.dag_compiler import compile_expression_params

    alpha = rigorous_alpha(expr, model)
    if not np.all(np.isfinite(alpha)):
        raise ValueError(
            "alphaBB: interval Hessian is unbounded for this expression/box; "
            "no valid alphaBB relaxation (falling back to McCormick)."
        )
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    f_fwd = compile_expression_params(expr, model)
    params = tuple(jnp.asarray(p.value, dtype=jnp.float64) for p in model._parameters)

    def fn(x_cv, x_cc, lb, ub, _f=f_fwd, _a=alpha_j, _p=params):
        mid = 0.5 * (x_cv + x_cc)
        fval = _f(mid, _p)
        # (mid - lb)(ub - mid) >= 0 on the box; subtract for cv, add for cc.
        pert = jnp.sum(_a * (mid - lb) * (ub - mid))
        return fval - pert, fval + pert

    return fn
