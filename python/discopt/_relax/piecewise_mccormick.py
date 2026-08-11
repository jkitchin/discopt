"""
Piecewise McCormick Relaxations.

Partitions variable domains into k sub-intervals and computes McCormick
envelopes on each piece, yielding tighter relaxations than standard McCormick.

For a function f(x) on [lb, ub], the domain is split into k sub-intervals
[lb_i, ub_i]. On each piece, standard McCormick envelopes are computed. The final
relaxation takes the tightest result: max of convex underestimators, min of concave
overestimators across all partitions.

Supports both uniform and adaptive partitioning. Adaptive partitioning concentrates
breakpoints where the function has higher curvature, yielding tighter relaxations.

IMPORTANT: Requires finite bounds [lb, ub]. Cannot partition infinite domains.

All functions are pure JAX and compatible with jax.jit, jax.grad, and jax.vmap.
No Python-level control flow over partitions -- uses jax.vmap/vectorized ops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from discopt._relax.mccormick import (
    _secant,
    relax_bilinear,
    relax_sin,
    relax_tan,
)

# ---------------------------------------------------------------------------
# Core: partition bounds
# ---------------------------------------------------------------------------


def _partition_bounds(lb, ub, k):
    """Create k equal sub-interval boundaries for [lb, ub].

    Returns (lbs, ubs) each of shape (k,) defining the sub-intervals.
    Uses linspace which is jit-compatible.
    """
    edges = jnp.linspace(lb, ub, k + 1)
    lbs = edges[:-1]
    ubs = edges[1:]
    return lbs, ubs


def _adaptive_partition_bounds(f, lb, ub, k):
    """Create k sub-intervals with breakpoints concentrated at high curvature.

    Evaluates |f''(x)| at equispaced sample points and distributes partition
    edges proportionally to a blend of curvature and uniform density. The blend
    prevents degenerate zero-width partitions when curvature is extremely
    concentrated.

    Args:
        f: scalar function (must be twice-differentiable via jax.grad)
        lb: lower bound
        ub: upper bound
        k: number of partitions

    Returns:
        (lbs, ubs) each of shape (k,)
    """
    n_samples = 4 * k + 1
    sample_pts = jnp.linspace(lb, ub, n_samples)

    f_double_prime = jax.grad(jax.grad(f))
    curvatures = jax.vmap(f_double_prime)(sample_pts)
    abs_curv = jnp.abs(curvatures)

    # Blend curvature density with uniform density (50/50) to prevent
    # degenerate partitions when curvature is extremely concentrated.
    uniform = jnp.ones_like(abs_curv)
    blended = 0.5 * abs_curv / jnp.maximum(jnp.max(abs_curv), 1e-15) + 0.5 * uniform

    # Build cumulative distribution
    cum = jnp.cumsum(blended)
    cum = cum / cum[-1]

    # Desired quantiles for k+1 edges
    desired = jnp.linspace(0.0, 1.0, k + 1)

    # Interpolate: find sample_pts values at these quantiles
    edges = jnp.interp(desired, cum, sample_pts)

    # Force exact endpoints
    edges = edges.at[0].set(lb)
    edges = edges.at[-1].set(ub)

    lbs = edges[:-1]
    ubs = edges[1:]
    return lbs, ubs


def _get_partition_bounds(f, lb, ub, k, adaptive):
    """Dispatch between uniform and adaptive partitioning."""
    if adaptive and f is not None:
        return _adaptive_partition_bounds(f, lb, ub, k)
    return _partition_bounds(lb, ub, k)


# ---------------------------------------------------------------------------
# Piecewise bilinear relaxation
# ---------------------------------------------------------------------------


def piecewise_mccormick_bilinear(x, y, x_lb, x_ub, y_lb, y_ub, k=8, adaptive=False):
    """Piecewise McCormick relaxation of x*y by partitioning the x domain.

    Partitions x domain [x_lb, x_ub] into k sub-intervals, computes
    standard McCormick envelopes on each piece, then takes the tightest:
      - cv = max over partitions where x is in the sub-interval
      - cc = min over partitions where x is in the sub-interval

    For sub-intervals where x does not lie, contributions are masked out.

    Args:
        x: point value for first variable
        y: point value for second variable
        x_lb: lower bound for x
        x_ub: upper bound for x
        y_lb: lower bound for y
        y_ub: upper bound for y
        k: number of partitions (default 8)
        adaptive: if True, use curvature-based partitioning on x^2

    Returns:
        (cv, cc) where cv <= x*y <= cc
    """
    if adaptive:
        part_lbs, part_ubs = _adaptive_partition_bounds(lambda t: t**2, x_lb, x_ub, k)
    else:
        part_lbs, part_ubs = _partition_bounds(x_lb, x_ub, k)

    def _envelope_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        cv_i, cc_i = relax_bilinear(x, y, p_lb, p_ub, y_lb, y_ub)
        return cv_i, cc_i

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    cvs, ccs = jax.vmap(_envelope_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)

    masked_cvs = jnp.where(in_partition, cvs, -jnp.inf)
    masked_ccs = jnp.where(in_partition, ccs, jnp.inf)

    cv = jnp.max(masked_cvs)
    cc = jnp.min(masked_ccs)

    no_match = ~jnp.any(in_partition)
    std_cv, std_cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
    cv = jnp.where(no_match, std_cv, cv)
    cc = jnp.where(no_match, std_cc, cc)

    return cv, cc


# ---------------------------------------------------------------------------
# Piecewise univariate relaxations
# ---------------------------------------------------------------------------


def _piecewise_convex_relax(f, x, lb, ub, k, adaptive=False):
    """Piecewise relaxation for a convex function f.

    For convex f: cv = f(x) on each piece, cc = secant on each piece.
    The tightest is: cv = max of f(x) [just f(x)], cc = min of secants
    over the partition containing x.
    """
    part_lbs, part_ubs = _get_partition_bounds(f, lb, ub, k, adaptive)

    cv_base = f(x)

    def _secant_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        return _secant(f, x, p_lb, p_ub)

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    ccs = jax.vmap(_secant_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)
    masked_ccs = jnp.where(in_partition, ccs, jnp.inf)
    cc = jnp.min(masked_ccs)

    no_match = ~jnp.any(in_partition)
    std_cc = _secant(f, x, lb, ub)
    cc = jnp.where(no_match, std_cc, cc)

    return cv_base, cc


def _piecewise_concave_relax(f, x, lb, ub, k, adaptive=False):
    """Piecewise relaxation for a concave function f.

    For concave f: cv = secant on each piece, cc = f(x) on each piece.
    The tightest is: cv = max of secants over the partition containing x,
    cc = f(x).
    """
    neg_f = lambda t: -f(t)  # noqa: E731
    part_lbs, part_ubs = _get_partition_bounds(neg_f, lb, ub, k, adaptive)

    cc_base = f(x)

    def _secant_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        return _secant(f, x, p_lb, p_ub)

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    cvs = jax.vmap(_secant_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)
    masked_cvs = jnp.where(in_partition, cvs, -jnp.inf)
    cv = jnp.max(masked_cvs)

    no_match = ~jnp.any(in_partition)
    std_cv = _secant(f, x, lb, ub)
    cv = jnp.where(no_match, std_cv, cv)

    return cv, cc_base


def piecewise_relax_exp(x, lb, ub, k=8, adaptive=False):
    """Piecewise McCormick relaxation of exp(x) on [lb, ub].

    exp is convex: cv = exp(x), cc = piecewise secant.
    Returns (cv, cc).
    """
    return _piecewise_convex_relax(jnp.exp, x, lb, ub, k, adaptive)


def piecewise_relax_log(x, lb, ub, k=8, adaptive=False):
    """Piecewise McCormick relaxation of log(x) on [lb, ub] (lb > 0).

    log is concave: cv = piecewise secant, cc = log(x).
    Returns (cv, cc).
    """
    return _piecewise_concave_relax(jnp.log, x, lb, ub, k, adaptive)


def piecewise_relax_sqrt(x, lb, ub, k=8, adaptive=False):
    """Piecewise McCormick relaxation of sqrt(x) on [lb, ub] (lb >= 0).

    sqrt is concave: cv = piecewise secant, cc = sqrt(x).
    Returns (cv, cc).
    """
    return _piecewise_concave_relax(jnp.sqrt, x, lb, ub, k, adaptive)


def piecewise_relax_square(x, lb, ub, k=8, adaptive=False):
    """Piecewise McCormick relaxation of x^2 on [lb, ub].

    x^2 is convex: cv = x^2, cc = piecewise secant.
    Returns (cv, cc).
    """
    return _piecewise_convex_relax(lambda t: t**2, x, lb, ub, k, adaptive)


def piecewise_relax_sin(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of sin(x) on [lb, ub].

    Partitions [lb, ub] into k sub-intervals and computes sin relaxation
    on each, taking the tightest envelope.

    Returns (cv, cc).
    """
    wide = (ub - lb) >= 2.0 * jnp.pi

    part_lbs, part_ubs = _partition_bounds(lb, ub, k)

    def _relax_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        return relax_sin(x, p_lb, p_ub)

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    cvs, ccs = jax.vmap(_relax_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)
    masked_cvs = jnp.where(in_partition, cvs, -jnp.inf)
    masked_ccs = jnp.where(in_partition, ccs, jnp.inf)

    narrow_cv = jnp.max(masked_cvs)
    narrow_cc = jnp.min(masked_ccs)

    no_match = ~jnp.any(in_partition)
    std_cv, std_cc = relax_sin(x, lb, ub)
    narrow_cv = jnp.where(no_match, std_cv, jnp.maximum(narrow_cv, std_cv))
    narrow_cc = jnp.where(no_match, std_cc, jnp.minimum(narrow_cc, std_cc))

    cv = jnp.where(wide, -1.0 * jnp.ones_like(x), narrow_cv)
    cc = jnp.where(wide, 1.0 * jnp.ones_like(x), narrow_cc)

    return cv, cc


def piecewise_relax_cos(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of cos(x) on [lb, ub].

    Uses the identity cos(x) = sin(x + pi/2).
    Returns (cv, cc).
    """
    return piecewise_relax_sin(x + jnp.pi / 2, lb + jnp.pi / 2, ub + jnp.pi / 2, k)


def piecewise_relax_tan(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of tan(x) on [lb, ub].

    Partitions [lb, ub] into k sub-intervals and computes tan relaxation
    on each piece. Requires [lb, ub] within a single period (-pi/2, pi/2) + k*pi.

    Returns (cv, cc).
    """
    part_lbs, part_ubs = _partition_bounds(lb, ub, k)

    def _relax_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        return relax_tan(x, p_lb, p_ub)

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    cvs, ccs = jax.vmap(_relax_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)
    masked_cvs = jnp.where(in_partition, cvs, -jnp.inf)
    masked_ccs = jnp.where(in_partition, ccs, jnp.inf)

    cv = jnp.max(masked_cvs)
    cc = jnp.min(masked_ccs)

    no_match = ~jnp.any(in_partition)
    std_cv, std_cc = relax_tan(x, lb, ub)
    cv = jnp.where(no_match, std_cv, jnp.maximum(cv, std_cv))
    cc = jnp.where(no_match, std_cc, jnp.minimum(cc, std_cc))

    return cv, cc
