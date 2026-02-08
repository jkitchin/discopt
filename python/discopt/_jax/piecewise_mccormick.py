"""
Piecewise McCormick Relaxations.

Partitions variable domains into k sub-intervals and computes McCormick
envelopes on each piece, yielding tighter relaxations than standard McCormick.

For a function f(x) on [lb, ub], the domain is split into k equal sub-intervals
[lb_i, ub_i]. On each piece, standard McCormick envelopes are computed. The final
relaxation takes the tightest result: max of convex underestimators, min of concave
overestimators across all partitions.

IMPORTANT: Requires finite bounds [lb, ub]. Cannot partition infinite domains.

All functions are pure JAX and compatible with jax.jit, jax.grad, and jax.vmap.
No Python-level control flow over partitions -- uses jax.vmap/vectorized ops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from discopt._jax.mccormick import (
    _secant,
    relax_bilinear,
    relax_sin,
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


# ---------------------------------------------------------------------------
# Piecewise bilinear relaxation
# ---------------------------------------------------------------------------


def piecewise_mccormick_bilinear(x, y, x_lb, x_ub, y_lb, y_ub, k=8):
    """Piecewise McCormick relaxation of x*y by partitioning the x domain.

    Partitions x domain [x_lb, x_ub] into k equal sub-intervals, computes
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

    Returns:
        (cv, cc) where cv <= x*y <= cc
    """
    part_lbs, part_ubs = _partition_bounds(x_lb, x_ub, k)

    # Compute McCormick envelopes for each partition (vectorized).
    # For each sub-interval i: compute envelope using (x, y) with
    # x bounds = [part_lbs[i], part_ubs[i]], y bounds = [y_lb, y_ub].
    def _envelope_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        cv_i, cc_i = relax_bilinear(x, y, p_lb, p_ub, y_lb, y_ub)
        return cv_i, cc_i

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)  # (k, 2)
    cvs, ccs = jax.vmap(_envelope_one)(bounds_stacked)  # each (k,)

    # Determine which partition x belongs to.
    # x is in partition i if part_lbs[i] <= x <= part_ubs[i].
    # Due to floating point, x might sit exactly on a boundary; use tolerance.
    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)

    # For partitions where x is not present, mask out their contributions.
    # For cv: we want max over valid partitions, so set invalid to -inf.
    # For cc: we want min over valid partitions, so set invalid to +inf.
    masked_cvs = jnp.where(in_partition, cvs, -jnp.inf)
    masked_ccs = jnp.where(in_partition, ccs, jnp.inf)

    cv = jnp.max(masked_cvs)
    cc = jnp.min(masked_ccs)

    # Fallback: if no partition matches (shouldn't happen), use standard.
    no_match = ~jnp.any(in_partition)
    std_cv, std_cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)
    cv = jnp.where(no_match, std_cv, cv)
    cc = jnp.where(no_match, std_cc, cc)

    return cv, cc


# ---------------------------------------------------------------------------
# Piecewise univariate relaxations
# ---------------------------------------------------------------------------


def _piecewise_convex_relax(f, x, lb, ub, k):
    """Piecewise relaxation for a convex function f.

    For convex f: cv = f(x) on each piece, cc = secant on each piece.
    The tightest is: cv = max of f(x) [just f(x)], cc = min of secants
    over the partition containing x.
    """
    part_lbs, part_ubs = _partition_bounds(lb, ub, k)

    # cv is always f(x) for a convex function, no matter the partition.
    cv_base = f(x)

    # cc = secant on the sub-interval containing x.
    def _secant_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        return _secant(f, x, p_lb, p_ub)

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    ccs = jax.vmap(_secant_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)
    masked_ccs = jnp.where(in_partition, ccs, jnp.inf)
    cc = jnp.min(masked_ccs)

    # Fallback
    no_match = ~jnp.any(in_partition)
    std_cc = _secant(f, x, lb, ub)
    cc = jnp.where(no_match, std_cc, cc)

    return cv_base, cc


def _piecewise_concave_relax(f, x, lb, ub, k):
    """Piecewise relaxation for a concave function f.

    For concave f: cv = secant on each piece, cc = f(x) on each piece.
    The tightest is: cv = max of secants over the partition containing x,
    cc = f(x).
    """
    part_lbs, part_ubs = _partition_bounds(lb, ub, k)

    cc_base = f(x)

    def _secant_one(bounds):
        p_lb, p_ub = bounds[0], bounds[1]
        return _secant(f, x, p_lb, p_ub)

    bounds_stacked = jnp.stack([part_lbs, part_ubs], axis=-1)
    cvs = jax.vmap(_secant_one)(bounds_stacked)

    in_partition = (x >= part_lbs - 1e-15) & (x <= part_ubs + 1e-15)
    masked_cvs = jnp.where(in_partition, cvs, -jnp.inf)
    cv = jnp.max(masked_cvs)

    # Fallback
    no_match = ~jnp.any(in_partition)
    std_cv = _secant(f, x, lb, ub)
    cv = jnp.where(no_match, std_cv, cv)

    return cv, cc_base


def piecewise_relax_exp(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of exp(x) on [lb, ub].

    exp is convex: cv = exp(x), cc = piecewise secant.
    Returns (cv, cc).
    """
    return _piecewise_convex_relax(jnp.exp, x, lb, ub, k)


def piecewise_relax_log(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of log(x) on [lb, ub] (lb > 0).

    log is concave: cv = piecewise secant, cc = log(x).
    Returns (cv, cc).
    """
    return _piecewise_concave_relax(jnp.log, x, lb, ub, k)


def piecewise_relax_sqrt(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of sqrt(x) on [lb, ub] (lb >= 0).

    sqrt is concave: cv = piecewise secant, cc = sqrt(x).
    Returns (cv, cc).
    """
    return _piecewise_concave_relax(jnp.sqrt, x, lb, ub, k)


def piecewise_relax_sin(x, lb, ub, k=8):
    """Piecewise McCormick relaxation of sin(x) on [lb, ub].

    Partitions [lb, ub] into k sub-intervals and computes sin relaxation
    on each, taking the tightest envelope.

    Returns (cv, cc).
    """
    # For intervals >= 2*pi, just return [-1, 1]
    wide = (ub - lb) >= 2.0 * jnp.pi

    part_lbs, part_ubs = _partition_bounds(lb, ub, k)

    # Compute relaxation on each partition using the standard relax_sin.
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

    # Fallback if no partition matched, and also ensure piecewise is
    # always at least as tight as the standard envelope. For functions
    # with mixed convexity like sin, sub-interval regime changes can
    # sometimes produce wider individual-point envelopes than standard.
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
