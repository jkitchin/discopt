"""
Convex Envelopes for Composite Operations.

Provides relaxation functions for trilinear products (x*y*z), fractional
expressions (x/y), signomial terms (x^a for non-integer a), and
specialized tight envelopes for exp, log, and power functions.

All functions return (cv, cc) tuples where cv <= f(x) <= cc, are pure JAX,
and are compatible with jax.jit, jax.grad, and jax.vmap.
"""

from __future__ import annotations

import jax.numpy as jnp

from discopt._jax.mccormick import (
    _secant,
    relax_bilinear,
)


def relax_trilinear(x, y, z, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub):
    """McCormick relaxation of x*y*z via nested bilinear decomposition.

    Decomposes as (x*y)*z (Meyer-Floudas approach). First relaxes the
    bilinear term w = x*y, then relaxes w*z.

    Args:
        x, y, z: point values
        x_lb, x_ub: bounds on x
        y_lb, y_ub: bounds on y
        z_lb, z_ub: bounds on z

    Returns:
        (cv, cc) where cv <= x*y*z <= cc
    """
    # Step 1: relax w = x*y
    w_cv, w_cc = relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub)

    # Compute range bounds for w = x*y
    corners = jnp.array([x_lb * y_lb, x_lb * y_ub, x_ub * y_lb, x_ub * y_ub])
    w_lb = jnp.min(corners)
    w_ub = jnp.max(corners)

    # Step 2: relax w*z using bilinear relaxation
    # Use w_cv for underestimator composition and w_cc for overestimator
    # For soundness, compose both and take the tightest
    cv1, cc1 = relax_bilinear(w_cv, z, w_lb, w_ub, z_lb, z_ub)
    cv2, cc2 = relax_bilinear(w_cc, z, w_lb, w_ub, z_lb, z_ub)

    cv = jnp.minimum(cv1, cv2)
    cc = jnp.maximum(cc1, cc2)

    return cv, cc


def relax_fractional(x, y, x_lb, x_ub, y_lb, y_ub):
    """McCormick relaxation of x/y where y bounds exclude zero.

    Composes: x/y = x * (1/y). First relaxes 1/y using convex envelope
    (1/y is convex on positive or negative half-line), then uses bilinear
    relaxation for x * (1/y).

    Args:
        x: point value for numerator
        y: point value for denominator
        x_lb, x_ub: bounds on x
        y_lb, y_ub: bounds on y (must not contain 0)

    Returns:
        (cv, cc) where cv <= x/y <= cc
    """

    def recip(t):
        return 1.0 / t

    # 1/y is convex on (0, inf) and on (-inf, 0)
    recip_val = recip(y)
    recip_sec = _secant(recip, y, y_lb, y_ub)

    # cv(1/y) = 1/y (function value for convex), cc(1/y) = secant
    recip_cv = recip_val
    recip_cc = recip_sec

    # Bounds on 1/y (note the swap for positive y)
    r1 = recip(y_lb)
    r2 = recip(y_ub)
    recip_lb = jnp.minimum(r1, r2)
    recip_ub = jnp.maximum(r1, r2)

    # Compose x * (1/y) via bilinear relaxation using both cv and cc of 1/y
    cv1, cc1 = relax_bilinear(x, recip_cv, x_lb, x_ub, recip_lb, recip_ub)
    cv2, cc2 = relax_bilinear(x, recip_cc, x_lb, x_ub, recip_lb, recip_ub)

    cv = jnp.minimum(cv1, cv2)
    cc = jnp.maximum(cc1, cc2)

    return cv, cc


def relax_signomial(x, lb, ub, a):
    """McCormick relaxation of x^a for non-integer exponent a on [lb, ub].

    Handles different regimes based on sign of a and convexity:
      - a in (0,1): concave on (0,inf), cv = secant, cc = x^a
      - a > 1: convex on (0,inf), cv = x^a, cc = secant
      - a < 0: convex on (0,inf), cv = x^a, cc = secant

    Requires lb > 0 for all cases.

    Args:
        x: point value
        lb: lower bound (must be > 0)
        ub: upper bound
        a: real exponent

    Returns:
        (cv, cc) where cv <= x^a <= cc
    """

    def f(t):
        return t**a

    f_val = f(x)
    sec_val = _secant(f, x, lb, ub)

    # a in (0, 1): concave => cv = secant, cc = f(x)
    concave_cv = sec_val
    concave_cc = f_val

    # a > 1 or a < 0: convex => cv = f(x), cc = secant
    convex_cv = f_val
    convex_cc = sec_val

    is_concave = (a > 0.0) & (a < 1.0)

    cv = jnp.where(is_concave, concave_cv, convex_cv)
    cc = jnp.where(is_concave, concave_cc, convex_cc)

    return cv, cc


# ---------------------------------------------------------------------------
# Tight envelopes for exp(x), log(x) — exact convex/concave envelopes
# ---------------------------------------------------------------------------


def relax_exp_tight(x, lb, ub):
    """Tight convex envelope for exp(x) on [lb, ub].

    exp is globally convex, so cv = exp(x) (exact). The concave
    overestimator is the secant through (lb, exp(lb)) and (ub, exp(ub)).

    This is equivalent to the standard McCormick relaxation for exp,
    but provided here for consistency with the envelope API.

    Returns (cv, cc).
    """
    cv = jnp.exp(x)
    cc = _secant(jnp.exp, x, lb, ub)
    return cv, cc


def relax_log_tight(x, lb, ub):
    """Tight concave envelope for log(x) on [lb, ub] (lb > 0).

    log is globally concave, so cc = log(x) (exact). The convex
    underestimator is the secant through (lb, log(lb)) and (ub, log(ub)).

    Returns (cv, cc).
    """
    cc = jnp.log(x)
    cv = _secant(jnp.log, x, lb, ub)
    return cv, cc


# ---------------------------------------------------------------------------
# Integer power envelopes: x^p for integer p
# ---------------------------------------------------------------------------


def relax_power_int(x, lb, ub, p):
    """Tight envelope for x^p where p is a positive integer on [lb, ub].

    - p even: x^p is convex on all of R.
      cv = x^p, cc = secant.
    - p odd, lb >= 0: x^p is convex.
      cv = x^p, cc = secant.
    - p odd, ub <= 0: x^p is concave.
      cv = secant, cc = x^p.
    - p odd, lb < 0 < ub: x^p is convex-concave (inflection at 0).
      cv = tangent at lb for x<0, function for x>=0, clamped by secant.
      cc = tangent at ub for x>0, function for x<=0, clamped by secant.

    Args:
        x: point value
        lb: lower bound
        ub: upper bound
        p: positive integer exponent

    Returns:
        (cv, cc) where cv <= x^p <= cc
    """

    def f(t):
        return t**p

    f_val = f(x)
    sec_val = _secant(f, x, lb, ub)

    # Even power: always convex
    even_cv = f_val
    even_cc = sec_val

    # Odd power, lb >= 0: convex
    odd_pos_cv = f_val
    odd_pos_cc = sec_val

    # Odd power, ub <= 0: concave
    odd_neg_cv = sec_val
    odd_neg_cc = f_val

    # Odd power, mixed sign: tangent line underestimator from lb
    # For convex-concave with inflection at 0:
    # Tangent at lb: f'(lb) = p * lb^(p-1), tangent: f(lb) + f'(lb)(x - lb)
    # Tangent at ub: f'(ub) = p * ub^(p-1), tangent: f(ub) + f'(ub)(x - ub)
    f_lb = lb**p
    f_ub = ub**p
    fp_lb = p * lb ** (p - 1)
    fp_ub = p * ub ** (p - 1)

    tangent_lb = f_lb + fp_lb * (x - lb)
    tangent_ub = f_ub + fp_ub * (x - ub)

    # cv: max(tangent_at_lb, secant) for x < 0, f(x) for x >= 0
    odd_mixed_cv = jnp.where(x >= 0, f_val, jnp.maximum(tangent_lb, sec_val))
    odd_mixed_cv = jnp.minimum(odd_mixed_cv, f_val)  # must be <= f(x)
    # cc: min(tangent_at_ub, secant) for x > 0, f(x) for x <= 0
    odd_mixed_cc = jnp.where(x <= 0, f_val, jnp.minimum(tangent_ub, sec_val))
    odd_mixed_cc = jnp.maximum(odd_mixed_cc, f_val)  # must be >= f(x)

    is_even = (p % 2) == 0
    is_pos = lb >= 0
    is_neg = ub <= 0

    cv = jnp.where(
        is_even,
        even_cv,
        jnp.where(is_pos, odd_pos_cv, jnp.where(is_neg, odd_neg_cv, odd_mixed_cv)),
    )
    cc = jnp.where(
        is_even,
        even_cc,
        jnp.where(is_pos, odd_pos_cc, jnp.where(is_neg, odd_neg_cc, odd_mixed_cc)),
    )

    return cv, cc


# ---------------------------------------------------------------------------
# Exp-log compositions
# ---------------------------------------------------------------------------


def relax_exp_bilinear(x, y, x_lb, x_ub, y_lb, y_ub):
    """Relaxation of exp(x) * y via composition.

    Decomposes as bilinear(exp(x), y) where exp(x) bounds are computed
    from x bounds.

    Returns (cv, cc).
    """
    exp_x = jnp.exp(x)
    exp_lb = jnp.exp(x_lb)
    exp_ub = jnp.exp(x_ub)
    return relax_bilinear(exp_x, y, exp_lb, exp_ub, y_lb, y_ub)


def relax_log_sum(x, y, x_lb, x_ub, y_lb, y_ub):
    """Relaxation of log(x + y) on [x_lb, x_ub] x [y_lb, y_ub].

    log(x+y) is concave in (x, y) jointly, so cc = log(x+y) and
    cv = secant plane (linearization).

    Uses the gradient-based secant: cv = log(x0+y0) + (1/(x0+y0))*(x+y-x0-y0)
    evaluated at the midpoint of the bounds.

    Returns (cv, cc).
    """
    s = x + y
    s_lb = x_lb + y_lb
    s_ub = x_ub + y_ub
    # Ensure positive domain
    s_lb = jnp.maximum(s_lb, 1e-30)

    cc = jnp.log(jnp.maximum(s, 1e-30))
    cv = _secant(lambda t: jnp.log(jnp.maximum(t, 1e-30)), s, s_lb, s_ub)

    return cv, cc
