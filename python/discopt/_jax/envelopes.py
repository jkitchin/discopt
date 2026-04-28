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
    """Loose McCormick relaxation of x*y*z via single nested bilinear decomposition.

    Decomposes as (x*y)*z: relaxes the bilinear term w = x*y, then relaxes
    w*z using compositional McCormick over the [w_cv, w_cc] interval. This is
    a sound relaxation but not the tightest available — see
    :func:`relax_trilinear_exact`, which considers all three orderings and is
    strictly tighter on most boxes.

    Kept as a fallback / reference path for the unbounded or non-trilinear
    cases that the exact-routing detector does not cover.

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


def _nested_trilinear_one_order(a, b, c, a_lb, a_ub, b_lb, b_ub, c_lb, c_ub):
    """Single-ordering nested bilinear: relax (a*b)*c.

    Helper used by relax_trilinear_exact. Returns (cv, cc) where
    cv <= a*b*c <= cc.
    """
    w_cv, w_cc = relax_bilinear(a, b, a_lb, a_ub, b_lb, b_ub)
    corners = jnp.array([a_lb * b_lb, a_lb * b_ub, a_ub * b_lb, a_ub * b_ub])
    w_lb = jnp.min(corners)
    w_ub = jnp.max(corners)

    cv1, cc1 = relax_bilinear(w_cv, c, w_lb, w_ub, c_lb, c_ub)
    cv2, cc2 = relax_bilinear(w_cc, c, w_lb, w_ub, c_lb, c_ub)
    cv = jnp.minimum(cv1, cv2)
    cc = jnp.maximum(cc1, cc2)
    return cv, cc


def relax_trilinear_exact(x, y, z, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub):
    """Tighter relaxation of x*y*z: permutation-symmetric nested McCormick.

    Computes convex/concave envelopes of the trilinear monomial x*y*z on a
    box by considering all three nested-bilinear orderings — (x*y)*z,
    (x*z)*y, (y*z)*x — and merging with the tightest valid bounds:
    ``cv = max(cv_1, cv_2, cv_3)`` and ``cc = min(cc_1, cc_2, cc_3)``. Each
    ordering is sound, so the merged bounds are sound and provably at least
    as tight as any single ordering, with strict improvement on most
    mixed-sign boxes.

    This is an honest improvement over :func:`relax_trilinear` (which uses a
    single ordering) but is *not* the literal exact convex hull of the
    trilinear monomial described in Rikun (1997) / Meyer & Floudas (2004) /
    Locatelli (2018), which requires additional facet families. Encoding
    those facets is future work.

    All operations are pure JAX (jit/vmap/grad compatible).

    Args:
        x, y, z: point values
        x_lb, x_ub: bounds on x
        y_lb, y_ub: bounds on y
        z_lb, z_ub: bounds on z

    Returns:
        (cv, cc) where cv <= x*y*z <= cc.
    """
    # Order 1: (x*y)*z
    cv1, cc1 = _nested_trilinear_one_order(x, y, z, x_lb, x_ub, y_lb, y_ub, z_lb, z_ub)
    # Order 2: (x*z)*y
    cv2, cc2 = _nested_trilinear_one_order(x, z, y, x_lb, x_ub, z_lb, z_ub, y_lb, y_ub)
    # Order 3: (y*z)*x
    cv3, cc3 = _nested_trilinear_one_order(y, z, x, y_lb, y_ub, z_lb, z_ub, x_lb, x_ub)

    cv = jnp.maximum(jnp.maximum(cv1, cv2), cv3)
    cc = jnp.minimum(jnp.minimum(cc1, cc2), cc3)
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


# ---------------------------------------------------------------------------
# Tight trigonometric envelopes: sin(x), cos(x) using actual variable bounds
# ---------------------------------------------------------------------------


def relax_sin_tight(x, lb, ub):
    """Tight convex/concave envelope for sin(x) on [lb, ub].

    Uses secant + tangent construction based on the regime:
      - Concave regime (sin >= 0 throughout): cv = secant, cc = sin(x)
      - Convex regime (sin <= 0 throughout): cv = sin(x), cc = secant
      - Mixed or wide (>= 2*pi): falls back to [-1, 1]

    For narrow intervals in a single-convexity regime, this is significantly
    tighter than compositional McCormick.

    Args:
        x: point value
        lb: lower bound
        ub: upper bound

    Returns:
        (cv, cc) where cv <= sin(x) <= cc
    """
    pi = jnp.pi
    sin_x = jnp.sin(x)
    width = ub - lb

    # Fall back to [-1, 1] for wide intervals
    wide = width >= 2.0 * pi

    # Secant line through (lb, sin(lb)) and (ub, sin(ub))
    sec = _secant(jnp.sin, x, lb, ub)

    # Determine regime by checking if interval stays in concave or convex region.
    # Normalize lb to [0, 2*pi)
    lb_mod = jnp.fmod(lb, 2.0 * pi)
    lb_mod = jnp.where(lb_mod < 0, lb_mod + 2.0 * pi, lb_mod)
    ub_mod = lb_mod + width

    # Concave regime: sin'' < 0 on (0, pi), i.e., the interval is within [2k*pi, (2k+1)*pi]
    # This means lb_mod in [0, pi] and ub_mod <= pi
    is_concave = (lb_mod >= 0.0) & (ub_mod <= pi)

    # Convex regime: sin'' > 0 on (pi, 2*pi), i.e., interval within [(2k-1)*pi, 2k*pi]
    is_convex = (lb_mod >= pi) & (ub_mod <= 2.0 * pi)

    # Concave: cv = secant, cc = sin(x)
    concave_cv = sec
    concave_cc = sin_x

    # Convex: cv = sin(x), cc = secant
    convex_cv = sin_x
    convex_cc = sec

    # Mixed: use tangent lines + function value for tighter bounds
    # cv = max(sin(x), secant) clamped; cc = min(sin(x), secant) clamped
    # Conservative: use secant for both since we can't guarantee tightness
    mixed_cv = jnp.minimum(sec, sin_x)
    mixed_cc = jnp.maximum(sec, sin_x)

    inner_cv = jnp.where(is_convex, convex_cv, mixed_cv)
    inner_cc = jnp.where(is_convex, convex_cc, mixed_cc)
    cv = jnp.where(wide, -1.0, jnp.where(is_concave, concave_cv, inner_cv))
    cc = jnp.where(wide, 1.0, jnp.where(is_concave, concave_cc, inner_cc))

    return cv, cc


def relax_cos_tight(x, lb, ub):
    """Tight convex/concave envelope for cos(x) on [lb, ub].

    Delegates to relax_sin_tight via cos(x) = sin(x + pi/2).

    Args:
        x: point value
        lb: lower bound
        ub: upper bound

    Returns:
        (cv, cc) where cv <= cos(x) <= cc
    """
    half_pi = jnp.pi / 2.0
    return relax_sin_tight(x + half_pi, lb + half_pi, ub + half_pi)


# ---------------------------------------------------------------------------
# Multivariate signomial envelopes
# ---------------------------------------------------------------------------


def relax_asinh(x, lb, ub):
    """Tight envelope for asinh(x) on [lb, ub].

    asinh''(x) = -x/(1+x^2)^{3/2}, so:
    - x >= 0 (lb >= 0): concave, cv = secant, cc = asinh(x)
    - x <= 0 (ub <= 0): convex, cv = asinh(x), cc = secant
    - mixed: conservative bounds

    Returns (cv, cc).
    """
    f_val = jnp.arcsinh(x)
    sec = _secant(jnp.arcsinh, x, lb, ub)

    # Concave regime (lb >= 0): cv = secant, cc = f
    concave_cv, concave_cc = sec, f_val
    # Convex regime (ub <= 0): cv = f, cc = secant
    convex_cv, convex_cc = f_val, sec
    # Mixed: conservative bounds
    mixed_cv = jnp.minimum(f_val, sec)
    mixed_cc = jnp.maximum(f_val, sec)

    is_concave = lb >= 0.0
    is_convex = ub <= 0.0

    cv = jnp.where(is_convex, convex_cv, jnp.where(is_concave, concave_cv, mixed_cv))
    cc = jnp.where(is_convex, convex_cc, jnp.where(is_concave, concave_cc, mixed_cc))
    return cv, cc


def relax_acosh(x, lb, ub):
    """Tight envelope for acosh(x) on [lb, ub] where lb >= 1.

    acosh is concave on [1, inf), so cc = acosh(x) and cv = secant.

    Returns (cv, cc).
    """
    safe_lb = jnp.maximum(lb, 1.0)
    safe_x = jnp.maximum(x, 1.0)
    cc = jnp.acosh(safe_x)
    cv = _secant(jnp.acosh, safe_x, safe_lb, ub)
    return cv, cc


def relax_atanh(x, lb, ub):
    """Tight envelope for atanh(x) on [lb, ub] where -1 < lb, ub < 1.

    atanh''(x) = 2x/(1-x^2)^2, so:
    - x >= 0 (lb >= 0): convex, cv = atanh(x), cc = secant
    - x <= 0 (ub <= 0): concave, cv = secant, cc = atanh(x)
    - mixed: conservative bounds

    Returns (cv, cc).
    """
    safe_lb = jnp.maximum(lb, -1.0 + 1e-10)
    safe_ub = jnp.minimum(ub, 1.0 - 1e-10)
    safe_x = jnp.clip(x, safe_lb, safe_ub)

    f_val = jnp.arctanh(safe_x)
    sec = _secant(jnp.arctanh, safe_x, safe_lb, safe_ub)

    # Convex regime (lb >= 0): cv = f, cc = secant
    convex_cv, convex_cc = f_val, sec
    # Concave regime (ub <= 0): cv = secant, cc = f
    concave_cv, concave_cc = sec, f_val
    # Mixed
    mixed_cv = jnp.minimum(f_val, sec)
    mixed_cc = jnp.maximum(f_val, sec)

    is_convex = lb >= 0.0
    is_concave = ub <= 0.0

    cv = jnp.where(is_convex, convex_cv, jnp.where(is_concave, concave_cv, mixed_cv))
    cc = jnp.where(is_convex, convex_cc, jnp.where(is_concave, concave_cc, mixed_cc))
    return cv, cc


def relax_erf(x, lb, ub):
    """Tight envelope for erf(x) on [lb, ub].

    erf is convex on (-inf, 0] and concave on [0, inf), with inflection at 0.
    - x <= 0: convex, cv = erf(x), cc = secant
    - x >= 0: concave, cv = secant, cc = erf(x)
    - mixed: conservative bounds

    Returns (cv, cc).
    """
    from jax.scipy.special import erf

    f_val = erf(x)
    sec = _secant(erf, x, lb, ub)

    # Convex regime (ub <= 0): cv = f, cc = secant
    convex_cv, convex_cc = f_val, sec
    # Concave regime (lb >= 0): cv = secant, cc = f
    concave_cv, concave_cc = sec, f_val
    # Mixed
    mixed_cv = jnp.minimum(f_val, sec)
    mixed_cc = jnp.maximum(f_val, sec)

    is_convex = ub <= 0.0
    is_concave = lb >= 0.0

    cv = jnp.where(is_convex, convex_cv, jnp.where(is_concave, concave_cv, mixed_cv))
    cc = jnp.where(is_convex, convex_cc, jnp.where(is_concave, concave_cc, mixed_cc))
    return cv, cc


def relax_log1p(x, lb, ub):
    """Tight envelope for log(1+x) on [lb, ub] where lb > -1.

    log(1+x) is concave on (-1, inf), so cc = log(1+x) and cv = secant.

    Returns (cv, cc).
    """
    safe_lb = jnp.maximum(lb, -1.0 + 1e-15)

    cc = jnp.log1p(x)
    cv = _secant(jnp.log1p, x, safe_lb, ub)
    return cv, cc


def relax_reciprocal(x, lb, ub):
    """Tight envelope for 1/x on [lb, ub] where bounds exclude zero.

    (1/x)'' = 2/x^3, so:
    - x > 0 (lb > 0): convex, cv = 1/x, cc = secant
    - x < 0 (ub < 0): concave, cv = secant, cc = 1/x

    Returns (cv, cc).
    """

    def recip(t):
        return 1.0 / t

    f_val = recip(x)
    sec = _secant(recip, x, lb, ub)

    # Positive domain: convex
    pos_cv, pos_cc = f_val, sec
    # Negative domain: concave
    neg_cv, neg_cc = sec, f_val

    is_pos = lb > 0.0
    cv = jnp.where(is_pos, pos_cv, neg_cv)
    cc = jnp.where(is_pos, pos_cc, neg_cc)
    return cv, cc


def relax_signomial_multi(xs, lbs, ubs, exponents):
    """Relaxation of prod(x_i^{a_i}) via logarithmic decomposition.

    Decomposes prod(x_i^{a_i}) = exp(sum(a_i * log(x_i))), then composes
    tight log, scaling, summation, and tight exp relaxations.

    Requires all lbs > 0 (positive domain).

    Args:
        xs: array of point values, shape (n,)
        lbs: array of lower bounds, shape (n,)
        ubs: array of upper bounds, shape (n,)
        exponents: array of exponents, shape (n,)

    Returns:
        (cv, cc) where cv <= prod(x_i^{a_i}) <= cc
    """
    # Guard: ensure positive domain
    safe_lbs = jnp.maximum(lbs, 1e-15)
    safe_xs = jnp.maximum(xs, 1e-15)

    # Direct approach: evaluate the true value and compute simple bounds.
    # Compute bounds on the product at all corners would be expensive.
    # Instead, use the univariate signomial relaxation per-variable and compose.
    # For a product of terms, cv = product of cvs (when all positive),
    # cc = product of ccs.
    # But this only works for positive terms. Use log-space composition:

    # Step 1: Relax log(x_i) for each variable (log is concave)
    log_cvs = _secant(jnp.log, safe_xs, safe_lbs, ubs)  # secant = cv for concave log
    log_ccs = jnp.log(safe_xs)  # function value = cc for concave log

    # Step 2: Scale by exponents: a_i * log(x_i)
    # For positive exponents: a*cv_log is cv, a*cc_log is cc
    # For negative exponents: a*cc_log is cv, a*cv_log is cc (swap)
    pos_exp = exponents >= 0
    scaled_cv = jnp.where(pos_exp, exponents * log_cvs, exponents * log_ccs)
    scaled_cc = jnp.where(pos_exp, exponents * log_ccs, exponents * log_cvs)

    # Step 3: Sum over variables (sum preserves convexity/concavity)
    sum_cv = jnp.sum(scaled_cv)
    sum_cc = jnp.sum(scaled_cc)

    # Step 4: exp(sum) — exp is convex
    # cv of exp(convex_underestimator) = exp(sum_cv) (convex of convex)
    cv = jnp.exp(sum_cv)
    # For cc: exp is convex so exp(concave_overestimator) is NOT guaranteed
    # to be a valid overestimator. Use the secant on [sum_cv, sum_cc].
    # But we need cc >= true_val. The true value = exp(true_sum_log).
    # true_sum_log = sum(a_i*log(x_i)) where each a_i*log(x_i) satisfies
    # scaled_cv_i <= a_i*log(x_i) <= scaled_cc_i, so sum_cv <= true_sum_log <= sum_cc.
    # Since exp is monotonically increasing: exp(sum_cv) <= exp(true_sum_log) <= exp(sum_cc)
    # So cc = exp(sum_cc) is a valid overestimator!
    cc = jnp.exp(sum_cc)

    return cv, cc
