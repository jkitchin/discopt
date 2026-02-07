"""
McCormick Relaxation Primitives.

Provides convex/concave envelope functions for each mathematical operation,
enabling the B&B solver to compute valid lower/upper bounds on subproblems.

For a function f(x) on interval [lb, ub], each relaxation returns (cv, cc) where:
  - cv <= f(x) for all x in [lb, ub]  (convex underestimator)
  - cc >= f(x) for all x in [lb, ub]  (concave overestimator)

All functions are pure JAX and compatible with jax.jit, jax.grad, and jax.vmap.
"""

from __future__ import annotations

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secant(f, x, lb, ub):
    """Secant line of f between (lb, f(lb)) and (ub, f(ub)) evaluated at x.

    When lb == ub, falls back to f(x) to avoid division by zero.
    """
    f_lb = f(lb)
    f_ub = f(ub)
    slope = (f_ub - f_lb) / (ub - lb)
    line = f_lb + slope * (x - lb)
    # Degenerate case: lb ≈ ub -> just return f(x)
    return jnp.where(jnp.abs(ub - lb) < 1e-15, f(x), line)


# ---------------------------------------------------------------------------
# Bilinear product:  f(x,y) = x * y
# ---------------------------------------------------------------------------

def relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub):
    """McCormick relaxation of x*y given bounds on x and y.

    Returns (cv, cc) where cv <= x*y <= cc.
    """
    # Convex underestimator: max of two affine underestimators
    cv1 = x_lb * y + x * y_lb - x_lb * y_lb
    cv2 = x_ub * y + x * y_ub - x_ub * y_ub
    cv = jnp.maximum(cv1, cv2)

    # Concave overestimator: min of two affine overestimators
    cc1 = x_ub * y + x * y_lb - x_ub * y_lb
    cc2 = x_lb * y + x * y_ub - x_lb * y_ub
    cc = jnp.minimum(cc1, cc2)

    return cv, cc


# ---------------------------------------------------------------------------
# Addition / Subtraction / Negation  (exact relaxations)
# ---------------------------------------------------------------------------

def relax_add(cv_x, cc_x, cv_y, cc_y):
    """Relaxation of x + y given relaxations of x and y.

    Returns (cv, cc). This is exact.
    """
    return cv_x + cv_y, cc_x + cc_y


def relax_sub(cv_x, cc_x, cv_y, cc_y):
    """Relaxation of x - y given relaxations of x and y.

    Returns (cv, cc). This is exact.
    """
    return cv_x - cc_y, cc_x - cv_y


def relax_neg(cv_x, cc_x):
    """Relaxation of -x given relaxation of x.

    Returns (cv, cc). This is exact.
    """
    return -cc_x, -cv_x


# ---------------------------------------------------------------------------
# Division:  f(x,y) = x / y  via  x * (1/y)
# ---------------------------------------------------------------------------

def _relax_reciprocal(y, y_lb, y_ub):
    """McCormick relaxation of 1/y on [y_lb, y_ub].

    Requires that 0 is not in [y_lb, y_ub].
    1/y is convex on (0, inf) and convex on (-inf, 0).
    """
    # 1/y is convex when y > 0 and convex when y < 0
    # Both cases: cv = 1/y, cc = secant
    f = lambda t: 1.0 / t
    cv = f(y)
    cc = _secant(f, y, y_lb, y_ub)
    return cv, cc


def relax_div(x, y, x_lb, x_ub, y_lb, y_ub):
    """McCormick relaxation of x/y given bounds.

    Requires 0 not in [y_lb, y_ub].
    Uses the composition: x/y = x * (1/y) with bilinear relaxation.

    Returns (cv, cc).
    """
    # First get relaxation of 1/y
    recip_cv, recip_cc = _relax_reciprocal(y, y_lb, y_ub)
    recip_lb = 1.0 / y_ub  # min of 1/y when y > 0 (note swap)
    recip_ub = 1.0 / y_lb
    # Ensure correct ordering
    recip_lb_sorted = jnp.minimum(recip_lb, recip_ub)
    recip_ub_sorted = jnp.maximum(recip_lb, recip_ub)
    # Use bilinear relaxation of x * recip
    return relax_bilinear(x, recip_cv, x_lb, x_ub, recip_lb_sorted, recip_ub_sorted)


# ---------------------------------------------------------------------------
# Power:  f(x) = x^n  (integer exponent)
# ---------------------------------------------------------------------------

def relax_pow(x, lb, ub, n):
    """McCormick relaxation of x^n for integer n on [lb, ub].

    Returns (cv, cc).

    - n == 1: exact (linear)
    - n even: x^n is convex -> cv = x^n, cc = secant
    - n odd >= 3: x^n is convex on [0,inf), concave on (-inf,0]
    """
    f = lambda t: t ** n

    if n == 1:
        return x, x

    if n % 2 == 0:
        # Even power: always convex
        cv = f(x)
        cc = _secant(f, x, lb, ub)
        return cv, cc

    # Odd power, n >= 3:
    # Convex on [0, inf), concave on (-inf, 0].
    # We must handle three bound regimes with jnp.where since bounds may be traced.

    # Case 1: lb >= 0 -> fully convex: cv = f(x), cc = secant
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)

    # Case 2: ub <= 0 -> fully concave: cv = secant, cc = f(x)
    case2_cv = _secant(f, x, lb, ub)
    case2_cc = f(x)

    # Case 3: lb < 0 < ub -> inflection at 0
    # Use piecewise secants on each half for soundness:
    # - x >= 0: f is convex -> cv = f(x), cc = secant on [0, ub]
    # - x < 0:  f is concave -> cv = secant on [lb, 0], cc = f(x)
    zero = jnp.zeros_like(x)
    sec_neg = _secant(f, x, lb, zero)
    sec_pos = _secant(f, x, zero, ub)
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))

    # Select regime based on bounds (may be JAX-traced under vmap)
    is_nonneg = lb >= 0
    is_nonpos = ub <= 0

    cv = jnp.where(is_nonneg, case1_cv,
         jnp.where(is_nonpos, case2_cv,
                    case3_cv))

    cc = jnp.where(is_nonneg, case1_cc,
         jnp.where(is_nonpos, case2_cc,
                    case3_cc))

    return cv, cc


# ---------------------------------------------------------------------------
# Univariate convex functions: cv = f(x), cc = secant
# ---------------------------------------------------------------------------

def relax_exp(x, lb, ub):
    """McCormick relaxation of exp(x) on [lb, ub].

    exp is convex: cv = exp(x), cc = secant line.
    Returns (cv, cc).
    """
    cv = jnp.exp(x)
    cc = _secant(jnp.exp, x, lb, ub)
    return cv, cc


def relax_square(x, lb, ub):
    """McCormick relaxation of x^2 on [lb, ub].

    x^2 is convex: cv = x^2, cc = secant line.
    Returns (cv, cc).
    """
    f = lambda t: t ** 2
    cv = f(x)
    cc = _secant(f, x, lb, ub)
    return cv, cc


def relax_abs(x, lb, ub):
    """McCormick relaxation of |x| on [lb, ub].

    |x| is convex: cv = |x|, cc = secant line when lb < 0 < ub, else |x|.
    Returns (cv, cc).
    """
    cv = jnp.abs(x)
    # When the interval doesn't contain 0, |x| is affine, so cc = |x| exactly.
    # When it does contain 0, use secant.
    cc_secant = _secant(jnp.abs, x, lb, ub)
    # If lb >= 0 or ub <= 0, |x| is affine on the interval -> cc = |x|
    contains_zero = (lb < 0) & (ub > 0)
    cc = jnp.where(contains_zero, cc_secant, jnp.abs(x))
    return cv, cc


# ---------------------------------------------------------------------------
# Univariate concave functions: cv = secant, cc = f(x)
# ---------------------------------------------------------------------------

def relax_sqrt(x, lb, ub):
    """McCormick relaxation of sqrt(x) on [lb, ub] (lb >= 0).

    sqrt is concave: cv = secant line, cc = sqrt(x).
    Returns (cv, cc).
    """
    cc = jnp.sqrt(x)
    cv = _secant(jnp.sqrt, x, lb, ub)
    return cv, cc


def relax_log(x, lb, ub):
    """McCormick relaxation of log(x) on [lb, ub] (lb > 0).

    log is concave: cv = secant line, cc = log(x).
    Returns (cv, cc).
    """
    cc = jnp.log(x)
    cv = _secant(jnp.log, x, lb, ub)
    return cv, cc


def relax_log2(x, lb, ub):
    """McCormick relaxation of log2(x) on [lb, ub] (lb > 0).

    log2 is concave: cv = secant line, cc = log2(x).
    Returns (cv, cc).
    """
    cc = jnp.log2(x)
    cv = _secant(jnp.log2, x, lb, ub)
    return cv, cc


def relax_log10(x, lb, ub):
    """McCormick relaxation of log10(x) on [lb, ub] (lb > 0).

    log10 is concave: cv = secant line, cc = log10(x).
    Returns (cv, cc).
    """
    cc = jnp.log10(x)
    cv = _secant(jnp.log10, x, lb, ub)
    return cv, cc


# ---------------------------------------------------------------------------
# Trigonometric: sin, cos, tan
# ---------------------------------------------------------------------------

def relax_sin(x, lb, ub):
    """McCormick relaxation of sin(x) on [lb, ub].

    For intervals wider than 2*pi, relaxation is [-1, 1].
    For narrower intervals, uses a sound approach based on the range of sin
    on the interval and secant/function-value envelopes.

    Returns (cv, cc).
    """
    # If interval spans >= 2*pi, use [-1, 1]
    wide = (ub - lb) >= 2.0 * jnp.pi

    # Compute min and max of sin on [lb, ub] for the narrow case.
    # sin achieves -1 at x = -pi/2 + 2*k*pi and +1 at x = pi/2 + 2*k*pi.
    # We sample critical points to bound the range.
    sin_lb = jnp.sin(lb)
    sin_ub = jnp.sin(ub)
    sin_x = jnp.sin(x)

    # Check if interval contains a maximum (pi/2 + 2*k*pi)
    # k range that could fall in [lb, ub]
    k_min_max = jnp.ceil((lb - jnp.pi / 2) / (2 * jnp.pi))
    max_point = jnp.pi / 2 + k_min_max * 2 * jnp.pi
    has_max = max_point <= ub

    # Check if interval contains a minimum (-pi/2 + 2*k*pi)
    k_min_min = jnp.ceil((lb + jnp.pi / 2) / (2 * jnp.pi))
    min_point = -jnp.pi / 2 + k_min_min * 2 * jnp.pi
    has_min = min_point <= ub

    sin_range_max = jnp.where(has_max, 1.0, jnp.maximum(sin_lb, sin_ub))
    sin_range_min = jnp.where(has_min, -1.0, jnp.minimum(sin_lb, sin_ub))

    # Sound relaxation for narrow intervals:
    # Use secant as one envelope, function value clamped as other.
    # The secant from (lb, sin(lb)) to (ub, sin(ub)) is a valid approximation.
    sec = _secant(jnp.sin, x, lb, ub)

    # Determine concavity on the interval.
    # sin'' = -sin, so sin is concave where sin > 0 and convex where sin < 0.
    # For a general interval, we use a safe approach:
    # cv = min(sin(x), secant) clamped to sin_range_min
    # cc = max(sin(x), secant) clamped to sin_range_max
    # But this doesn't guarantee cv <= sin(x) everywhere.

    # Instead, use the range-based approach for soundness:
    # cv = guaranteed lower bound: max of (secant, sin(x)) whichever is lower
    # For soundness, the simplest correct approach:
    # cv = min(sin(x), secant) -- NO, this might be > sin(x) at some points
    # Actually cv <= sin(x) always, so we need the smaller of our approximations.

    # Correct approach: if sin is concave on [lb, ub] (sin > 0 throughout):
    #   cv = secant (below concave function), cc = sin(x)
    # If sin is convex on [lb, ub] (sin < 0 throughout):
    #   cv = sin(x), cc = secant (above convex function)
    # Mixed: use both secant and function value

    # For fully concave region (sin_range_min >= 0):
    # secant underestimates, function overestimates
    concave_cv = sec
    concave_cc = sin_x

    # For fully convex region (sin_range_max <= 0):
    # function underestimates, secant overestimates
    convex_cv = sin_x
    convex_cc = sec

    # For mixed region, use range bounds for safety:
    # cv: the minimum of secant and function is <= sin(x)
    # cc: the maximum of secant and function is >= sin(x)
    mixed_cv = jnp.minimum(sin_x, sec)
    mixed_cc = jnp.maximum(sin_x, sec)

    is_concave = sin_range_min >= -1e-10
    is_convex = sin_range_max <= 1e-10

    narrow_cv = jnp.where(is_concave, concave_cv,
                jnp.where(is_convex, convex_cv,
                          mixed_cv))
    narrow_cc = jnp.where(is_concave, concave_cc,
                jnp.where(is_convex, convex_cc,
                          mixed_cc))

    cv = jnp.where(wide, -1.0 * jnp.ones_like(x), narrow_cv)
    cc = jnp.where(wide, 1.0 * jnp.ones_like(x), narrow_cc)

    return cv, cc


def relax_cos(x, lb, ub):
    """McCormick relaxation of cos(x) on [lb, ub].

    Uses the identity cos(x) = sin(x + pi/2).
    Returns (cv, cc).
    """
    return relax_sin(x + jnp.pi / 2, lb + jnp.pi / 2, ub + jnp.pi / 2)


def relax_tan(x, lb, ub):
    """McCormick relaxation of tan(x) on [lb, ub].

    Requires [lb, ub] within a single period (-pi/2, pi/2) + k*pi.
    tan has an inflection point at k*pi: convex on [k*pi, pi/2+k*pi),
    concave on (-pi/2+k*pi, k*pi].

    For the principal period (-pi/2, pi/2):
    - lb >= 0: convex -> cv = tan(x), cc = secant
    - ub <= 0: concave -> cv = secant, cc = tan(x)
    - lb < 0 < ub: piecewise with separate secants per half

    Returns (cv, cc).
    """
    f = jnp.tan

    # Shift to principal period by finding the nearest inflection point
    # For simplicity, assume the interval is within one period.
    # Determine the inflection point center = k*pi nearest to midpoint
    mid = 0.5 * (lb + ub)
    k = jnp.round(mid / jnp.pi)
    center = k * jnp.pi

    # Case 1: lb >= center -> convex half: cv = f(x), cc = secant
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)

    # Case 2: ub <= center -> concave half: cv = secant, cc = f(x)
    case2_cv = _secant(f, x, lb, ub)
    case2_cc = f(x)

    # Case 3: lb < center < ub -> piecewise
    sec_neg = _secant(f, x, lb, center)
    sec_pos = _secant(f, x, center, ub)
    case3_cv = jnp.where(x >= center, f(x), sec_neg)
    case3_cc = jnp.where(x >= center, sec_pos, f(x))

    is_convex_half = lb >= center
    is_concave_half = ub <= center

    cv = jnp.where(is_convex_half, case1_cv,
         jnp.where(is_concave_half, case2_cv,
                    case3_cv))
    cc = jnp.where(is_convex_half, case1_cc,
         jnp.where(is_concave_half, case2_cc,
                    case3_cc))

    return cv, cc


# ---------------------------------------------------------------------------
# Composite: sign, min, max
# ---------------------------------------------------------------------------

def relax_sign(x, lb, ub):
    """McCormick relaxation of sign(x) on [lb, ub].

    sign(x) = -1 if x < 0, 0 if x == 0, +1 if x > 0.
    Returns (cv, cc).
    """
    # If lb >= 0: sign = +1 (or 0 at x=0, but we approximate)
    # If ub <= 0: sign = -1 (or 0 at x=0)
    # If lb < 0 < ub: sign ranges from -1 to +1

    sign_x = jnp.sign(x)

    # When lb >= 0: cv = cc = sign(x) (which is 0 or 1)
    # Actually for soundness with sign at boundary:
    # If lb > 0: cv = cc = 1
    # If lb == 0: cv = 0, cc = 1 (sign can be 0 or positive)
    # If ub < 0: cv = cc = -1
    # If ub == 0: cv = -1, cc = 0
    # If lb < 0 < ub: cv = -1, cc = 1

    cv = jnp.where(lb > 0, 1.0,
         jnp.where(ub < 0, -1.0,
         jnp.where(lb == 0, 0.0,
                    -1.0)))

    cc = jnp.where(lb > 0, 1.0,
         jnp.where(ub < 0, -1.0,
         jnp.where(ub == 0, 0.0,
                    1.0)))

    # Tighten: ensure cv <= sign(x) <= cc
    # The constant bounds above are always sound.
    # We can tighten by using sign(x) where it's exact.
    cv = jnp.where(lb > 0, sign_x,
         jnp.where(ub < 0, sign_x,
                    cv))
    cc = jnp.where(lb > 0, sign_x,
         jnp.where(ub < 0, sign_x,
                    cc))

    return cv, cc


def relax_min(x, y, cv_x, cc_x, cv_y, cc_y):
    """McCormick relaxation of min(x, y).

    min is concave: cv = secant-based, cc = min(cc_x, cc_y).
    Uses the identity: min(x,y) = 0.5 * (x + y - |x - y|).

    Returns (cv, cc).
    """
    # min(x, y) <= min(cc_x, cc_y) since cc_x >= x, cc_y >= y
    cc = jnp.minimum(cc_x, cc_y)
    # min(x, y) >= min(cv_x, cv_y) is NOT correct (min is not monotone like that)
    # Instead: min(x, y) >= cv_x if x <= y, else min(x,y) = y >= cv_y
    # Safe lower bound: min(cv_x, cv_y) - but this is too loose.
    # Better: since min(x,y) = (x + y - |x-y|)/2 and cv_z >= cv_x + cv_y - (cc of |x-y|)
    # For soundness, use: cv = min(cv_x, cv_y) which may not always be <= min(x,y).
    # Actually min(x,y) >= min(cv_x, cv_y)? No: cv_x <= x and cv_y <= y,
    # so min(cv_x, cv_y) <= min(x, y)? Not necessarily.
    # Example: cv_x = 0, cv_y = 5, x = 1, y = 6 -> min(cv) = 0, min(x,y) = 1. OK.
    # Example: cv_x = 5, cv_y = 0, x = 6, y = 1 -> min(cv) = 0, min(x,y) = 1. OK.
    # In general: min(cv_x, cv_y) <= min(x, y)?
    # Since cv_x <= x and cv_y <= y: min(cv_x, cv_y) <= max(cv_x, cv_y) <= max(x, y)
    # But also min(cv_x, cv_y) <= cv_x <= x and min(cv_x, cv_y) <= cv_y <= y
    # So min(cv_x, cv_y) <= min(x, y). Yes!
    cv = jnp.minimum(cv_x, cv_y)
    return cv, cc


def relax_max(x, y, cv_x, cc_x, cv_y, cc_y):
    """McCormick relaxation of max(x, y).

    max is convex: cv = max(cv_x, cv_y), cc uses concave overestimator.

    Returns (cv, cc).
    """
    # max(x,y) >= max(cv_x, cv_y) since cv_x <= x, cv_y <= y ->
    # max(cv_x, cv_y) <= max(x, y)
    cv = jnp.maximum(cv_x, cv_y)
    # max(x,y) <= max(cc_x, cc_y) since cc_x >= x, cc_y >= y
    cc = jnp.maximum(cc_x, cc_y)
    return cv, cc
