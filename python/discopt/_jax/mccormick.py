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


def _secant(f, x, lb, ub, fallback=jnp.inf):
    """Secant line of f between (lb, f(lb)) and (ub, f(ub)) evaluated at x.

    When ``lb == ub``, falls back to ``f(x)`` to avoid division by zero.

    When either bound is non-finite the secant is undefined
    (``slope = (f(ub) - f(lb)) / (ub - lb)`` becomes ``inf/inf`` = ``NaN``): a
    secant is only a valid envelope over a *bounded* interval. Rather than leak a
    NaN (which is not a valid over-/under-estimator and silently defeats every
    downstream ``cv <= f`` / ``f <= cc`` soundness check), we return an explicit
    *no-information* value ``fallback``. Callers pass ``+inf`` when the secant
    plays the concave-overestimator (``cc``) role and ``-inf`` when it plays the
    convex-underestimator (``cv``) role, so the resulting envelope still brackets
    ``f`` (``-inf <= f <= +inf``) without fabricating a finite bound. See C-24.
    """
    f_lb = f(lb)
    f_ub = f(ub)
    width = ub - lb
    degenerate = jnp.abs(width) < 1e-15
    # Guard the divisor so the degenerate branch never evaluates 0/0 (which is a
    # ZeroDivisionError on Python-float scalars and a NaN on arrays).
    safe_width = jnp.where(degenerate, 1.0, width)
    slope = (f_ub - f_lb) / safe_width
    line = f_lb + slope * (x - lb)
    # Degenerate case: lb ≈ ub -> just return f(x)
    line = jnp.where(degenerate, f(x), line)
    # Non-finite bound: the secant carries no information -> explicit fallback.
    both_finite = jnp.isfinite(lb) & jnp.isfinite(ub)
    return jnp.where(both_finite, line, fallback)


# ---------------------------------------------------------------------------
# Bilinear product:  f(x,y) = x * y
# ---------------------------------------------------------------------------


def relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub):
    """McCormick relaxation of x*y given bounds on x and y.

    Returns (cv, cc) where cv <= x*y <= cc.
    When both factors are non-negative (x_lb >= 0, y_lb >= 0), the
    product is non-negative — the underestimator is clamped to the
    implied lower bound x_lb * y_lb.
    """
    # Convex underestimator: max of two affine underestimators
    cv1 = x_lb * y + x * y_lb - x_lb * y_lb
    cv2 = x_ub * y + x * y_ub - x_ub * y_ub
    cv = jnp.maximum(cv1, cv2)

    # Concave overestimator: min of two affine overestimators
    cc1 = x_ub * y + x * y_lb - x_ub * y_lb
    cc2 = x_lb * y + x * y_ub - x_lb * y_ub
    cc = jnp.minimum(cc1, cc2)

    # When both factors are non-negative, the product x*y >= x_lb*y_lb.
    # The standard McCormick underestimator can dip below this implied
    # bound at interior points; clamping tightens the relaxation.
    both_nonneg = (x_lb >= 0.0) & (y_lb >= 0.0)
    cv = jnp.where(both_nonneg, jnp.maximum(cv, x_lb * y_lb), cv)

    # The McCormick envelope of x*y is only valid over a *bounded* box: with a
    # non-finite factor bound the affine terms carry ``inf*·`` / ``inf - inf`` =
    # ``NaN`` (an invalid, non-bracketing value). Replace the whole envelope with
    # an explicit no-information bracket (cv=-inf, cc=+inf) rather than leaking a
    # NaN a downstream consumer might use unguarded. See C-24.
    all_finite = jnp.isfinite(x_lb) & jnp.isfinite(x_ub) & jnp.isfinite(y_lb) & jnp.isfinite(y_ub)
    cv = jnp.where(all_finite, cv, -jnp.inf)
    cc = jnp.where(all_finite, cc, jnp.inf)

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
    def f(t):
        return 1.0 / t

    cv = f(y)
    cc = _secant(f, y, y_lb, y_ub)
    return cv, cc


def relax_div(x, y, x_lb, x_ub, y_lb, y_ub):
    """McCormick relaxation of x/y given bounds on the numerator and denominator.

    In the compiler, ``[x_lb, x_ub] = [cv_num, cc_num]`` and
    ``[y_lb, y_ub] = [cv_den, cc_den]`` are the *relaxation intervals* of the
    numerator and denominator at the evaluation point, and ``x``/``y`` are points
    inside them. Requires ``0`` not in ``[y_lb, y_ub]``.

    Two regimes (C-23):

    - **Point/linear denominator** (``y_lb == y_ub``): the denominator relaxation
      collapses to a point, so ``1/y`` is exact and the classic bilinear
      composition ``x*(1/y)`` is sound and tight. This covers a constant, a bare
      variable, or an affine denominator.
    - **Nonlinear denominator** (``y_lb != y_ub``): the bilinear composition
      reciprocated at the *midpoint* of the denominator interval is **not** a
      valid envelope — ``1/mid`` sits above the true ``1/(.)`` on a curved
      denominator, so the "convex underestimator" can exceed ``f`` (``cv > f``),
      an invalid dual bound. We instead return the sound **interval enclosure**
      of ``x/y`` over the box (``[x_lb,x_ub] * [1/y_ub, 1/y_lb]``): a constant
      ``[cv, cc]`` with ``cv <= x/y <= cc`` at every point. This is looser but
      never crosses the function; spatial branching tightens the box (and the
      denominator interval) so the enclosure shrinks.
    """
    # Sound reciprocal bounds over the denominator interval (0 excluded, so the
    # two endpoints share a sign): 1/y is monotone-decreasing, hence the range is
    # [1/y_ub, 1/y_lb] up to ordering.
    recip_lb = 1.0 / y_ub
    recip_ub = 1.0 / y_lb
    recip_lb_sorted = jnp.minimum(recip_lb, recip_ub)
    recip_ub_sorted = jnp.maximum(recip_lb, recip_ub)

    # Tight bilinear composition, valid only where the denominator relaxation
    # collapses to a point (1/y exact).
    recip_cv, _recip_cc = _relax_reciprocal(y, y_lb, y_ub)
    tight_cv, tight_cc = relax_bilinear(x, recip_cv, x_lb, x_ub, recip_lb_sorted, recip_ub_sorted)

    # Sound interval enclosure of x/y = [x_lb,x_ub] * [recip_lb,recip_ub],
    # valid for a nonlinear (non-degenerate) denominator interval.
    p1 = x_lb * recip_lb_sorted
    p2 = x_lb * recip_ub_sorted
    p3 = x_ub * recip_lb_sorted
    p4 = x_ub * recip_ub_sorted
    encl_cv = jnp.minimum(jnp.minimum(p1, p2), jnp.minimum(p3, p4))
    encl_cc = jnp.maximum(jnp.maximum(p1, p2), jnp.maximum(p3, p4))

    denom_is_point = jnp.abs(y_ub - y_lb) < 1e-12
    cv = jnp.where(denom_is_point, tight_cv, encl_cv)
    cc = jnp.where(denom_is_point, tight_cc, encl_cc)
    return cv, cc


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

    def f(t):
        return t**n

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
    case2_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case2_cc = f(x)

    # Case 3: lb < 0 < ub -> inflection at 0
    # Use piecewise secants on each half for soundness:
    # - x >= 0: f is convex -> cv = f(x), cc = secant on [0, ub]
    # - x < 0:  f is concave -> cv = secant on [lb, 0], cc = f(x)
    zero = jnp.zeros_like(x)
    sec_neg = _secant(f, x, lb, zero, fallback=-jnp.inf)  # concave-half cv role
    sec_pos = _secant(f, x, zero, ub)  # convex-half cc role
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))

    # Select regime based on bounds (may be JAX-traced under vmap)
    is_nonneg = lb >= 0
    is_nonpos = ub <= 0

    cv = jnp.where(is_nonneg, case1_cv, jnp.where(is_nonpos, case2_cv, case3_cv))

    cc = jnp.where(is_nonneg, case1_cc, jnp.where(is_nonpos, case2_cc, case3_cc))

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

    def f(t):
        return t**2

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
    cv = _secant(jnp.sqrt, x, lb, ub, fallback=-jnp.inf)
    return cv, cc


def relax_log(x, lb, ub):
    """McCormick relaxation of log(x) on [lb, ub] (lb > 0).

    log is concave: cv = secant line, cc = log(x).
    Returns (cv, cc).
    """
    cc = jnp.log(x)
    cv = _secant(jnp.log, x, lb, ub, fallback=-jnp.inf)
    return cv, cc


def relax_log2(x, lb, ub):
    """McCormick relaxation of log2(x) on [lb, ub] (lb > 0).

    log2 is concave: cv = secant line, cc = log2(x).
    Returns (cv, cc).
    """
    cc = jnp.log2(x)
    cv = _secant(jnp.log2, x, lb, ub, fallback=-jnp.inf)
    return cv, cc


def relax_log10(x, lb, ub):
    """McCormick relaxation of log10(x) on [lb, ub] (lb > 0).

    log10 is concave: cv = secant line, cc = log10(x).
    Returns (cv, cc).
    """
    cc = jnp.log10(x)
    cv = _secant(jnp.log10, x, lb, ub, fallback=-jnp.inf)
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

    narrow_cv = jnp.where(is_concave, concave_cv, jnp.where(is_convex, convex_cv, mixed_cv))
    narrow_cc = jnp.where(is_concave, concave_cc, jnp.where(is_convex, convex_cc, mixed_cc))

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

    ``tan`` is only continuous on an open branch ``(-pi/2 + k*pi, pi/2 + k*pi)``;
    it diverges at each pole ``pi/2 + k*pi``. Within one branch it has an
    inflection point at ``k*pi`` (convex on ``[k*pi, pi/2+k*pi)``, concave on
    ``(-pi/2+k*pi, k*pi]``) and a valid convex/concave envelope can be drawn.

    If ``[lb, ub]`` **straddles a pole** the branch classification below is not
    applicable — a secant drawn across the pole is not a valid under/over
    estimator (C-19). In that case we *abstain*, returning the no-information
    envelope ``(-inf, +inf)`` (matching how other relaxations abstain), leaving
    FBBT / spatial branching to shrink the box below the pole spacing.

    For a pole-free interval within the principal period ``(-pi/2, pi/2)``:
    - lb >= 0: convex -> cv = tan(x), cc = secant
    - ub <= 0: concave -> cv = secant, cc = tan(x)
    - lb < 0 < ub: piecewise with separate secants per half

    Returns (cv, cc).
    """
    f = jnp.tan

    # Shift to principal period by finding the nearest inflection point
    # (center = k*pi nearest to the midpoint); the continuous branch spanning
    # the box is (center - pi/2, center + pi/2).
    mid = 0.5 * (lb + ub)
    k = jnp.round(mid / jnp.pi)
    center = k * jnp.pi

    # --- Pole detection (C-19) -------------------------------------------
    # tan is finite and single-branch on the box iff both endpoints lie
    # strictly inside the branch centered at ``center``, whose bounding poles
    # are at ``center +/- pi/2``. Any box that reaches or crosses either pole
    # (equivalently spans >= a full period, so it would contain a pole for some
    # k) is not classifiable by the branch logic below and must abstain — a
    # secant across a pole is neither a valid under- nor over-estimator.
    lo_pole = center - 0.5 * jnp.pi
    hi_pole = center + 0.5 * jnp.pi
    pole_free = (lb > lo_pole) & (ub < hi_pole)

    # Case 1: lb >= center -> convex half: cv = f(x), cc = secant
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)

    # Case 2: ub <= center -> concave half: cv = secant, cc = f(x)
    case2_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case2_cc = f(x)

    # Case 3: lb < center < ub -> piecewise
    sec_neg = _secant(f, x, lb, center, fallback=-jnp.inf)  # concave-half cv role
    sec_pos = _secant(f, x, center, ub)  # convex-half cc role
    case3_cv = jnp.where(x >= center, f(x), sec_neg)
    case3_cc = jnp.where(x >= center, sec_pos, f(x))

    is_convex_half = lb >= center
    is_concave_half = ub <= center

    cv = jnp.where(is_convex_half, case1_cv, jnp.where(is_concave_half, case2_cv, case3_cv))
    cc = jnp.where(is_convex_half, case1_cc, jnp.where(is_concave_half, case2_cc, case3_cc))

    # Abstain across a pole: emit the no-information envelope (-inf, +inf)
    # rather than a secant drawn through the singularity.
    neg_inf = -jnp.inf * jnp.ones_like(cv)
    pos_inf = jnp.inf * jnp.ones_like(cc)
    cv = jnp.where(pole_free, cv, neg_inf)
    cc = jnp.where(pole_free, cc, pos_inf)

    return cv, cc


# ---------------------------------------------------------------------------
# Inverse trigonometric: atan, asin, acos
# ---------------------------------------------------------------------------


def relax_atan(x, lb, ub):
    """McCormick relaxation of atan(x) on [lb, ub].

    atan is concave on [0, inf) and convex on (-inf, 0].
    Returns (cv, cc).
    """
    f = jnp.arctan

    # Case 1: lb >= 0 -> concave: cv = secant, cc = f(x)
    case1_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case1_cc = f(x)

    # Case 2: ub <= 0 -> convex: cv = f(x), cc = secant
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)

    # Case 3: lb < 0 < ub -> mixed
    sec_neg = _secant(f, x, lb, 0.0)  # convex-half cc role (x < 0)
    sec_pos = _secant(f, x, 0.0, ub, fallback=-jnp.inf)  # concave-half cv role (x >= 0)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)

    is_concave = lb >= 0
    is_convex = ub <= 0

    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def relax_asin(x, lb, ub):
    """McCormick relaxation of asin(x) on [lb, ub] (subset of [-1, 1]).

    asin''(x) = x*(1 - x**2)**(-3/2), so asin is convex on [0, 1] and
    concave on [-1, 0] (mirror image of acos). On the convex branch the
    function itself is the underestimator (cv) and the secant the
    overestimator (cc); on the concave branch the roles reverse. This is
    the same curvature layout as sinh (convex on [0, inf)).
    Returns (cv, cc).
    """
    f = jnp.arcsin

    # Case 1: lb >= 0 -> convex: cv = f(x), cc = secant (cc/overestimator role)
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)

    # Case 2: ub <= 0 -> concave: cv = secant (cv/underestimator role), cc = f(x)
    case2_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case2_cc = f(x)

    # Case 3: lb < 0 < ub -> straddles the inflection at 0. Split at 0:
    # positive (convex) side -> f(x)/sec_pos; negative (concave) side ->
    # sec_neg/f(x).
    sec_neg = _secant(f, x, lb, 0.0, fallback=-jnp.inf)  # cv role (x < 0 concave half)
    sec_pos = _secant(f, x, 0.0, ub)  # cc role (x >= 0 convex half)
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))

    is_convex = lb >= 0
    is_concave = ub <= 0

    cv = jnp.where(is_convex, case1_cv, jnp.where(is_concave, case2_cv, case3_cv))
    cc = jnp.where(is_convex, case1_cc, jnp.where(is_concave, case2_cc, case3_cc))
    return cv, cc


def relax_acos(x, lb, ub):
    """McCormick relaxation of acos(x) on [lb, ub] (subset of [-1, 1]).

    acos''(x) = -x*(1 - x**2)**(-3/2), so acos is concave on [0, 1] and
    convex on [-1, 0] (mirror image of asin). acos is decreasing, but the
    secant/tangent under- vs over-estimator roles depend only on curvature,
    not on the sign of the slope. On the concave branch the secant is the
    underestimator (cv) and the function the overestimator (cc); on the
    convex branch the roles reverse. Same curvature layout as tanh
    (concave on [0, inf)).
    Returns (cv, cc).
    """
    f = jnp.arccos

    # Case 1: lb >= 0 -> concave: cv = secant (cv/underestimator role), cc = f(x)
    case1_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case1_cc = f(x)

    # Case 2: ub <= 0 -> convex: cv = f(x), cc = secant (cc/overestimator role)
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)

    # Case 3: lb < 0 < ub -> straddles the inflection at 0. Split at 0:
    # positive (concave) side -> sec_pos/f(x); negative (convex) side ->
    # f(x)/sec_neg.
    sec_neg = _secant(f, x, lb, 0.0)  # cc role (x < 0 convex half)
    sec_pos = _secant(f, x, 0.0, ub, fallback=-jnp.inf)  # cv role (x >= 0 concave half)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)

    is_concave = lb >= 0
    is_convex = ub <= 0

    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


# ---------------------------------------------------------------------------
# Hyperbolic: sinh, cosh, tanh
# ---------------------------------------------------------------------------


def relax_sinh(x, lb, ub):
    """McCormick relaxation of sinh(x) on [lb, ub].

    sinh is convex on [0, inf) and concave on (-inf, 0].
    Returns (cv, cc).
    """
    f = jnp.sinh

    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)

    case2_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case2_cc = f(x)

    sec_neg = _secant(f, x, lb, 0.0, fallback=-jnp.inf)  # concave-half cv role (x < 0)
    sec_pos = _secant(f, x, 0.0, ub)  # convex-half cc role (x >= 0)
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))

    is_convex = lb >= 0
    is_concave = ub <= 0

    cv = jnp.where(is_convex, case1_cv, jnp.where(is_concave, case2_cv, case3_cv))
    cc = jnp.where(is_convex, case1_cc, jnp.where(is_concave, case2_cc, case3_cc))
    return cv, cc


def relax_cosh(x, lb, ub):
    """McCormick relaxation of cosh(x) on [lb, ub].

    cosh is convex everywhere.
    Returns (cv, cc).
    """
    cv = jnp.cosh(x)
    cc = _secant(jnp.cosh, x, lb, ub)
    return cv, cc


def relax_tanh(x, lb, ub):
    """McCormick relaxation of tanh(x) on [lb, ub].

    tanh is concave on [0, inf) and convex on (-inf, 0].
    Returns (cv, cc).
    """
    f = jnp.tanh

    case1_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case1_cc = f(x)

    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)

    sec_neg = _secant(f, x, lb, 0.0)  # convex-half cc role (x < 0)
    sec_pos = _secant(f, x, 0.0, ub, fallback=-jnp.inf)  # concave-half cv role (x >= 0)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)

    is_concave = lb >= 0
    is_convex = ub <= 0

    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def relax_sigmoid(x, lb, ub):
    """McCormick relaxation of sigmoid(x) = 1/(1+exp(-x)) on [lb, ub].

    sigmoid is concave on [0, inf) and convex on (-inf, 0].
    Returns (cv, cc).
    """
    import jax.nn as jnn

    f = jnn.sigmoid

    # Case 1: lb >= 0 → concave region
    case1_cv = _secant(f, x, lb, ub, fallback=-jnp.inf)
    case1_cc = f(x)

    # Case 2: ub <= 0 → convex region
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)

    # Case 3: lb < 0 < ub → mixed
    sec_neg = _secant(f, x, lb, 0.0)  # convex-half cc role (x < 0)
    sec_pos = _secant(f, x, 0.0, ub, fallback=-jnp.inf)  # concave-half cv role (x >= 0)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)

    is_concave = lb >= 0
    is_convex = ub <= 0

    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def relax_softplus(x, lb, ub):
    """McCormick relaxation of softplus(x) = log(1+exp(x)) on [lb, ub].

    softplus is convex everywhere.
    Returns (cv, cc).
    """
    f = lambda t: jnp.logaddexp(t, 0.0)  # noqa: E731

    cv = f(x)
    cc = _secant(f, x, lb, ub)
    return cv, cc


def relax_entropy(x, lb, ub):
    """McCormick relaxation of entropy(x) = x*log(x) on [lb, ub], lb >= 0.

    entropy''(x) = 1/x > 0 for x > 0, so entropy is convex on its domain:
    cv = x*log(x), cc = secant line. The argument is clamped to a tiny positive
    value inside the log so the underestimator stays finite at x -> 0+ (where
    x*log(x) -> 0). Valid only for nonnegative arguments.
    Returns (cv, cc).
    """
    f = lambda t: t * jnp.log(jnp.maximum(t, 1e-300))  # noqa: E731

    cv = f(x)
    cc = _secant(f, x, lb, ub)
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

    cv = jnp.where(lb > 0, 1.0, jnp.where(ub < 0, -1.0, jnp.where(lb == 0, 0.0, -1.0)))

    cc = jnp.where(lb > 0, 1.0, jnp.where(ub < 0, -1.0, jnp.where(ub == 0, 0.0, 1.0)))

    # Tighten: ensure cv <= sign(x) <= cc
    # The constant bounds above are always sound.
    # We can tighten by using sign(x) where it's exact.
    cv = jnp.where(lb > 0, sign_x, jnp.where(ub < 0, sign_x, cv))
    cc = jnp.where(lb > 0, sign_x, jnp.where(ub < 0, sign_x, cc))

    return cv, cc


def relax_min(x, y, cv_x, cc_x, cv_y, cc_y):
    """McCormick relaxation of min(x, y).

    ``min`` is concave. Using the identity ``min(a, b) = 0.5*(a + b - |a - b|)``:

    - **cc** (concave overestimator): ``min(cc_x, cc_y)``. The minimum of two
      concave functions is concave, and ``cc_x >= x, cc_y >= y`` give
      ``min(cc_x, cc_y) >= min(x, y)``. Valid and concave.
    - **cv** (convex underestimator): ``0.5*(cv_x + cv_y - S)`` where ``S`` is the
      *concave* (affine secant) overestimator of ``|a - b|`` evaluated at the
      actual underestimator difference ``cv_x - cv_y``. Since
      ``S >= |cv_x - cv_y|`` we get ``cv <= 0.5*(cv_x + cv_y - |cv_x - cv_y|) =
      min(cv_x, cv_y) <= min(x, y)`` -- sound -- while ``-S`` (concave) keeps the
      whole expression convex.

    The earlier implementation used ``cv = min(cv_x, cv_y)``: a valid *value*
    bound, but the pointwise minimum of two convex functions is **not convex**,
    so a local relaxation solve over that nonconvex feasible set yields invalid
    (too-tight) bounds -- pruning true optima in spatial B&B. See issue #27a.

    Returns (cv, cc).
    """
    cc = jnp.minimum(cc_x, cc_y)
    # Concave (affine secant) overestimator of |a - b| over [cv_x - cc_y,
    # cc_x - cv_y], evaluated at the actual underestimator difference
    # cv_x - cv_y (which lies inside that interval).  Evaluating the secant at
    # the *actual* difference -- not the interval midpoint -- is what keeps the
    # result sound: S >= |cv_x - cv_y|.  See issue #27a.
    cv_d = cv_x - cc_y
    cc_d = cc_x - cv_y
    _, cc_abs = relax_abs(cv_x - cv_y, cv_d, cc_d)
    cv = 0.5 * (cv_x + cv_y - cc_abs)
    return cv, cc


def relax_max(x, y, cv_x, cc_x, cv_y, cc_y):
    """McCormick relaxation of max(x, y).

    ``max`` is convex. Using the identity ``max(a, b) = 0.5*(a + b + |a - b|)``:

    - **cv** (convex underestimator): ``max(cv_x, cv_y)``. The maximum of two
      convex functions is convex, and ``cv_x <= x, cv_y <= y`` give
      ``max(cv_x, cv_y) <= max(x, y)``. Valid and convex.
    - **cc** (concave overestimator): ``0.5*(cc_x + cc_y + S)`` where ``S`` is the
      *concave* (affine secant) overestimator of ``|a - b|`` evaluated at the
      actual overestimator difference ``cc_x - cc_y``. Since
      ``S >= |cc_x - cc_y|`` we get ``cc >= 0.5*(cc_x + cc_y + |cc_x - cc_y|) =
      max(cc_x, cc_y) >= max(x, y)`` -- sound -- while ``+S`` (concave) keeps the
      whole expression concave.

    The earlier implementation used ``cc = max(cc_x, cc_y)``: a valid *value*
    bound, but the pointwise maximum of two concave functions is **not
    concave**, so a local relaxation solve over that nonconvex feasible set
    yields invalid bounds -- pruning true optima in spatial B&B (e.g.
    ``w == max(...)`` from ``if_else``). See issue #27a.

    Returns (cv, cc).
    """
    cv = jnp.maximum(cv_x, cv_y)
    # Concave (affine secant) overestimator of |a - b| over [cv_x - cc_y,
    # cc_x - cv_y], evaluated at the actual overestimator difference
    # cc_x - cc_y (which lies inside that interval).  Evaluating the secant at
    # the *actual* difference -- not the interval midpoint -- is what keeps the
    # result sound: S >= |cc_x - cc_y|.  See issue #27a.
    cv_d = cv_x - cc_y
    cc_d = cc_x - cv_y
    _, cc_abs = relax_abs(cc_x - cc_y, cv_d, cc_d)
    cc = 0.5 * (cc_x + cc_y + cc_abs)
    return cv, cc
