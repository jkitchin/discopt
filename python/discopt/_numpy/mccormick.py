"""Pure-numpy McCormick relaxation primitives.

Ported from ``discopt._jax.mccormick``. All functions have identical
mathematical behavior; only the numerical backend changes. Avoids the
JAX trace/compile floor that dominates time on small B&B nodes.
"""

from __future__ import annotations

import numpy as jnp


def _secant(f, x, lb, ub):
    f_lb = f(lb)
    f_ub = f(ub)
    slope = (f_ub - f_lb) / (ub - lb)
    line = f_lb + slope * (x - lb)
    return jnp.where(jnp.abs(ub - lb) < 1e-15, f(x), line)


def relax_bilinear(x, y, x_lb, x_ub, y_lb, y_ub):
    cv1 = x_lb * y + x * y_lb - x_lb * y_lb
    cv2 = x_ub * y + x * y_ub - x_ub * y_ub
    cv = jnp.maximum(cv1, cv2)
    cc1 = x_ub * y + x * y_lb - x_ub * y_lb
    cc2 = x_lb * y + x * y_ub - x_lb * y_ub
    cc = jnp.minimum(cc1, cc2)
    both_nonneg = (x_lb >= 0.0) & (y_lb >= 0.0)
    cv = jnp.where(both_nonneg, jnp.maximum(cv, x_lb * y_lb), cv)
    return cv, cc


def relax_add(cv_x, cc_x, cv_y, cc_y):
    return cv_x + cv_y, cc_x + cc_y


def relax_sub(cv_x, cc_x, cv_y, cc_y):
    return cv_x - cc_y, cc_x - cv_y


def relax_neg(cv_x, cc_x):
    return -cc_x, -cv_x


def _relax_reciprocal(y, y_lb, y_ub):
    def f(t):
        return 1.0 / t

    cv = f(y)
    cc = _secant(f, y, y_lb, y_ub)
    return cv, cc


def relax_div(x, y, x_lb, x_ub, y_lb, y_ub):
    recip_cv, _recip_cc = _relax_reciprocal(y, y_lb, y_ub)
    recip_lb = 1.0 / y_ub
    recip_ub = 1.0 / y_lb
    recip_lb_sorted = jnp.minimum(recip_lb, recip_ub)
    recip_ub_sorted = jnp.maximum(recip_lb, recip_ub)
    return relax_bilinear(x, recip_cv, x_lb, x_ub, recip_lb_sorted, recip_ub_sorted)


def relax_pow(x, lb, ub, n):
    def f(t):
        return t**n

    if n == 1:
        return x, x

    if n % 2 == 0:
        cv = f(x)
        cc = _secant(f, x, lb, ub)
        return cv, cc

    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)
    case2_cv = _secant(f, x, lb, ub)
    case2_cc = f(x)

    zero = jnp.zeros_like(x)
    sec_neg = _secant(f, x, lb, zero)
    sec_pos = _secant(f, x, zero, ub)
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))

    is_nonneg = lb >= 0
    is_nonpos = ub <= 0
    cv = jnp.where(is_nonneg, case1_cv, jnp.where(is_nonpos, case2_cv, case3_cv))
    cc = jnp.where(is_nonneg, case1_cc, jnp.where(is_nonpos, case2_cc, case3_cc))
    return cv, cc


def relax_exp(x, lb, ub):
    cv = jnp.exp(x)
    cc = _secant(jnp.exp, x, lb, ub)
    return cv, cc


def relax_square(x, lb, ub):
    def f(t):
        return t**2

    cv = f(x)
    cc = _secant(f, x, lb, ub)
    return cv, cc


def relax_abs(x, lb, ub):
    cv = jnp.abs(x)
    cc_secant = _secant(jnp.abs, x, lb, ub)
    contains_zero = (lb < 0) & (ub > 0)
    cc = jnp.where(contains_zero, cc_secant, jnp.abs(x))
    return cv, cc


def relax_sqrt(x, lb, ub):
    cc = jnp.sqrt(x)
    cv = _secant(jnp.sqrt, x, lb, ub)
    return cv, cc


def relax_log(x, lb, ub):
    cc = jnp.log(x)
    cv = _secant(jnp.log, x, lb, ub)
    return cv, cc


def relax_log2(x, lb, ub):
    cc = jnp.log2(x)
    cv = _secant(jnp.log2, x, lb, ub)
    return cv, cc


def relax_log10(x, lb, ub):
    cc = jnp.log10(x)
    cv = _secant(jnp.log10, x, lb, ub)
    return cv, cc


def relax_sin(x, lb, ub):
    wide = (ub - lb) >= 2.0 * jnp.pi
    sin_lb = jnp.sin(lb)
    sin_ub = jnp.sin(ub)
    sin_x = jnp.sin(x)
    k_min_max = jnp.ceil((lb - jnp.pi / 2) / (2 * jnp.pi))
    max_point = jnp.pi / 2 + k_min_max * 2 * jnp.pi
    has_max = max_point <= ub
    k_min_min = jnp.ceil((lb + jnp.pi / 2) / (2 * jnp.pi))
    min_point = -jnp.pi / 2 + k_min_min * 2 * jnp.pi
    has_min = min_point <= ub
    sin_range_max = jnp.where(has_max, 1.0, jnp.maximum(sin_lb, sin_ub))
    sin_range_min = jnp.where(has_min, -1.0, jnp.minimum(sin_lb, sin_ub))
    sec = _secant(jnp.sin, x, lb, ub)
    concave_cv = sec
    concave_cc = sin_x
    convex_cv = sin_x
    convex_cc = sec
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
    return relax_sin(x + jnp.pi / 2, lb + jnp.pi / 2, ub + jnp.pi / 2)


def relax_tan(x, lb, ub):
    f = jnp.tan
    mid = 0.5 * (lb + ub)
    k = jnp.round(mid / jnp.pi)
    center = k * jnp.pi
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)
    case2_cv = _secant(f, x, lb, ub)
    case2_cc = f(x)
    sec_neg = _secant(f, x, lb, center)
    sec_pos = _secant(f, x, center, ub)
    case3_cv = jnp.where(x >= center, f(x), sec_neg)
    case3_cc = jnp.where(x >= center, sec_pos, f(x))
    is_convex_half = lb >= center
    is_concave_half = ub <= center
    cv = jnp.where(is_convex_half, case1_cv, jnp.where(is_concave_half, case2_cv, case3_cv))
    cc = jnp.where(is_convex_half, case1_cc, jnp.where(is_concave_half, case2_cc, case3_cc))
    return cv, cc


def relax_atan(x, lb, ub):
    f = jnp.arctan
    case1_cv = _secant(f, x, lb, ub)
    case1_cc = f(x)
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)
    sec_neg = _secant(f, x, lb, 0.0)
    sec_pos = _secant(f, x, 0.0, ub)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)
    is_concave = lb >= 0
    is_convex = ub <= 0
    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def relax_asin(x, lb, ub):
    # asin''(x) = x*(1-x**2)**(-3/2): convex on [0,1], concave on [-1,0]
    # (mirror of acos; same curvature layout as sinh). See C-32.
    f = jnp.arcsin
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)
    case2_cv = _secant(f, x, lb, ub)
    case2_cc = f(x)
    sec_neg = _secant(f, x, lb, 0.0)
    sec_pos = _secant(f, x, 0.0, ub)
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))
    is_convex = lb >= 0
    is_concave = ub <= 0
    cv = jnp.where(is_convex, case1_cv, jnp.where(is_concave, case2_cv, case3_cv))
    cc = jnp.where(is_convex, case1_cc, jnp.where(is_concave, case2_cc, case3_cc))
    return cv, cc


def relax_acos(x, lb, ub):
    # acos''(x) = -x*(1-x**2)**(-3/2): concave on [0,1], convex on [-1,0]
    # (mirror of asin; same curvature layout as tanh). See C-32.
    f = jnp.arccos
    case1_cv = _secant(f, x, lb, ub)
    case1_cc = f(x)
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)
    sec_neg = _secant(f, x, lb, 0.0)
    sec_pos = _secant(f, x, 0.0, ub)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)
    is_concave = lb >= 0
    is_convex = ub <= 0
    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def relax_sinh(x, lb, ub):
    f = jnp.sinh
    case1_cv = f(x)
    case1_cc = _secant(f, x, lb, ub)
    case2_cv = _secant(f, x, lb, ub)
    case2_cc = f(x)
    sec_neg = _secant(f, x, lb, 0.0)
    sec_pos = _secant(f, x, 0.0, ub)
    case3_cv = jnp.where(x >= 0, f(x), sec_neg)
    case3_cc = jnp.where(x >= 0, sec_pos, f(x))
    is_convex = lb >= 0
    is_concave = ub <= 0
    cv = jnp.where(is_convex, case1_cv, jnp.where(is_concave, case2_cv, case3_cv))
    cc = jnp.where(is_convex, case1_cc, jnp.where(is_concave, case2_cc, case3_cc))
    return cv, cc


def relax_cosh(x, lb, ub):
    cv = jnp.cosh(x)
    cc = _secant(jnp.cosh, x, lb, ub)
    return cv, cc


def relax_tanh(x, lb, ub):
    f = jnp.tanh
    case1_cv = _secant(f, x, lb, ub)
    case1_cc = f(x)
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)
    sec_neg = _secant(f, x, lb, 0.0)
    sec_pos = _secant(f, x, 0.0, ub)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)
    is_concave = lb >= 0
    is_convex = ub <= 0
    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def relax_sigmoid(x, lb, ub):
    f = _sigmoid
    case1_cv = _secant(f, x, lb, ub)
    case1_cc = f(x)
    case2_cv = f(x)
    case2_cc = _secant(f, x, lb, ub)
    sec_neg = _secant(f, x, lb, 0.0)
    sec_pos = _secant(f, x, 0.0, ub)
    case3_cv = jnp.where(x >= 0, sec_pos, f(x))
    case3_cc = jnp.where(x >= 0, f(x), sec_neg)
    is_concave = lb >= 0
    is_convex = ub <= 0
    cv = jnp.where(is_concave, case1_cv, jnp.where(is_convex, case2_cv, case3_cv))
    cc = jnp.where(is_concave, case1_cc, jnp.where(is_convex, case2_cc, case3_cc))
    return cv, cc


def relax_softplus(x, lb, ub):
    def f(t):
        return jnp.logaddexp(t, 0.0)

    cv = f(x)
    cc = _secant(f, x, lb, ub)
    return cv, cc


def relax_sign(x, lb, ub):
    sign_x = jnp.sign(x)
    cv = jnp.where(lb > 0, 1.0, jnp.where(ub < 0, -1.0, jnp.where(lb == 0, 0.0, -1.0)))
    cc = jnp.where(lb > 0, 1.0, jnp.where(ub < 0, -1.0, jnp.where(ub == 0, 0.0, 1.0)))
    cv = jnp.where(lb > 0, sign_x, jnp.where(ub < 0, sign_x, cv))
    cc = jnp.where(lb > 0, sign_x, jnp.where(ub < 0, sign_x, cc))
    return cv, cc


def relax_min(x, y, cv_x, cc_x, cv_y, cc_y):
    # min(a,b) = 0.5*(a + b - |a-b|).  cc = min(cc_x, cc_y) is concave (min of
    # concave).  cv must be convex, so subtract the *concave* affine-secant
    # overestimator S of |a-b| (S >= |a-b|) rather than using min(cv_x, cv_y)
    # (non-convex -> invalid spatial-B&B bounds).  Evaluate the secant at the
    # actual underestimator difference cv_x - cv_y (inside [cv_x-cc_y, cc_x-cv_y])
    # -- NOT the interval midpoint -- so S >= |cv_x - cv_y| and cv stays sound.
    # See issue #27a and discopt._jax.mccormick.
    cc = jnp.minimum(cc_x, cc_y)
    cv_d = cv_x - cc_y
    cc_d = cc_x - cv_y
    _, cc_abs = relax_abs(cv_x - cv_y, cv_d, cc_d)
    cv = 0.5 * (cv_x + cv_y - cc_abs)
    return cv, cc


def relax_max(x, y, cv_x, cc_x, cv_y, cc_y):
    # max(a,b) = 0.5*(a + b + |a-b|).  cv = max(cv_x, cv_y) is convex (max of
    # convex).  cc must be concave, so add the *concave* affine-secant
    # overestimator S of |a-b| (S >= |a-b|) rather than using max(cc_x, cc_y)
    # (non-concave -> invalid spatial-B&B bounds).  Evaluate the secant at the
    # actual overestimator difference cc_x - cc_y (inside [cv_x-cc_y, cc_x-cv_y])
    # -- NOT the interval midpoint -- so S >= |cc_x - cc_y| and cc stays sound.
    # See issue #27a and discopt._jax.mccormick.
    cv = jnp.maximum(cv_x, cv_y)
    cv_d = cv_x - cc_y
    cc_d = cc_x - cv_y
    _, cc_abs = relax_abs(cc_x - cc_y, cv_d, cc_d)
    cc = 0.5 * (cc_x + cc_y + cc_abs)
    return cv, cc
