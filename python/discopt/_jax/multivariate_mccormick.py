"""
Multivariate McCormick relaxations (Tsoukalas and Mitsos, 2014).

This module implements the tightened univariate composition rule from

    A. Tsoukalas and A. Mitsos, "Multivariate McCormick Relaxations,"
    Journal of Global Optimization 59(2), 633-662, 2014.

For a composition ``f(g(x))`` where ``g`` has McCormick relaxations
``cv_g(x) <= g(x) <= cc_g(x)`` over a domain in which ``g(x)`` lies in
``[g_lb, g_ub]``, the classical McCormick composition rule is

    cv(x) = f^{cv}(z_cv(x)),     z_cv(x) = mid(cv_g(x), cc_g(x), z_cv_star)
    cc(x) = f^{cc}(z_cc(x)),     z_cc(x) = mid(cv_g(x), cc_g(x), z_cc_star)

where ``f^{cv}`` and ``f^{cc}`` are the convex/concave envelopes of ``f`` on
``[g_lb, g_ub]``, ``z_cv_star`` minimises ``f^{cv}`` on that interval,
``z_cc_star`` maximises ``f^{cc}``, and ``mid(a, b, c)`` is the median of the
three values. When ``cv_g <= cc_g`` (which always holds for valid relaxations),
``mid(cv_g, cc_g, z) = clip(z, cv_g, cc_g)``.

This rule is provably:
    1. Sound: ``cv(x) <= f(g(x)) <= cc(x)`` for every ``x``.
    2. Convexity-preserving: ``cv(x)`` is convex in ``x`` and ``cc(x)`` is
       concave.
    3. Strictly tighter than the legacy midpoint composition rule
       (``relax_f(0.5*(cv_g+cc_g), cv_g, cc_g)``) whenever ``cv_g < cc_g``.

All functions are pure JAX and compatible with ``jax.jit``, ``jax.grad``, and
``jax.vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _safe_slope(f_lb, f_ub, g_lb, g_ub):
    """Slope of the secant line from (g_lb, f_lb) to (g_ub, f_ub)."""
    width = jnp.where(jnp.abs(g_ub - g_lb) < 1e-15, 1.0, g_ub - g_lb)
    return (f_ub - f_lb) / width


def clip_inner(t, lo, hi):
    """``jnp.clip(t, lo, hi)`` whose derivative **w.r.t. t** is 1 on the CLOSED
    interval ``[lo, hi]`` (and 0 strictly outside), instead of the 0.5 that
    ``jnp.clip`` returns at the exact boundary.

    Why this matters for soundness: the composition rules feed ``clip(cv_g, g_lb,
    g_ub)`` into a convex/concave envelope, and MCBox differentiates that composite to
    get the McCormick **subgradient** (:func:`discopt._jax.mcbox._univariate_kernel`).
    ``jnp.clip`` is ``min(max(t, lo), hi)``; at ``t == lo`` or ``t == hi`` the max/min
    tie makes ``jax.grad`` return the *average* (0.5) subderivative. On the interior
    that never fires, but a Kelley/LP iterate sits exactly on a **box face**, where
    ``cv_g`` equals ``g_lb``/``g_ub`` — there the halved factor yields an INVALID
    subgradient (the tangent over-/under-slopes the envelope and its cut can exclude the
    true optimum → too-high bound → false fathom). Since ``cv_g``/``cc_g`` are always
    within ``[g_lb, g_ub]`` for a valid relaxation, the valid subdifferential element at
    a face is the *inner one-sided* derivative (slope 1), which this returns. The value
    is identical to ``jnp.clip`` — only the boundary gradient is corrected."""
    ct = jnp.clip(t, lo, hi)
    inside = (t >= lo) & (t <= hi)
    through = t + jax.lax.stop_gradient(ct - t)  # value ct, d/dt = 1
    return jnp.where(inside, through, jax.lax.stop_gradient(ct))


def clip_into(z, lo, hi):
    """``jnp.clip(z, lo, hi)`` for a constant ``z`` clamped into the data-dependent
    ``[lo, hi]`` (the ``mid(z_star, cv_g, cc_g)`` selection in the square/abs/pow
    rules), whose derivative **w.r.t. the active bound** is 1 at the boundary tie.

    When ``z <= lo`` the lower bound is active (``d/dlo = 1``); when ``z >= hi`` the
    upper bound is active (``d/dhi = 1``); strictly inside, ``z`` is a constant so both
    are 0. At the exact tie ``z == lo`` (or ``z == hi``) ``jnp.clip`` would split the
    gradient 0.5/0.5; the inner element (the active bound taking full slope 1) is the
    valid subdifferential choice — same reasoning as :func:`clip_inner`. Value
    identical to ``jnp.clip``."""
    cz = jnp.clip(z, lo, hi)
    lo_active = z <= lo
    hi_active = z >= hi
    # value cz; d/dlo = 1 where lo active, d/dhi = 1 where hi active, else 0.
    val = jnp.where(
        lo_active,
        lo + jax.lax.stop_gradient(cz - lo),
        jnp.where(hi_active, hi + jax.lax.stop_gradient(cz - hi), jax.lax.stop_gradient(cz)),
    )
    return val


def _secant_at(z, g_lb, f_lb, slope):
    """Value of the secant line ``f_lb + slope * (z - g_lb)`` at ``z``."""
    return f_lb + slope * (z - g_lb)


# ---------------------------------------------------------------------------
# Monotone convex functions: f convex, f' >= 0 on [g_lb, g_ub]
# (e.g. exp, cosh on [0, inf), softplus, exp on any interval)
# ---------------------------------------------------------------------------


def _compose_monotone_convex_inc(f, cv_g, cc_g, g_lb, g_ub):
    """Composition rule for monotone-increasing convex ``f`` on ``[g_lb, g_ub]``.

    f^{cv} = f, argmin = g_lb -> z_cv = clip(g_lb, cv_g, cc_g) = cv_g (when cv_g >= g_lb)
    f^{cc} = secant, argmax = g_ub -> z_cc = clip(g_ub, cv_g, cc_g) = cc_g (when cc_g <= g_ub)
    """
    z_cv = clip_inner(cv_g, g_lb, g_ub)
    z_cc = clip_inner(cc_g, g_lb, g_ub)
    f_lb, f_ub = f(g_lb), f(g_ub)
    slope = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    cv = f(z_cv)
    cc = _secant_at(z_cc, g_lb, f_lb, slope)
    return cv, cc


# ---------------------------------------------------------------------------
# Monotone concave functions: f concave, f' >= 0 on [g_lb, g_ub]
# (e.g. log, sqrt, atan on [0, inf))
# ---------------------------------------------------------------------------


def _compose_monotone_concave_inc(f, cv_g, cc_g, g_lb, g_ub):
    """Composition rule for monotone-increasing concave ``f`` on ``[g_lb, g_ub]``.

    f^{cv} = secant, argmin = g_lb (secant is increasing) -> z_cv = cv_g
    f^{cc} = f, argmax = g_ub -> z_cc = cc_g
    """
    z_cv = clip_inner(cv_g, g_lb, g_ub)
    z_cc = clip_inner(cc_g, g_lb, g_ub)
    f_lb, f_ub = f(g_lb), f(g_ub)
    slope = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    cv = _secant_at(z_cv, g_lb, f_lb, slope)
    cc = f(z_cc)
    return cv, cc


# ---------------------------------------------------------------------------
# Public composition rules (one per primitive). Each takes the relaxations of
# the inner expression ``g`` and the bounds [g_lb, g_ub] of g over the domain.
# ---------------------------------------------------------------------------


def compose_exp(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``exp(g(x))``."""
    return _compose_monotone_convex_inc(jnp.exp, cv_g, cc_g, g_lb, g_ub)


def compose_log(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``log(g(x))`` with ``g_lb > 0``."""
    return _compose_monotone_concave_inc(jnp.log, cv_g, cc_g, g_lb, g_ub)


def compose_log2(cv_g, cc_g, g_lb, g_ub):
    return _compose_monotone_concave_inc(jnp.log2, cv_g, cc_g, g_lb, g_ub)


def compose_log10(cv_g, cc_g, g_lb, g_ub):
    return _compose_monotone_concave_inc(jnp.log10, cv_g, cc_g, g_lb, g_ub)


def compose_sqrt(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``sqrt(g(x))`` with ``g_lb >= 0``."""
    return _compose_monotone_concave_inc(jnp.sqrt, cv_g, cc_g, g_lb, g_ub)


def compose_softplus(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``softplus(g(x)) = log(1 + exp(g(x)))``."""
    f = lambda t: jnp.logaddexp(t, 0.0)  # noqa: E731
    return _compose_monotone_convex_inc(f, cv_g, cc_g, g_lb, g_ub)


def compose_pow_frac(cv_g, cc_g, g_lb, g_ub, a):
    """TM2014 composition rule for the **signomial** term ``g(x) ** a`` with a
    non-integer exponent ``a`` over a **positive** base (``g_lb > 0``), P1.4.

    ``x ** a`` on ``(0, inf)`` is monotone with a single, fixed curvature per regime:

    * ``a > 1``   -> convex, increasing  (``f^cv = f`` at ``g_lb``, ``f^cc`` = secant);
    * ``0 < a < 1`` -> concave, increasing (``f^cv`` = secant, ``f^cc = f`` at ``g_ub``);
    * ``a < 0``   -> convex, **decreasing** (``f^cv = f`` at ``g_ub``, ``f^cc`` = secant).

    The mid-rule selects, for each envelope, the operand relaxation value at the
    argmin/argmax of that envelope over ``[g_lb, g_ub]`` (``g_lb`` for an increasing
    role, ``g_ub`` for a decreasing one), clamped by :func:`clip_inner` so the boundary
    subgradient is the valid inner slope (not the ``jnp.clip`` tie). Caller guarantees
    ``g_lb > 0`` (a base that can reach ``<= 0`` yields a no-information bracket upstream
    — ``x ** a`` is undefined there for non-integer ``a``)."""
    f = lambda t: t**a  # noqa: E731
    f_lb, f_ub = f(g_lb), f(g_ub)
    slope = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    z_lo = clip_inner(cv_g, g_lb, g_ub)
    z_hi = clip_inner(cc_g, g_lb, g_ub)
    sec = lambda z: f_lb + slope * (z - g_lb)  # noqa: E731
    conc_inc = (a > 0.0) & (a < 1.0)
    conv_dec = a < 0.0
    # cv:  a>1 -> f(z_lo);  0<a<1 -> sec(z_lo);  a<0 -> f(z_hi)
    cv = jnp.where(conv_dec, f(z_hi), jnp.where(conc_inc, sec(z_lo), f(z_lo)))
    # cc:  a>1 -> sec(z_hi); 0<a<1 -> f(z_hi);  a<0 -> sec(z_lo)
    cc = jnp.where(conv_dec, sec(z_lo), jnp.where(conc_inc, f(z_hi), sec(z_hi)))
    return cv, cc


def compose_square(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``g(x)^2``.

    f(z) = z^2 is convex with f^{cv} = f, f^{cc} = secant.
    argmin of f^{cv} on [g_lb, g_ub] is clip(0, g_lb, g_ub) (interior 0 if 0 in [g_lb, g_ub]).
    secant slope = g_lb + g_ub; argmax of secant is g_ub if slope > 0, g_lb otherwise.
    """
    z_star_cv = jnp.clip(0.0, g_lb, g_ub)
    z_cv = clip_into(z_star_cv, cv_g, cc_g)

    slope_sign_pos = (g_lb + g_ub) >= 0.0
    z_star_cc = jnp.where(slope_sign_pos, g_ub, g_lb)
    z_cc = clip_into(z_star_cc, cv_g, cc_g)

    f_lb = g_lb * g_lb
    f_ub = g_ub * g_ub
    slope = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    cv = z_cv * z_cv
    cc = _secant_at(z_cc, g_lb, f_lb, slope)
    return cv, cc


def compose_abs(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``|g(x)|``.

    f(z) = |z| is convex; f^{cv} = |z|, f^{cc} = secant on [g_lb, g_ub].
    argmin of |z| on [g_lb, g_ub] is clip(0, g_lb, g_ub).
    """
    z_star_cv = jnp.clip(0.0, g_lb, g_ub)
    z_cv = clip_into(z_star_cv, cv_g, cc_g)

    f_lb = jnp.abs(g_lb)
    f_ub = jnp.abs(g_ub)
    # Secant slope sign: |g_ub| >= |g_lb| iff g_lb + g_ub >= 0 (when interval contains 0).
    # In all cases, argmax of secant is g_ub if f_ub >= f_lb else g_lb.
    z_star_cc = jnp.where(f_ub >= f_lb, g_ub, g_lb)
    z_cc = clip_into(z_star_cc, cv_g, cc_g)

    slope = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    cv = jnp.abs(z_cv)
    cc = _secant_at(z_cc, g_lb, f_lb, slope)
    # When the interval doesn't contain 0, |.| is affine -> tighten cc to f(z_cc).
    contains_zero = (g_lb < 0) & (g_ub > 0)
    cc = jnp.where(contains_zero, cc, jnp.abs(z_cc))
    return cv, cc


def compose_even_pow(cv_g, cc_g, g_lb, g_ub, n: int):
    """TM2014 composition rule for ``g(x)^n`` with even integer ``n >= 2``.

    z^n is convex with minimum at 0 (an interior critical point on intervals
    that contain 0). Same shape as ``compose_square``.
    """
    if n % 2 != 0 or n < 2:
        raise ValueError(f"compose_even_pow requires even n >= 2, got {n}")

    z_star_cv = jnp.clip(0.0, g_lb, g_ub)
    z_cv = clip_into(z_star_cv, cv_g, cc_g)

    f_lb = g_lb**n
    f_ub = g_ub**n
    z_star_cc = jnp.where(f_ub >= f_lb, g_ub, g_lb)
    z_cc = clip_into(z_star_cc, cv_g, cc_g)

    slope = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    cv = z_cv**n
    cc = _secant_at(z_cc, g_lb, f_lb, slope)
    return cv, cc


def odd_pow_tangent_coeff(n: int) -> float:
    """``t*/|lb|`` for the convex-envelope tangent of ``x**n`` (odd ``n``) over a box
    ``[lb, ub]`` that spans 0 — the point ``t* in (0, |lb|)`` where the line from
    ``(lb, lb**n)`` is tangent to ``x**n``. By homogeneity ``t* = c_n · |lb|`` with
    ``c_n`` depending only on ``n``; ``c_n`` solves ``n t^{n-1}(t+1) - (t^n + 1) = 0``
    on ``(0, 1)`` (n=3 -> 1/2). Static ``n`` -> a plain Newton solve at build time
    (Liberti & Pantelides-style monomial convex hull)."""
    if n % 2 == 0 or n < 3:
        raise ValueError(f"odd_pow_tangent_coeff requires odd n >= 3, got {n}")
    t = 0.5
    for _ in range(80):
        h = n * t ** (n - 1) * (t + 1.0) - (t**n + 1.0)
        hp = n * (n - 1) * t ** (n - 2) * (t + 1.0)  # d/dt: the t^{n-1} terms cancel
        step = h / hp
        t -= step
        if abs(step) < 1e-15:
            break
    return float(t)


def _odd_cv_env(z, g_lb, g_ub, n, cn):
    """Convex envelope of ``x**n`` (odd ``n``) on ``[g_lb, g_ub]``, evaluated at ``z``.

    * ``g_lb >= 0`` (convex regime): ``x**n``;
    * ``g_ub <= 0`` (concave regime): the secant chord;
    * spanning ``g_lb < 0 < g_ub``: the line from ``(g_lb, g_lb**n)`` tangent at
      ``t* = -g_lb·cn`` for ``z <= t*`` then ``x**n`` for ``z >= t*`` — unless ``t*``
      falls beyond ``g_ub`` (box too far negative), where the envelope is the chord.
    """
    fz = z**n
    f_lb, f_ub = g_lb**n, g_ub**n
    slope_sec = _safe_slope(f_lb, f_ub, g_lb, g_ub)
    sec = f_lb + slope_sec * (z - g_lb)
    tstar = -g_lb * cn  # tangent point (>0 when g_lb<0)
    m = n * tstar ** (n - 1)  # tangent slope; n-1 even -> real for any tstar sign
    tang = f_lb + m * (z - g_lb)
    spanning = jnp.where(z <= tstar, tang, fz)
    spanning = jnp.where(tstar <= g_ub, spanning, sec)  # tangent beyond box -> chord
    return jnp.where(g_lb >= 0.0, fz, jnp.where(g_ub <= 0.0, sec, spanning))


def compose_odd_pow(cv_g, cc_g, g_lb, g_ub, n: int, cn: float):
    """TM2014 composition rule for ``g(x)**n`` with odd integer ``n >= 3`` — the TIGHT
    monomial hull (convex/concave envelope), valid over sign-spanning boxes where the
    naive ``relax_pow`` cv is non-convex. ``x**n`` (odd) is monotone increasing, so the
    convex envelope's argmin is ``g_lb`` (-> use ``cv_g``) and the concave envelope's
    argmax is ``g_ub`` (-> use ``cc_g``). The concave envelope follows by antisymmetry
    (``x**n`` is odd): ``cc_env(z) = -cv_env(-z)`` on the reflected box. Boundary
    subgradients via ``clip_inner``; the tangent point ``t*`` is a per-box constant so
    the kernel-chain derivative sees the active piece's slope."""
    if n % 2 == 0 or n < 3:
        raise ValueError(f"compose_odd_pow requires odd n >= 3, got {n}")
    z_cv = clip_inner(cv_g, g_lb, g_ub)
    z_cc = clip_inner(cc_g, g_lb, g_ub)
    cv = _odd_cv_env(z_cv, g_lb, g_ub, n, cn)
    cc = -_odd_cv_env(-z_cc, -g_ub, -g_lb, n, cn)  # antisymmetry of the odd monomial
    return cv, cc


# ---------------------------------------------------------------------------
# Sigmoidal functions: f has a single inflection point at the origin and is
# globally monotone-increasing. On intervals that don't span the inflection,
# they are purely convex (left half) or purely concave (right half).
# ---------------------------------------------------------------------------


def _compose_sigmoid_like(f, cv_g, cc_g, g_lb, g_ub, *, concave_on_right: bool):
    """Composition rule for monotone-increasing sigmoidal ``f`` with inflection at 0.

    ``concave_on_right=True``  -> f is concave on [0, inf) and convex on (-inf, 0]
        (e.g. tanh, atan, sigmoid).
    ``concave_on_right=False`` -> f is convex on [0, inf) and concave on (-inf, 0]
        (e.g. sinh, asinh).

    On intervals fully on one side of 0, this reduces to either the monotone
    convex or monotone concave rule above. On intervals spanning 0, falls back
    to a sound enclosure via the convex/concave envelopes constructed from the
    two halves; for the initial M1 implementation we use the secant for the
    looser side and ``f(z)`` for the tighter side, which is sound but loses the
    TM2014 tightening on the spanning case. A full envelope construction
    (root-finding for the tangent point on the convex/concave half) is left as
    a follow-up — see issue #51 (M1).
    """
    f_lb_v, f_ub_v = f(g_lb), f(g_ub)
    slope = _safe_slope(f_lb_v, f_ub_v, g_lb, g_ub)

    z_cv_inner = clip_inner(cv_g, g_lb, g_ub)
    z_cc_inner = clip_inner(cc_g, g_lb, g_ub)

    # Pure-half cases (TM2014 tight)
    if concave_on_right:
        # f concave on right -> on [g_lb >= 0]: monotone concave-inc rule.
        # On [g_ub <= 0]: monotone convex-inc rule (f is convex there and inc).
        cv_concave_right = _secant_at(z_cv_inner, g_lb, f_lb_v, slope)
        cc_concave_right = f(z_cc_inner)
        cv_convex_left = f(z_cv_inner)
        cc_convex_left = _secant_at(z_cc_inner, g_lb, f_lb_v, slope)
    else:
        cv_convex_right = f(z_cv_inner)
        cc_convex_right = _secant_at(z_cc_inner, g_lb, f_lb_v, slope)
        cv_concave_left = _secant_at(z_cv_inner, g_lb, f_lb_v, slope)
        cc_concave_left = f(z_cc_inner)

    # Spanning case (sound but conservative): use the wider of {f, secant} for
    # each side. Specifically, for a sigmoidal monotone-increasing function:
    #   cv >= min(f(z_cv_inner), secant(z_cv_inner))  is sound.
    #   cc <= max(f(z_cc_inner), secant(z_cc_inner))  is sound.
    f_at_cv = f(z_cv_inner)
    f_at_cc = f(z_cc_inner)
    sec_at_cv = _secant_at(z_cv_inner, g_lb, f_lb_v, slope)
    sec_at_cc = _secant_at(z_cc_inner, g_lb, f_lb_v, slope)
    cv_span = jnp.minimum(f_at_cv, sec_at_cv)
    cc_span = jnp.maximum(f_at_cc, sec_at_cc)

    if concave_on_right:
        cv = jnp.where(g_lb >= 0, cv_concave_right, jnp.where(g_ub <= 0, cv_convex_left, cv_span))
        cc = jnp.where(g_lb >= 0, cc_concave_right, jnp.where(g_ub <= 0, cc_convex_left, cc_span))
    else:
        cv = jnp.where(g_lb >= 0, cv_convex_right, jnp.where(g_ub <= 0, cv_concave_left, cv_span))
        cc = jnp.where(g_lb >= 0, cc_convex_right, jnp.where(g_ub <= 0, cc_concave_left, cc_span))
    return cv, cc


def compose_tanh(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``tanh(g(x))``."""
    return _compose_sigmoid_like(jnp.tanh, cv_g, cc_g, g_lb, g_ub, concave_on_right=True)


def compose_atan(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``atan(g(x))``."""
    return _compose_sigmoid_like(jnp.arctan, cv_g, cc_g, g_lb, g_ub, concave_on_right=True)


def compose_sigmoid(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``sigmoid(g(x)) = 1/(1+exp(-g(x)))``."""
    import jax.nn as jnn

    return _compose_sigmoid_like(jnn.sigmoid, cv_g, cc_g, g_lb, g_ub, concave_on_right=True)


def compose_sinh(cv_g, cc_g, g_lb, g_ub):
    """TM2014 composition rule for ``sinh(g(x))``."""
    return _compose_sigmoid_like(jnp.sinh, cv_g, cc_g, g_lb, g_ub, concave_on_right=False)


# ---------------------------------------------------------------------------
# Dispatch table: maps function name to composition rule.
# Used by relaxation_compiler to wire in the tightened composition path.
# ---------------------------------------------------------------------------


_COMPOSITION_RULES = {
    "exp": compose_exp,
    "log": compose_log,
    "log2": compose_log2,
    "log10": compose_log10,
    "sqrt": compose_sqrt,
    "softplus": compose_softplus,
    "abs": compose_abs,
    "tanh": compose_tanh,
    "atan": compose_atan,
    "sigmoid": compose_sigmoid,
    "sinh": compose_sinh,
}


def get_composition_rule(name: str):
    """Return the TM2014 composition rule for ``name``, or ``None`` if not implemented."""
    return _COMPOSITION_RULES.get(name)
