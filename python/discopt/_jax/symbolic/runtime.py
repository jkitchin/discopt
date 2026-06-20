"""Sympy-free runtime envelope assembly shared by code-gen and domain packs.

This module contains the pure-JAX construction of convex/concave envelopes from
a function ``f`` and its derivative ``f'`` (supplied as JAX callables) plus the
curvature data. It is imported on the solver hot path; it does **not** import
SymPy. The SymPy code generator (:mod:`discopt._jax.symbolic.codegen`) lambdifies
``f``/``f'`` and calls these helpers, and the hand-written domain packs
(:mod:`discopt._jax.symbolic.domains`) call them with hand-written ``f``/``f'`` —
so a single, tested envelope construction backs both paths.

Contract for every helper: ``(x, lb, ub) -> (cv, cc)`` with ``cv <= f(x) <= cc``
on ``[lb, ub]``, ``cv`` convex and ``cc`` concave, all branching via
:func:`jax.numpy.where` (jit/grad/vmap safe).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

_DEGENERATE = 1e-15
# Bisection iterations for the numeric tangent-point fallback.
_BISECTION_ITERS = 60


def secant(f: Callable, x, lb, ub):
    """Secant line of ``f`` between ``(lb, f(lb))`` and ``(ub, f(ub))`` at ``x``."""
    f_lb, f_ub = f(lb), f(ub)
    denom = jnp.where(jnp.abs(ub - lb) < _DEGENERATE, 1.0, ub - lb)
    line = f_lb + (f_ub - f_lb) / denom * (x - lb)
    return jnp.where(jnp.abs(ub - lb) < _DEGENERATE, f(x), line)


def convex_envelope(x, lb, ub, *, f: Callable):
    """Envelope of a convex ``f``: ``cv = f``, ``cc = secant``."""
    return f(x), secant(f, x, lb, ub)


def concave_envelope(x, lb, ub, *, f: Callable):
    """Envelope of a concave ``f``: ``cv = secant``, ``cc = f``."""
    return secant(f, x, lb, ub), f(x)


def _numeric_tangent_point(f, fp, c, e, lb, ub, positive_side: bool):
    """Bisect ``g(t)=f'(t)(t-e)-(f(t)-f(e))=0`` on the indicated branch.

    Returns ``(t, has_root)``; ``has_root`` is ``False`` if the branch ends do not
    bracket a sign change (no supporting tangent -> the secant is the envelope).
    The branch endpoint is nudged off ``c`` to dodge removable ``0/0`` gradient
    singularities of ``f'`` at the inflection (e.g. fractional-power ``|x|``).
    """
    eps = 1e-9 * (ub - lb) + 1e-12
    lo0 = jnp.where(positive_side, c + eps, lb)
    hi0 = jnp.where(positive_side, ub, c - eps)

    def g(t):
        return fp(t) * (t - e) - (f(t) - f(e))

    has_root = jnp.sign(g(lo0)) != jnp.sign(g(hi0))

    def body(_, carry):
        lo, hi, g_lo = carry
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        same = jnp.sign(g_mid) == jnp.sign(g_lo)
        lo = jnp.where(same, mid, lo)
        g_lo = jnp.where(same, g_mid, g_lo)
        hi = jnp.where(same, hi, mid)
        return (lo, hi, g_lo)

    lo, hi, _ = jax.lax.fori_loop(0, _BISECTION_ITERS, body, (lo0, hi0, g(lo0)))
    return 0.5 * (lo + hi), has_root


def _tangent_side(
    x,
    lb,
    ub,
    *,
    f,
    fp,
    c,
    from_lower: bool,
    f_on_left: bool,
    positive_side: bool,
    tangent_point: Callable | None,
):
    """One tangent-or-secant side of a single-inflection envelope.

    ``tangent_point`` is a closed-form ``endpoint -> t`` callable when available,
    else ``None`` (numeric bisection). Falls back to the secant when the tangent
    point exits its curvature branch.
    """
    endpoint = lb if from_lower else ub
    if tangent_point is not None:
        t = tangent_point(endpoint)
        valid = (t > c) & (t < ub) if positive_side else (t > lb) & (t < c)
    else:
        t, valid = _numeric_tangent_point(f, fp, c, endpoint, lb, ub, positive_side)
    line = f(t) + fp(t) * (x - t)
    if f_on_left:
        tangent_env = jnp.where(x <= t, f(x), line)
    else:
        tangent_env = jnp.where(x >= t, f(x), line)
    return jnp.where(valid, tangent_env, secant(f, x, lb, ub))


def single_inflection_envelope(
    x,
    lb,
    ub,
    *,
    f: Callable,
    fp: Callable,
    c: float,
    concavo_convex: bool,
    cv_tangent_point: Callable | None = None,
    cc_tangent_point: Callable | None = None,
):
    """Envelope of a single-inflection function over ``[lb, ub]``.

    Args:
        f, fp: ``f`` and ``f'`` as JAX callables.
        c: The inflection point.
        concavo_convex: ``True`` if ``f`` is concave for ``x < c`` and convex for
            ``x > c``; ``False`` for the convex-then-concave mirror image.
        cv_tangent_point, cc_tangent_point: Optional closed-form
            ``endpoint -> tangent_point`` callables; ``None`` selects the numeric
            bisection solver.

    Returns:
        ``(cv, cc)``.
    """
    sec = secant(f, x, lb, ub)
    straddle = (lb < c) & (ub > c)

    if concavo_convex:
        convex_box = lb >= c  # entirely convex -> cv = f, cc = secant
        concave_box = ub <= c  # entirely concave -> cv = secant, cc = f
        cv_straddle = _tangent_side(
            x,
            lb,
            ub,
            f=f,
            fp=fp,
            c=c,
            from_lower=True,
            f_on_left=False,
            positive_side=True,
            tangent_point=cv_tangent_point,
        )
        cc_straddle = _tangent_side(
            x,
            lb,
            ub,
            f=f,
            fp=fp,
            c=c,
            from_lower=False,
            f_on_left=True,
            positive_side=False,
            tangent_point=cc_tangent_point,
        )
    else:
        convex_box = ub <= c
        concave_box = lb >= c
        cv_straddle = _tangent_side(
            x,
            lb,
            ub,
            f=f,
            fp=fp,
            c=c,
            from_lower=False,
            f_on_left=True,
            positive_side=False,
            tangent_point=cv_tangent_point,
        )
        cc_straddle = _tangent_side(
            x,
            lb,
            ub,
            f=f,
            fp=fp,
            c=c,
            from_lower=True,
            f_on_left=False,
            positive_side=True,
            tangent_point=cc_tangent_point,
        )

    cv = jnp.where(convex_box, f(x), jnp.where(straddle, cv_straddle, sec))
    cc = jnp.where(concave_box, f(x), jnp.where(straddle, cc_straddle, sec))
    return cv, cc
