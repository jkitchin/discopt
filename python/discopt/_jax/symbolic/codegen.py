"""Code generation: :class:`EnvelopeResult` -> pure-JAX relaxation closure.

The emitted closure matches the univariate McCormick primitive contract used
throughout :mod:`discopt._jax.mccormick`::

    relax_fn(x, lb, ub) -> (cv, cc)

with ``cv <= f(x) <= cc`` for every ``x`` in ``[lb, ub]``. The closure is built
from SymPy expressions via :func:`sympy.lambdify` onto :mod:`jax.numpy`, so it is
``jax.jit`` / ``jax.grad`` / ``jax.vmap`` compatible and carries no SymPy
dependency at call time.

All box-dependent branching is expressed with :func:`jax.numpy.where` (never
Python ``if``) so a single compiled closure serves every branch-and-bound node,
exactly like the hand-written primitives.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import sympy as sp

from discopt._jax.symbolic.envelope_deriver import (
    Curvature,
    EnvelopeResult,
    Tangent,
)

# Below this box width we treat the interval as degenerate and return f(x).
_DEGENERATE = 1e-15

# Bisection iterations for the numeric tangent-point fallback. 60 halvings take
# a unit-width bracket below 1e-18, well past float64 resolution.
_BISECTION_ITERS = 60


def _to_jax(expr: sp.Expr, args: list[sp.Symbol]) -> Callable:
    """Lambdify ``expr`` over ``args`` onto jax.numpy.

    Uses SymPy's dedicated ``"jax"`` printer so non-smooth atoms (``sign``,
    ``Abs``, ``Heaviside``) and transcendentals render to ``jax.numpy`` calls
    rather than Python conditionals — the latter break under ``jax.vmap``.
    """
    return sp.lambdify(args, expr, modules="jax")


def _secant(f_fn: Callable, x, lb, ub):
    """Secant line of ``f`` between ``(lb, f(lb))`` and ``(ub, f(ub))`` at ``x``."""
    f_lb = f_fn(lb)
    f_ub = f_fn(ub)
    slope = (f_ub - f_lb) / (ub - lb)
    line = f_lb + slope * (x - lb)
    return jnp.where(jnp.abs(ub - lb) < _DEGENERATE, f_fn(x), line)


def lambdify_envelope(result: EnvelopeResult) -> Callable:
    """Build a JAX ``relax_fn(x, lb, ub) -> (cv, cc)`` closure for ``result``.

    Args:
        result: A derived :class:`EnvelopeResult`.

    Returns:
        A pure-JAX callable with the univariate primitive signature.
    """
    var, a_sym, b_sym = result.var, result.lower, result.upper
    f_fn = _to_jax(result.expr, [var])
    fp_fn = _to_jax(result.f_prime, [var])

    if result.curvature == Curvature.CONVEX:

        def relax_convex(x, lb, ub):
            return f_fn(x), _secant(f_fn, x, lb, ub)

        return relax_convex

    if result.curvature == Curvature.CONCAVE:

        def relax_concave(x, lb, ub):
            return _secant(f_fn, x, lb, ub), f_fn(x)

        return relax_concave

    # Single-inflection: build per-side tangent helpers.
    c = float(result.inflection)
    assert result.cv_tangent is not None and result.cc_tangent is not None
    cv_tan = result.cv_tangent
    cc_tan = result.cc_tangent

    def _closed_form_point_fn(tan: Tangent):
        sym = a_sym if tan.from_lower else b_sym
        return _to_jax(tan.point, [sym])

    cv_point_fn = _closed_form_point_fn(cv_tan) if cv_tan.point is not None else None
    cc_point_fn = _closed_form_point_fn(cc_tan) if cc_tan.point is not None else None

    def _numeric_tangent_point(tan: Tangent, lb, ub):
        """Bisect ``g(t) = f'(t)(t-e) - (f(t)-f(e)) = 0`` on the tangent branch.

        ``g`` has at most one sign change on the opposite-curvature branch. Returns
        ``(t, has_root)``: ``has_root`` is ``False`` when the branch ends do not
        bracket a sign change (no supporting tangent exists — the secant is then
        the envelope). Used when SymPy could not solve the tangent equation in
        closed form (e.g. transcendental exponents).
        """
        e = lb if tan.from_lower else ub
        # Offset the branch endpoint slightly off the inflection: f' can have a
        # removable 0/0 singularity exactly at c (e.g. fractional-power |x|), and
        # evaluating it there would feed NaN into the bracket.
        eps = 1e-9 * (ub - lb) + 1e-12
        lo0 = jnp.where(tan.tangent_positive_side, c + eps, lb)
        hi0 = jnp.where(tan.tangent_positive_side, ub, c - eps)

        def g(t):
            return fp_fn(t) * (t - e) - (f_fn(t) - f_fn(e))

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

    def _tangent_branch(tan: Tangent, point_fn, x, lb, ub):
        # The tangent point must lie strictly inside its curvature branch:
        # (c, ub) for the upper branch, (lb, c) for the lower one. When the box
        # is "mostly" on the far side, no supporting tangent exists and the
        # extended line would leave the function graph (unsound). There the
        # secant *is* the convex/concave envelope.
        if point_fn is not None:
            endpoint = lb if tan.from_lower else ub
            t = point_fn(endpoint)
            valid = (t > c) & (t < ub) if tan.tangent_positive_side else (t > lb) & (t < c)
        else:
            t, valid = _numeric_tangent_point(tan, lb, ub)
        line = f_fn(t) + fp_fn(t) * (x - t)
        if tan.f_on_left:
            tangent_env = jnp.where(x <= t, f_fn(x), line)
        else:
            tangent_env = jnp.where(x >= t, f_fn(x), line)
        return jnp.where(valid, tangent_env, _secant(f_fn, x, lb, ub))

    if result.curvature == Curvature.CONCAVO_CONVEX:

        def relax_cc_cv(x, lb, ub):
            straddle = (lb < c) & (ub > c)
            convex_box = lb >= c
            concave_box = ub <= c
            sec = _secant(f_fn, x, lb, ub)
            cv_straddle = _tangent_branch(cv_tan, cv_point_fn, x, lb, ub)
            cc_straddle = _tangent_branch(cc_tan, cc_point_fn, x, lb, ub)
            cv = jnp.where(convex_box, f_fn(x), jnp.where(straddle, cv_straddle, sec))
            cc = jnp.where(concave_box, f_fn(x), jnp.where(straddle, cc_straddle, sec))
            return cv, cc

        return relax_cc_cv

    # CONVEXO_CONCAVE
    def relax_cv_cc(x, lb, ub):
        straddle = (lb < c) & (ub > c)
        convex_box = ub <= c
        concave_box = lb >= c
        sec = _secant(f_fn, x, lb, ub)
        cv_straddle = _tangent_branch(cv_tan, cv_point_fn, x, lb, ub)
        cc_straddle = _tangent_branch(cc_tan, cc_point_fn, x, lb, ub)
        cv = jnp.where(convex_box, f_fn(x), jnp.where(straddle, cv_straddle, sec))
        cc = jnp.where(concave_box, f_fn(x), jnp.where(straddle, cc_straddle, sec))
        return cv, cc

    return relax_cv_cc
