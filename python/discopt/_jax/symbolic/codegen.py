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

import jax.numpy as jnp
import sympy as sp

from discopt._jax.symbolic.envelope_deriver import (
    Curvature,
    EnvelopeResult,
    Tangent,
)

# Below this box width we treat the interval as degenerate and return f(x).
_DEGENERATE = 1e-15


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

    def _point_fn(tan: Tangent) -> Callable:
        sym = a_sym if tan.from_lower else b_sym
        return _to_jax(tan.point, [sym])

    cv_point_fn = _point_fn(result.cv_tangent)
    cc_point_fn = _point_fn(result.cc_tangent)
    cv_tan = result.cv_tangent
    cc_tan = result.cc_tangent

    def _tangent_branch(tan: Tangent, point_fn: Callable, x, lb, ub):
        endpoint = lb if tan.from_lower else ub
        t = point_fn(endpoint)
        line = f_fn(t) + fp_fn(t) * (x - t)
        if tan.f_on_left:
            return jnp.where(x <= t, f_fn(x), line)
        return jnp.where(x >= t, f_fn(x), line)

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
