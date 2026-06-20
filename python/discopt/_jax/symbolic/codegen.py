"""Code generation: :class:`EnvelopeResult` -> pure-JAX relaxation closure.

The emitted closure matches the univariate McCormick primitive contract used
throughout :mod:`discopt._jax.mccormick`::

    relax_fn(x, lb, ub) -> (cv, cc)

with ``cv <= f(x) <= cc`` for every ``x`` in ``[lb, ub]``. The closure lambdifies
the SymPy ``f``/``f'`` onto :mod:`jax.numpy` (so it is ``jax.jit`` / ``jax.grad``
/ ``jax.vmap`` compatible and carries no SymPy dependency at call time) and
delegates the convex/concave assembly to :mod:`discopt._jax.symbolic.runtime`,
the same sympy-free construction the hand-written domain packs use.
"""

from __future__ import annotations

from typing import Callable

import sympy as sp

from discopt._jax.symbolic import runtime
from discopt._jax.symbolic.envelope_deriver import (
    Curvature,
    EnvelopeResult,
    Tangent,
)


def _to_jax(expr: sp.Expr, args: list[sp.Symbol]) -> Callable:
    """Lambdify ``expr`` over ``args`` onto jax.numpy.

    Uses SymPy's dedicated ``"jax"`` printer so non-smooth atoms (``sign``,
    ``Abs``, ``Heaviside``) and transcendentals render to ``jax.numpy`` calls
    rather than Python conditionals — the latter break under ``jax.vmap``.
    """
    fn: Callable = sp.lambdify(args, expr, modules="jax")
    return fn


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
            return runtime.convex_envelope(x, lb, ub, f=f_fn)

        return relax_convex

    if result.curvature == Curvature.CONCAVE:

        def relax_concave(x, lb, ub):
            return runtime.concave_envelope(x, lb, ub, f=f_fn)

        return relax_concave

    # Single-inflection: lambdify the closed-form tangent points when available,
    # else leave them None so the runtime solves them numerically per box.
    assert (
        result.inflection is not None
        and result.cv_tangent is not None
        and result.cc_tangent is not None
    )
    c = float(result.inflection)
    cv_tan: Tangent = result.cv_tangent
    cc_tan: Tangent = result.cc_tangent
    concavo_convex = result.curvature == Curvature.CONCAVO_CONVEX

    def _point_fn(tan: Tangent):
        if tan.point is None:
            return None
        sym = a_sym if tan.from_lower else b_sym
        return _to_jax(tan.point, [sym])

    cv_point_fn = _point_fn(cv_tan)
    cc_point_fn = _point_fn(cc_tan)

    def relax_single_inflection(x, lb, ub):
        return runtime.single_inflection_envelope(
            x,
            lb,
            ub,
            f=f_fn,
            fp=fp_fn,
            c=c,
            concavo_convex=concavo_convex,
            cv_tangent_point=cv_point_fn,
            cc_tangent_point=cc_point_fn,
        )

    return relax_single_inflection
