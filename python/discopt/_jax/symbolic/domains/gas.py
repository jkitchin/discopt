"""Gas-network relaxations: the Weymouth ``f|f|`` pressure-drop term.

Steady-state gas pipe flow couples squared nodal pressures to flow through the
Weymouth equation,

.. math::  p_i^2 - p_j^2 = c \\cdot f \\, |f| ,

where ``f`` is the (signed) mass flow and ``c`` a pipe constant. The single
non-convexity is the odd term ``f|f| = sign(f) f^2`` — concave for ``f < 0``,
convex for ``f > 0``, with a kink at ``0``. Relaxing it as a generic product
``f * |f|`` (bilinear of ``f`` and ``|f|``) is valid but loose; the dedicated
single-inflection envelope below is the tightest convex/concave pair.

The closure :func:`weymouth_relax` is **hand-written JAX** — the solver hot path
never imports SymPy. Its formulas are exactly what
:func:`discopt._jax.symbolic.derive_envelope` produces for ``x*Abs(x)`` (convex
underestimator tangent from the lower bound at ``a(1-√2)``; concave overestimator
tangent from the upper bound at ``b(1-√2)``), and the test suite certifies the
two agree. :func:`derive_weymouth_symbolic` exposes the design-time derivation
used for that certification.

Both ``cv`` and ``cc`` are pure JAX, jit/grad/vmap compatible, and branch on the
box only through :func:`jax.numpy.where`, so a single compiled closure serves
every branch-and-bound node.
"""

from __future__ import annotations

import jax.numpy as jnp

# 1 - sqrt(2): the tangent-point coefficient for the f|f| envelope, derived
# symbolically (see derive_weymouth_symbolic). Tangent from endpoint e touches
# the opposite-curvature branch at t = e * (1 - sqrt(2)).
_ONE_MINUS_SQRT2 = 1.0 - 1.4142135623730951
_DEGENERATE = 1e-15


def _f(x):
    return x * jnp.abs(x)


def _fp(x):
    # d/dx (x|x|) = 2|x|
    return 2.0 * jnp.abs(x)


def _secant(x, lb, ub):
    f_lb, f_ub = _f(lb), _f(ub)
    denom = jnp.where(jnp.abs(ub - lb) < _DEGENERATE, 1.0, ub - lb)
    line = f_lb + (f_ub - f_lb) / denom * (x - lb)
    return jnp.where(jnp.abs(ub - lb) < _DEGENERATE, _f(x), line)


def weymouth_relax(x, lb, ub):
    """Tight convex/concave envelope of the Weymouth term ``f|f|`` on ``[lb, ub]``.

    Args:
        x: Flow value(s) ``f``.
        lb, ub: Box bounds on ``f``.

    Returns:
        ``(cv, cc)`` with ``cv <= f|f| <= cc`` over ``[lb, ub]``; ``cv`` convex
        and ``cc`` concave.
    """
    sec = _secant(x, lb, ub)
    straddle = (lb < 0.0) & (ub > 0.0)
    convex_box = lb >= 0.0  # f|f| = f^2, convex -> cv = f, cc = secant
    concave_box = ub <= 0.0  # f|f| = -f^2, concave -> cv = secant, cc = f

    # Convex underestimator: tangent from the lower bound onto the convex branch.
    t_cv = lb * _ONE_MINUS_SQRT2
    line_cv = _f(t_cv) + _fp(t_cv) * (x - t_cv)
    cv_straddle = jnp.where(x >= t_cv, _f(x), line_cv)

    # Concave overestimator: tangent from the upper bound onto the concave branch.
    t_cc = ub * _ONE_MINUS_SQRT2
    line_cc = _f(t_cc) + _fp(t_cc) * (x - t_cc)
    cc_straddle = jnp.where(x <= t_cc, _f(x), line_cc)

    cv = jnp.where(convex_box, _f(x), jnp.where(straddle, cv_straddle, sec))
    cc = jnp.where(concave_box, _f(x), jnp.where(straddle, cc_straddle, sec))
    return cv, cc


def derive_weymouth_symbolic():
    """Return the SymPy-derived envelope closure for ``f|f|`` (design-time only).

    Imports SymPy; used by the test suite to certify that the hand-written
    :func:`weymouth_relax` matches the symbolic derivation. Not for the hot path.

    Returns:
        A JAX closure ``(x, lb, ub) -> (cv, cc)``.
    """
    import sympy as sp

    from discopt._jax.symbolic import derive_envelope, lambdify_envelope

    x = sp.Symbol("x", real=True)
    return lambdify_envelope(derive_envelope(x * sp.Abs(x), x, name="weymouth"))


def derive_signed_power_symbolic(beta: float):
    """SymPy-derived envelope for the signed-power flow term ``f|f|**(beta-1)``.

    Generalizes Weymouth (``beta = 2``) to Panhandle/Weymouth friction exponents
    (``beta`` in ``(1, 2]``). Imports SymPy; intended for design-time derivation
    and analysis. Raises ``EnvelopeDerivationError`` if SymPy cannot solve the
    tangent equation in closed form for the given exponent.

    Args:
        beta: Friction exponent (e.g. 1.85 for Panhandle-A, 2.0 for Weymouth).

    Returns:
        A JAX closure ``(x, lb, ub) -> (cv, cc)``.
    """
    import sympy as sp

    from discopt._jax.symbolic import derive_envelope, lambdify_envelope

    x = sp.Symbol("x", real=True)
    expr = x * sp.Abs(x) ** sp.nsimplify(beta - 1.0)
    return lambdify_envelope(derive_envelope(expr, x, name=f"signed_power_{beta}"))
