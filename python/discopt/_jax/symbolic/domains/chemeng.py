"""Chemical-engineering relaxations derived with the symbolic engine.

Hand-written JAX closures (no SymPy on the hot path) for nonlinear atoms common
in reactor and separation models, built on the shared
:mod:`discopt._jax.symbolic.runtime` envelope construction and certified against
the SymPy derivation in the test suite.

Atoms:
    * :func:`arrhenius_relax` — reaction rate ``exp(-E/(R T))`` (convex-then-
      concave in temperature, inflection at ``T = E/(2R)``).
    * :func:`saturating_relax` — Langmuir/Monod saturation ``x/(K + x)``
      (concave for ``x >= 0``).
    * :func:`xlogx_relax` — entropy/mixing term ``x ln x`` (convex for ``x > 0``).
"""

from __future__ import annotations

import jax.numpy as jnp

from discopt._jax.symbolic import runtime


def arrhenius_relax(activation_over_r: float):
    """Envelope factory for the Arrhenius rate term ``exp(-c / T)``, ``c = E/R``.

    As a function of temperature ``T > 0`` the rate is convex for ``T < c/2`` and
    concave for ``T > c/2`` (inflection at ``T = c/2``). For typical kinetics
    ``c = E/R`` is large, so normal temperature windows sit on the convex branch
    and the envelope reduces to ``(f, secant)``.

    Args:
        activation_over_r: ``E/R`` in kelvin (``E`` activation energy, ``R`` gas
            constant). Must be positive.

    Returns:
        A closure ``(T, lb, ub) -> (cv, cc)`` with ``T`` in kelvin.
    """
    c = float(activation_over_r)
    if c <= 0.0:
        raise ValueError("activation_over_r (E/R) must be positive")

    def f(t):
        return jnp.exp(-c / t)

    def fp(t):
        return jnp.exp(-c / t) * c / t**2

    def relax(t, lb, ub):
        return runtime.single_inflection_envelope(
            t, lb, ub, f=f, fp=fp, c=0.5 * c, concavo_convex=False
        )

    return relax


def saturating_relax(half_saturation: float = 1.0):
    """Envelope factory for the Langmuir/Monod saturation ``x / (K + x)``.

    Concave for ``x >= 0`` (``K > 0``): adsorbed fraction / specific growth rate.

    Args:
        half_saturation: ``K`` (the half-saturation constant). Must be positive.

    Returns:
        A closure ``(x, lb, ub) -> (cv, cc)`` valid for ``x >= 0``.
    """
    k = float(half_saturation)
    if k <= 0.0:
        raise ValueError("half_saturation (K) must be positive")

    def f(x):
        return x / (k + x)

    return lambda x, lb, ub: runtime.concave_envelope(x, lb, ub, f=f)


def xlogx_relax(x, lb, ub):
    """Envelope of the entropy/mixing term ``x ln x`` (convex for ``x > 0``)."""
    return runtime.convex_envelope(x, lb, ub, f=lambda t: t * jnp.log(t))
