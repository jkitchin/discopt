"""Electrical-grid (AC OPF) relaxations: trigonometric angle terms.

The AC power-flow equations couple bus voltages and angle differences through
``cos(θ_ij)`` and ``sin(θ_ij)``. Convex relaxations of OPF (e.g. the QC
relaxation of Coffrin et al.) relax these trigonometric terms over an angle-
difference box ``[θ_lb, θ_ub]`` and combine them with envelopes of the voltage
products. This module provides the univariate trigonometric building blocks as
hand-written JAX closures (no SymPy on the hot path), built on the shared
:mod:`discopt._jax.symbolic.runtime` construction.

The voltage-product terms ``V_i V_j cos(θ)`` are genuinely multivariate (a
product of an angle envelope with a voltage bilinear) and are handled by
combining these envelopes with the McCormick product rules; see the design plan.

Atoms:
    * :func:`cos_angle_relax` — ``cos(θ)`` on an angle box within
      ``(-π/2, π/2)``, where it is concave (the QC under/over-estimators).
    * :func:`sin_angle_relax` — ``sin(θ)`` on an angle box within ``(-π, π)``,
      convex for ``θ < 0`` and concave for ``θ > 0`` (inflection at 0).
"""

from __future__ import annotations

import jax.numpy as jnp

from discopt._jax.symbolic import runtime

#: Largest angle magnitude for which ``cos`` stays concave (its inflections).
HALF_PI = 1.5707963267948966


def cos_angle_relax(theta, lb, ub):
    """Concave-regime envelope of ``cos(θ)`` for an angle box in ``(-π/2, π/2)``.

    ``cos`` is concave on ``(-π/2, π/2)`` (``cos'' = -cos < 0``), so the concave
    overestimator is ``cos`` itself and the convex underestimator is the secant —
    exactly the QC-relaxation cosine envelope. Valid only when
    ``[lb, ub] ⊆ (-π/2, π/2)``; outside that range ``cos`` is no longer concave
    and a different (multi-inflection) construction is required.

    Returns:
        ``(cv, cc)`` with ``cv <= cos(θ) <= cc``.
    """
    return runtime.concave_envelope(theta, lb, ub, f=jnp.cos)


def sin_angle_relax(theta, lb, ub):
    """Single-inflection envelope of ``sin(θ)`` for an angle box in ``(-π, π)``.

    ``sin`` is convex for ``θ < 0`` and concave for ``θ > 0`` (inflection at 0),
    so over a straddling angle box the envelope is the tangent/secant
    construction. Valid when ``[lb, ub] ⊆ (-π, π)`` (one inflection in range).

    Returns:
        ``(cv, cc)`` with ``cv <= sin(θ) <= cc``.
    """
    return runtime.single_inflection_envelope(
        theta, lb, ub, f=jnp.sin, fp=jnp.cos, c=0.0, concavo_convex=False
    )
