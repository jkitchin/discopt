"""Registry of symbolic-derived relaxation atoms (Phase 2).

A small, explicit catalogue that maps atom names to their pure-JAX relaxation
closures ``(x, lb, ub) -> (cv, cc)`` and the true function plus a representative
domain for certification. It gives the domain packs a single discovery point and
a uniform way to *certify* every atom with the same theorem-style gate used by
the engine (:func:`discopt._jax.symbolic.verify_envelope`).

The registry is sympy-free: the closures are the committed hand-written JAX from
the domain packs. ``certify_all`` is the design-time check that every registered
atom is sound before it is relied upon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from discopt._jax.symbolic.domains import chemeng, gas, power
from discopt._jax.symbolic.verification import VerificationReport, verify_envelope


@dataclass(frozen=True)
class AtomSpec:
    """A registered relaxation atom and how to certify it.

    Attributes:
        name: Catalogue key.
        relax_fn: Closure ``(x, lb, ub) -> (cv, cc)``.
        true_fn: The exact function ``f(x)`` (JAX-callable).
        domain: Representative ``(lo, hi)`` box family for certification.
        description: Human-readable summary.
    """

    name: str
    relax_fn: Callable
    true_fn: Callable
    domain: tuple[float, float]
    description: str


# Default Arrhenius / saturation parameters for the catalogue entries.
_ARRHENIUS_C = 6000.0  # E/R in kelvin (e.g. E ~ 50 kJ/mol)
_SATURATION_K = 1.0

_REGISTRY: dict[str, AtomSpec] = {
    "weymouth": AtomSpec(
        "weymouth",
        gas.weymouth_relax,
        lambda x: x * jnp.abs(x),
        (-50.0, 50.0),
        "Gas-network Weymouth pressure-drop term f|f|.",
    ),
    "arrhenius": AtomSpec(
        "arrhenius",
        chemeng.arrhenius_relax(_ARRHENIUS_C),
        lambda t: jnp.exp(-_ARRHENIUS_C / t),
        (250.0, 900.0),
        "Arrhenius reaction rate exp(-E/(R T)).",
    ),
    "saturating": AtomSpec(
        "saturating",
        chemeng.saturating_relax(_SATURATION_K),
        lambda x: x / (_SATURATION_K + x),
        (0.0, 20.0),
        "Langmuir/Monod saturation x/(K+x).",
    ),
    "xlogx": AtomSpec(
        "xlogx",
        chemeng.xlogx_relax,
        lambda x: x * jnp.log(x),
        (1e-2, 10.0),
        "Entropy/mixing term x ln x.",
    ),
    "cos_angle": AtomSpec(
        "cos_angle",
        power.cos_angle_relax,
        jnp.cos,
        (-1.4, 1.4),
        "AC-OPF cosine of angle difference on (-pi/2, pi/2).",
    ),
    "sin_angle": AtomSpec(
        "sin_angle",
        power.sin_angle_relax,
        jnp.sin,
        (-3.0, 3.0),
        "AC-OPF sine of angle difference on (-pi, pi).",
    ),
}


def available() -> list[str]:
    """Return the sorted names of registered atoms."""
    return sorted(_REGISTRY)


def get(name: str) -> AtomSpec:
    """Return the :class:`AtomSpec` for ``name`` (raises ``KeyError`` if absent)."""
    return _REGISTRY[name]


def certify(name: str, *, n_boxes: int = 400, seed: int = 0) -> VerificationReport:
    """Certify a single registered atom's soundness over its domain."""
    spec = _REGISTRY[name]
    return verify_envelope(
        spec.relax_fn, spec.true_fn, domain=spec.domain, n_boxes=n_boxes, seed=seed
    )


def certify_all(*, n_boxes: int = 400, seed: int = 0) -> dict[str, VerificationReport]:
    """Certify every registered atom; returns a name -> report mapping."""
    return {name: certify(name, n_boxes=n_boxes, seed=seed) for name in _REGISTRY}
