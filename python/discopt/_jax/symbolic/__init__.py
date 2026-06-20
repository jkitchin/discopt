"""SymPy-driven derivation of convex/concave envelopes and relaxations.

This subpackage is a **design-time** toolkit: it uses SymPy to *derive*,
*verify*, and *code-generate* convex/concave envelopes for nonlinear atoms that
the hand-written McCormick library (:mod:`discopt._jax.mccormick`,
:mod:`discopt._jax.envelopes`) does not yet cover tightly — with a focus on
terms arising in chemical-engineering, gas-network, and electrical-grid
optimization.

Design principles
-----------------
* **SymPy never runs on the solver hot path.** The symbolic engine produces
  closed-form envelope formulas and emits pure-JAX closures that match the
  existing relaxation contract. The generated closures are what the
  branch-and-bound solver executes; SymPy is only imported by developers /
  researchers deriving new atoms (it is an optional ``[sympy]`` extra).
* **Soundness is non-negotiable.** Every derived envelope must satisfy
  ``cv(x) <= f(x) <= cc(x)`` for all ``x`` in the box, with ``cv`` convex and
  ``cc`` concave. :mod:`discopt._jax.symbolic.verification` checks this before
  any atom is registered.

Univariate primitive contract
------------------------------
The code generator emits closures with the same signature as the univariate
McCormick primitives (e.g. :func:`discopt._jax.mccormick.relax_square`)::

    relax_fn(x, lb, ub) -> (cv, cc)

so that they are drop-in compatible with the relaxation compiler.

Public API
----------
``derive_envelope`` / ``EnvelopeResult``
    Symbolic derivation of the convex/concave envelope of a univariate
    SymPy expression over a box ``[a, b]``.
``lambdify_envelope``
    Convert an :class:`EnvelopeResult` into a JAX ``(x, lb, ub) -> (cv, cc)``
    closure.
``verify_envelope`` / ``VerificationReport``
    Numerically certify soundness (containment) and report tightness.
"""

from __future__ import annotations

from discopt._jax.symbolic.codegen import lambdify_envelope
from discopt._jax.symbolic.envelope_deriver import (
    Curvature,
    EnvelopeResult,
    derive_envelope,
)
from discopt._jax.symbolic.verification import (
    VerificationReport,
    verify_envelope,
)

__all__ = [
    "Curvature",
    "EnvelopeResult",
    "VerificationReport",
    "derive_envelope",
    "lambdify_envelope",
    "verify_envelope",
]
