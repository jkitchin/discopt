"""SymPy-driven derivation of convex/concave envelopes and relaxations.

This subpackage is a **design-time** toolkit: it uses SymPy to *derive*, *verify*,
and *code-generate* convex/concave envelopes for nonlinear atoms. SymPy is an
optional ``[sympy]`` extra and **must not** be required to import the sympy-free
parts of this package (the pure-JAX :mod:`~discopt._jax.symbolic.runtime`
assembly, the hand-written :mod:`~discopt._jax.symbolic.domains` packs, the
numeric :mod:`~discopt._jax.symbolic.patterns`, etc.).

To honor that, the top-level names that depend on SymPy are exposed **lazily**
(PEP 562): importing this package — or any sympy-free submodule — does not import
SymPy; SymPy is imported only when a SymPy-backed name is first accessed.

Public API
----------
``derive_envelope`` / ``EnvelopeResult`` / ``Curvature``
    Symbolic derivation of the envelope of a univariate SymPy expression.
``lambdify_envelope``
    Convert an :class:`EnvelopeResult` into a JAX ``(x, lb, ub) -> (cv, cc)``
    closure.
``verify_envelope`` / ``VerificationReport``
    Numerically certify soundness (containment + curvature) and report tightness.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# name -> submodule that defines it. ``verification`` is JAX-only (no SymPy);
# ``envelope_deriver`` and ``codegen`` import SymPy, so they are loaded lazily.
_LAZY = {
    "Curvature": "envelope_deriver",
    "EnvelopeResult": "envelope_deriver",
    "derive_envelope": "envelope_deriver",
    "lambdify_envelope": "codegen",
    "VerificationReport": "verification",
    "verify_envelope": "verification",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


if TYPE_CHECKING:  # static type-checkers / IDEs see the real symbols
    from discopt._jax.symbolic.codegen import lambdify_envelope  # noqa: F401
    from discopt._jax.symbolic.envelope_deriver import (  # noqa: F401
        Curvature,
        EnvelopeResult,
        derive_envelope,
    )
    from discopt._jax.symbolic.verification import (  # noqa: F401
        VerificationReport,
        verify_envelope,
    )
