"""Machine-checkable certificates of solution for discopt.

A *certificate* restates a solved model and its incumbent in a self-contained,
exact-rational form that an external checker -- the Lean development under
``lean/`` -- can consume to *prove* the claim discopt makes, rather than trust
the solver's ``gap_certified`` flag.

Tiers (see ``docs/dev/lean-certificate-plan.md``):

* **Tier 1 (feasibility)** -- proves the incumbent is genuinely feasible with the
  reported objective value (an honest bound). This module ships Tier 1.
* Tier 2 (convex/KKT) and Tier 3 (spatial branch-and-bound) are future phases.

The emitter reads only a :class:`~discopt.modeling.core.Model` and its
:class:`~discopt.modeling.core.SolveResult` -- no solver internals -- so it is
bound-neutral by construction.

Public API::

    from discopt.certificate import build_feasibility_certificate, write_certificate
    cert = build_feasibility_certificate(model, result)
    write_certificate(cert, "cert.json")

    from discopt.certificate import check_certificate      # reference checker
    ok, reason = check_certificate(cert)
"""

from __future__ import annotations

from .emit import (
    CertificateError,
    build_bnb_certificate,
    build_convex_certificate,
    build_feasibility_certificate,
    write_certificate,
)
from .refcheck import check_certificate

__all__ = [
    "CertificateError",
    "build_bnb_certificate",
    "build_convex_certificate",
    "build_feasibility_certificate",
    "write_certificate",
    "check_certificate",
]
