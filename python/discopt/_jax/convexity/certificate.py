"""Sound box-local convexity certificate.

The certificate answers the question "is ``f`` convex on the given
box?" with a proof, leveraging:

1. :mod:`interval_ad` for a sound interval enclosure of the Hessian
   over the box.
2. :mod:`eigenvalue` for a sound lower bound on the minimum
   eigenvalue across every concrete Hessian in that enclosure.

If the lower eigenvalue bound is ≥ 0 on the box, ``f`` is convex
there (second-order sufficient condition, Boyd & Vandenberghe §3.1.4).
Symmetrically, an upper bound ≤ 0 proves concavity. Any other
outcome returns ``None`` — a conservative abstention, not a claim
of nonconvexity.

This routine never loosens a verdict from the syntactic walker
:mod:`rules`. Callers combine the two sources by preferring the
syntactic CONVEX/CONCAVE (cheaper) and only falling back to the
certificate when the syntactic walker says UNKNOWN.

References
----------
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.1.4.
Adjiman, Dallwig, Floudas, Neumaier (1998), "αBB — I. Theoretical
  advances," Comput. Chem. Eng. — the interval-Hessian foundation
  this certificate operationalises.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.modeling.core import Expression, Model

from .eigenvalue import gershgorin_lambda_max, gershgorin_lambda_min
from .interval_ad import interval_hessian
from .lattice import Curvature

# Tolerance for accepting "λ_min ≥ 0" despite floating-point slop. The
# interval Hessian already outward-rounds, so genuine zero eigenvalues
# may appear as small negatives; a very tight tolerance suffices.
_PSD_TOL = 1e-10


def certify_convex(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
) -> Optional[Curvature]:
    """Return a sound convex/concave verdict or ``None``.

    Args:
        expr: A scalar expression.
        model: The model defining the variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared
            bounds — used when the caller has a tighter box from
            FBBT or branching than the model's static declaration.

    Returns:
        * ``Curvature.CONVEX`` if the interval Hessian is provably
          PSD on the box.
        * ``Curvature.CONCAVE`` if the interval Hessian is provably
          NSD on the box.
        * ``None`` if neither test succeeds (indefinite, unsupported
          atoms, or a looseness failure in Gershgorin). Returning
          ``None`` is a deliberate abstention — the caller must treat
          the expression as non-convex.
    """
    try:
        ad = interval_hessian(expr, model, box=box)
    except ValueError:
        # Expressions referencing array variables directly are not
        # supported by v1; abstain rather than guess.
        return None

    hess = ad.hess
    if not (np.all(np.isfinite(hess.lo)) and np.all(np.isfinite(hess.hi))):
        return None

    lam_min = gershgorin_lambda_min(hess)
    if lam_min >= -_PSD_TOL:
        return Curvature.CONVEX

    lam_max = gershgorin_lambda_max(hess)
    if lam_max <= _PSD_TOL:
        return Curvature.CONCAVE

    return None


__all__ = ["certify_convex"]
