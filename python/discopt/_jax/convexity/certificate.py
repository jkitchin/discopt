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

from discopt.modeling.core import Constraint, Expression, Model

from .eigenvalue import gershgorin_lambda_max, gershgorin_lambda_min, psd_2x2_sufficient
from .interval import Interval
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

    # Structural rank-1 PSD fast path. When the AD walker has attached
    # a ``Rank1Factor`` with nonneg coefficient, the Hessian equals
    # ``c · v vᵀ`` pointwise (sound by construction) and is therefore
    # PSD on the entire box even when the entry-wise interval matrix
    # is too loose for Gershgorin to certify.
    rank1 = ad.rank1_factor
    if rank1 is not None and np.all(np.isfinite(rank1.c.lo)) and np.all(rank1.c.lo >= -_PSD_TOL):
        return Curvature.CONVEX

    # 2×2 sufficient PSD test (Sylvester) — useful when the interval
    # Hessian is tight enough that Gershgorin's row-sum loosening
    # would cross zero but the determinant proof still holds.
    if hess.lo.shape == (2, 2) and psd_2x2_sufficient(hess):
        return Curvature.CONVEX

    lam_min = gershgorin_lambda_min(hess)
    if lam_min >= -_PSD_TOL:
        return Curvature.CONVEX

    lam_max = gershgorin_lambda_max(hess)
    if lam_max <= _PSD_TOL:
        return Curvature.CONCAVE

    return None


def refresh_convex_mask(
    model: Model,
    root_mask: list[bool],
    node_lb: np.ndarray,
    node_ub: np.ndarray,
) -> list[bool]:
    """Re-run the certificate against a B&B node's tightened bounds.

    For every constraint already proven convex at the root, the entry
    stays ``True`` (the node box is a subset of the root box and
    soundness propagates). For every constraint still ``False``, the
    certificate is consulted on the node box; when it proves the body
    convex in the sense implied by the constraint direction, the entry
    flips to ``True``.

    Returns a new list without mutating ``root_mask``. Falls back to
    returning the original mask unchanged if ``model`` or the bounds
    are shape-incompatible — the caller must remain functional even
    when the refresh cannot run.

    This function only ever tightens the mask. It never flips a
    ``True`` entry to ``False``, preserving the soundness invariant
    required by the solver's OA-cut and αBB-skip gates.
    """
    n_vars = sum(v.size for v in model._variables)
    if len(node_lb) != n_vars or len(node_ub) != n_vars:
        return list(root_mask)

    # Skip work when nothing can change — every slot is already True,
    # or there are no constraints at all.
    if not root_mask or all(root_mask):
        return list(root_mask)

    # Build the per-variable box from the node's flat bounds.
    box: dict = {}
    offset = 0
    for v in model._variables:
        size = v.size
        shape = v.shape if v.shape else (1,)
        lb_slice = np.asarray(node_lb[offset : offset + size], dtype=np.float64)
        ub_slice = np.asarray(node_ub[offset : offset + size], dtype=np.float64)
        try:
            box[v] = Interval(lb_slice.reshape(shape), ub_slice.reshape(shape))
        except ValueError:
            # lb > ub somewhere — the node is infeasible; return the
            # root mask unchanged. The caller will discover the
            # infeasibility via its own channels.
            return list(root_mask)
        offset += size

    refreshed = list(root_mask)
    constraint_index = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            constraint_index += 1
            continue
        if refreshed[constraint_index]:
            constraint_index += 1
            continue
        try:
            cert = certify_convex(c.body, model, box=box)
        except Exception:
            cert = None
        if cert is None:
            constraint_index += 1
            continue
        if c.sense == "<=" and cert == Curvature.CONVEX:
            refreshed[constraint_index] = True
        elif c.sense == ">=" and cert == Curvature.CONCAVE:
            refreshed[constraint_index] = True
        elif c.sense == "==" and cert == Curvature.CONVEX and cert == Curvature.CONCAVE:
            # Equality requires affine; the certificate doesn't return
            # AFFINE, so no tightening is possible here.
            pass
        constraint_index += 1
    return refreshed


__all__ = ["certify_convex", "refresh_convex_mask"]
