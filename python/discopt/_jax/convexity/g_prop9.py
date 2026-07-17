"""Proposition-9 gating for transformation relaxations (issue #181, item 7).

The transformation relaxation (items 2/3) is not always worth applying. KMS
2012 Proposition 9 characterizes exactly when the *recursive factorable*
outer-approximation already captures the G-convexity of a composite
``h = f(φ)``, so the transformation buys nothing:

    the transformation is REDUNDANT when the outer function ``f`` is convex
    (or ``f`` concave **and** the transform ``G`` convex);

    it adds strength only where the recursive decomposition leaves a gap —
    a nonconvex (curvature-unknown/indefinite) outer composition.

Intuition: if ``f`` is convex and ``φ`` already has a convex/concave envelope
from the recursive scheme, the standard factorable relaxation of ``f(φ)`` is
already as tight as the transformation would make it; running the transform
just duplicates work and adds redundant cuts. The gate lets a caller skip the
transformation exactly there.

This module is a pure predicate over curvature tags (from the existing
curvature walker, :mod:`rules`) — no bounds, no DAG mutation. It is consumed
as an optional gate by the cut generators in :mod:`g_convex_cut`.
"""

from __future__ import annotations

from typing import Optional

from .lattice import Curvature


def transformation_adds_value(
    outer_curvature: Optional[Curvature],
    *,
    transform_is_convex: bool = True,
) -> bool:
    """Whether applying the G-convexity transformation is non-redundant.

    Args:
        outer_curvature: Curvature of the outer function ``f`` in the
            composite ``f(φ)`` where ``φ`` is the G-convex intermediate — as
            classified by the recursive walker. ``None`` means "no known
            outer context" (e.g. ``φ`` is related to an auxiliary variable
            directly, ``f = identity``): treated as *unknown*, so the
            transformation is allowed (the recursive scheme has nothing extra
            to capture on a bare intermediate).
        transform_is_convex: Whether the convexifying transform ``G`` is
            convex. The constant-``ρ`` ``exp(ρ·)`` transform (item 2) is
            always convex, so this defaults to ``True``.

    Returns:
        ``True`` when the transformation may add strength (outer curvature
        unknown/indefinite, or concave with a non-convex ``G``); ``False``
        when Proposition 9 says the recursive factorable relaxation already
        captures it — the caller should then skip the transformation.
    """
    if outer_curvature in (Curvature.CONVEX, Curvature.AFFINE):
        # Recursive scheme already captures a convex/affine outer composition.
        return False
    if outer_curvature == Curvature.CONCAVE and transform_is_convex:
        # f concave AND G convex ⇒ recursive scheme captures it (Prop. 9).
        return False
    # Outer curvature UNKNOWN/indefinite (or None), or the atypical
    # concave-f with non-convex-G case — the transformation can help.
    return True


__all__ = ["transformation_adds_value"]
