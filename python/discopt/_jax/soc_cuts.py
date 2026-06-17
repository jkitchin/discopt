"""Second-order-cone (SOC) outer-approximation cuts.

A second-order cone constraint ``||y||_2 <= t`` (``y in R^k``, ``t >= 0``) is convex
but not polyhedral, so an LP relaxation cannot represent it exactly. It can,
however, be *outer-approximated* to any accuracy by linear **gradient cuts**: at a
point ``(y*, t*)`` that violates the cone (``||y*|| > t*``), the unit vector
``u = y* / ||y*||`` yields

    u^T y <= t,

which is valid for the whole cone — for any feasible ``(y, t)``,
``u^T y <= ||u|| ||y|| = ||y|| <= t`` — yet violated at ``(y*, t*)`` because
``u^T y* = ||y*|| > t*``. Separating these on demand drives the LP relaxation
toward the true cone without an SOC/conic solver (Ben-Tal & Nemirovski 2001).

This complements the PSD (moment) cuts in :mod:`discopt._jax.psd_cuts`: SOC cuts
tighten the *convex-cone* substructure that conic-representable terms expose
(e.g. ``sqrt(x^T Q x)`` Euclidean norms detected in
``convexity/patterns.py``), where PSD cuts target the nonconvex moment matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from discopt._jax.cutting_planes import LinearCut

__all__ = ["soc_gradient_cut"]


def soc_gradient_cut(
    y_vals: np.ndarray,
    t_val: float,
    y_cols: Sequence[int],
    t_col: int,
    n_total: int,
    *,
    tol: float = 1e-7,
) -> Optional[LinearCut]:
    """Separate one SOC outer-approximation cut, or ``None`` if satisfied.

    Parameters
    ----------
    y_vals : (k,) array
        Values of the cone's vector part ``y`` at the current relaxation point.
    t_val : float
        Value of the cone's scalar bound ``t`` at the point.
    y_cols : (k,) sequence of int
        Relaxation column index of each ``y`` component.
    t_col : int
        Relaxation column index of ``t``.
    n_total : int
        Total number of relaxation columns (length of the cut vector).
    tol : float
        Only emit a cut when ``||y*|| > t* + tol`` (the point is outside the cone).

    Returns
    -------
    LinearCut or None
        The valid inequality ``u^T y - t <= 0`` (``u = y*/||y*||``), violated at
        the current point. ``None`` when the point already satisfies the cone or
        ``y* = 0`` (no separating direction).
    """
    y = np.asarray(y_vals, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(y))
    if norm <= float(t_val) + tol:
        return None  # inside the cone (to tolerance): nothing to separate
    if norm <= tol:
        return None  # y* = 0 has no separating direction
    u = y / norm
    coeffs = np.zeros(n_total, dtype=np.float64)
    for a, col in enumerate(y_cols):
        coeffs[int(col)] += u[a]
    coeffs[int(t_col)] += -1.0
    # u^T y - t <= 0
    return LinearCut(coeffs=coeffs, rhs=0.0, sense="<=")
