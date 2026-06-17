"""Targeted Reformulation-Linearization (RLT) bound-factor cuts.

The textbook McCormick envelope of a product ``x_i x_j`` is the RLT product of the
two variable bound factors. This module adds the other half of RLT level-1: the
product of a **linear constraint factor** with a **variable bound factor**.

For a constraint ``a^T x <= b`` (so ``b - a^T x >= 0``) and a bound factor
``x_j - l >= 0`` (or ``u - x_j >= 0``), the product of two non-negative
quantities is non-negative:

    (b - a^T x)(x_j - l) >= 0.

Expanded it is quadratic; linearizing each ``x_i x_j`` with the relaxation's
lifted column ``X_ij`` gives a *linear* valid inequality in ``(x, X)``. It is
valid for the whole feasible region (where ``X = x x^T`` makes the linearization
exact and both factors are non-negative), so it only tightens the relaxation. We
separate only violated cuts (the "targeted" part), reusing the same lifted column
map (``info``) as the PSD cuts — not the full, combinatorial Sherali-Adams
hierarchy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt._jax.cutting_planes import LinearCut

__all__ = ["rlt_constraint_bound_cut"]


def _prod_col(info: dict, i: int, j: int) -> Optional[int]:
    """Relaxation column for the lifted product ``X_ij`` (square if ``i == j``)."""
    if i == j:
        mono = info.get("monomial", {})
        usq = info.get("univariate_square", {})
        if (i, 2) in mono:
            return int(mono[(i, 2)])
        if (i, 2) in usq:
            return int(usq[(i, 2)])
        return None
    bil = info.get("bilinear", {})
    key = (min(i, j), max(i, j))
    return int(bil[key]) if key in bil else None


def rlt_constraint_bound_cut(
    a: dict[int, float],
    b: float,
    j: int,
    bound: float,
    lower: bool,
    info: dict,
    x_full: np.ndarray,
    n_total: int,
    *,
    tol: float = 1e-7,
) -> Optional[LinearCut]:
    """Separate the RLT cut ``(b - a^T x)(bound-factor on x_j) >= 0``.

    Parameters
    ----------
    a : dict[int, float]
        Constraint coefficients over original variable columns, ``a^T x <= b``.
    b : float
        Constraint right-hand side.
    j : int
        Original variable index providing the bound factor.
    bound : float
        The bound value: ``l`` if ``lower`` else ``u``.
    lower : bool
        ``True`` for factor ``x_j - l >= 0``; ``False`` for ``u - x_j >= 0``.
    info, x_full, n_total :
        Relaxation column map, current LP point, and column count.

    Returns
    -------
    LinearCut or None
        Valid inequality ``coeffs . z >= rhs`` violated at the current point, or
        ``None`` if every required lifted column is missing or the cut is
        satisfied. The cut never removes a feasible point: at any true point
        (``X = x x^T``, ``a^T x <= b``, bound factor ``>= 0``) the product of two
        non-negatives is non-negative.
    """
    orig = info.get("original", {})
    if j not in orig:
        return None
    # Resolve the lifted column for every X_ij that appears.
    prod = {}
    for i in a:
        if i not in orig:
            return None
        c = _prod_col(info, i, j)
        if c is None:
            return None
        prod[i] = c

    coeffs = np.zeros(n_total, dtype=np.float64)
    cj = int(orig[j])
    if lower:
        # (b - sum a_i x_i)(x_j - L) = b x_j - b L - sum a_i X_ij + L sum a_i x_i
        coeffs[cj] += b
        for i, ai in a.items():
            coeffs[int(orig[i])] += bound * ai
            coeffs[prod[i]] += -ai
        rhs = b * bound
    else:
        # (b - sum a_i x_i)(U - x_j) = b U - b x_j - U sum a_i x_i + sum a_i X_ij
        coeffs[cj] += -b
        for i, ai in a.items():
            coeffs[int(orig[i])] += -bound * ai
            coeffs[prod[i]] += ai
        rhs = -b * bound

    # Separate only if violated at the current point: coeffs . z < rhs.
    if float(coeffs @ x_full) >= rhs - tol:
        return None
    return LinearCut(coeffs=coeffs, rhs=float(rhs), sense=">=")
