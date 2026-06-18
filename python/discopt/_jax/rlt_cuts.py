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

from typing import Callable, Optional

import numpy as np

from discopt._jax.cutting_planes import LinearCut

__all__ = ["rlt_constraint_bound_cut", "rlt_quadratic_bound_cut_row"]


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


# Phase 2 — nonlinear (quadratic) constraint-factor RLT.
#
# A *quadratic* constraint factor ``g(x) <= 0`` (or ``g(x) == 0``) with
#   g(x) = const + sum_i lin_i x_i + sum_{(k,l)} q_{kl} x_k x_l
# multiplied by a variable bound factor gives the valid product
#   (-g(x)) * (x_j - L) >= 0      [lower factor]
#   (-g(x)) * (U - x_j) >= 0      [upper factor]
# because each operand is non-negative on the feasible region. Expanding the
# product yields degree-3 monomials ``x_k x_l x_j`` (and the degree-2 cross terms
# ``x_i x_j`` and ``x_k x_l``); linearizing each over its lifted product column
# gives a *linear* valid inequality in the lifted space. At any true point
# ``(x, lifted=products)`` the row value equals the product above, so the row
# never removes a feasible point — it only tightens the relaxation. For an
# *equality* parent ``g == 0`` the product is identically zero, so the row is
# emitted as a two-sided equality (strictly tighter than the one-sided
# inequality, and it needs no opposite-sign factor companion).
#
# The cut math is single-sourced here and reused by the build-time level-1 RLT
# path (which lifts the required product columns on demand) so the deployed rows
# are exactly the rows the soundness audit exercises.


def rlt_quadratic_bound_cut_row(
    quad: dict[tuple[int, ...], float],
    lin: dict[int, float],
    const: float,
    j: int,
    bound: float,
    lower: bool,
    orig_col: Callable[[int], Optional[int]],
    prod_col: Callable[[tuple[int, ...]], Optional[int]],
    n_total: int,
) -> Optional[tuple[np.ndarray, float]]:
    """Assemble the RLT product row for a quadratic factor times a bound factor.

    Parameters
    ----------
    quad : dict[tuple[int, ...], float]
        Quadratic part of ``g``: each key is a *sorted* index pair ``(k, l)``
        (``k <= l``; ``(k, k)`` is the square ``x_k**2``) mapping to the monomial
        coefficient ``q_{kl}``.
    lin : dict[int, float]
        Linear part of ``g``: original variable index -> coefficient.
    const : float
        Constant term of ``g``.
    j : int
        Original variable index providing the bound factor.
    bound : float
        ``L`` if ``lower`` else ``U``.
    lower : bool
        ``True`` for factor ``x_j - L >= 0``; ``False`` for ``U - x_j >= 0``.
    orig_col : callable
        ``orig_col(i)`` -> column of original variable ``i`` (or ``None``).
    prod_col : callable
        ``prod_col(sorted_multiset)`` -> lifted column for that product (or
        ``None`` if it is not available). The multiset is a sorted tuple of
        original indices with multiplicity, e.g. ``(k, l)`` or ``(k, l, j)``.
    n_total : int
        Total column count of the lifted space.

    Returns
    -------
    (coeffs, rhs) for the valid inequality ``coeffs . z >= rhs`` (the product
    ``(-g) * factor >= 0``), or ``None`` if any required lifted column is
    missing (abstaining only enlarges the relaxation, so it stays sound).
    """
    cj = orig_col(j)
    if cj is None:
        return None
    coeffs = np.zeros(n_total, dtype=np.float64)

    def _mult(idxs: tuple[int, ...], value: float) -> bool:
        """Add ``value`` onto the column for product ``idxs`` (len 1, 2 or 3)."""
        key = tuple(sorted(idxs))
        if len(key) == 1:
            c = orig_col(key[0])
        else:
            c = prod_col(key)
        if c is None:
            return False
        coeffs[c] += value
        return True

    sign = 1.0 if lower else -1.0  # lower: (x_j - L); upper: (U - x_j) = -(x_j - U)
    # (-g)*(x_j - bound) for lower; (-g)*(bound - x_j) = -(-g)*(x_j - bound) for upper.
    #   term*x_j contributes with +sign; term*(-bound) contributes with -sign*bound.
    # Constant term of g contributes ``-const * x_j`` (its ``+const*bound`` part
    # folds into the rhs below).
    if not _mult((j,), -sign * const):
        return None
    # Linear part of g.
    for i, ai in lin.items():
        if ai == 0.0:
            continue
        if not _mult((i, j), -sign * ai):  # -lin_i * (x_i x_j)
            return None
        if not _mult((i,), sign * bound * ai):  # +bound*lin_i * x_i
            return None
    # Quadratic part of g.
    for key, q in quad.items():
        if q == 0.0:
            continue
        if not _mult(tuple(key) + (j,), -sign * q):  # -q * (x_k x_l x_j)
            return None
        if not _mult(tuple(key), sign * bound * q):  # +bound*q * (x_k x_l)
            return None
    rhs = -sign * bound * const
    return coeffs, float(rhs)
