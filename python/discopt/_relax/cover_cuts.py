"""Knapsack cover cuts for binary MILP relaxations (roadmap Phase 3).

For a binary-knapsack row ``sum_{j} a_j x_j <= b`` with ``a_j > 0`` and
``x_j in {0,1}``, a *cover* ``C`` is a set with ``sum_{j in C} a_j > b`` — its
items cannot all be chosen, giving the valid cut

    sum_{j in C} x_j <= |C| - 1.

This cut never excludes a feasible 0/1 point (picking every item in a cover
violates the knapsack row), so it is rigorously valid regardless of the LP
point used to find it. Separation looks for a cover that the current LP
solution ``x*`` *violates* (``sum_{j in C} x*_j > |C| - 1``).

Basis-free (no simplex tableau / crossover needed), so it composes with the
interior-point relaxations of the self-hosted B&B.
"""

from __future__ import annotations

import numpy as np


def has_binary_knapsack_rows(A_ub, b_ub, is_binary, tol: float = 1e-9) -> bool:
    """Cheap precheck: is any inequality row a pure positive binary knapsack?"""
    if A_ub is None or b_ub is None:
        return False
    A = np.asarray(A_ub, dtype=np.float64)
    b = np.asarray(b_ub, dtype=np.float64).ravel()
    is_binary = np.asarray(is_binary, dtype=bool)
    for i in range(A.shape[0]):
        if _knapsack_support(A[i], b[i], is_binary, tol) is not None:
            return True
    return False


def _knapsack_support(a: np.ndarray, b: float, is_binary: np.ndarray, tol: float):
    """Return the nonzero support if row ``a^T x <= b`` is a positive binary
    knapsack (all nonzeros on binary columns, positive, and ``b > 0``), else
    ``None``."""
    nz = np.where(np.abs(a) > tol)[0]
    if nz.size == 0 or b <= tol:
        return None
    for j in nz:
        if not is_binary[j] or a[j] <= tol:
            return None
    return nz


def separate_cover_cuts(
    A_ub,
    b_ub,
    x_star: np.ndarray,
    is_binary,
    tol: float = 1e-6,
    max_cuts: int = 64,
) -> list[tuple[frozenset[int], float]]:
    """Find violated minimal cover cuts for the binary-knapsack rows.

    Returns a list of ``(C, rhs)`` where the cut is
    ``sum_{j in C} x_j <= rhs`` with ``rhs = |C| - 1``. Every returned cut is
    valid; each is also violated by ``x_star`` (``sum_{j in C} x*_j > rhs``).
    """
    if A_ub is None or b_ub is None:
        return []
    A = np.asarray(A_ub, dtype=np.float64)
    b = np.asarray(b_ub, dtype=np.float64).ravel()
    x = np.asarray(x_star, dtype=np.float64)
    is_binary = np.asarray(is_binary, dtype=bool)

    cuts: list[tuple[frozenset[int], float]] = []
    seen: set[frozenset[int]] = set()
    for i in range(A.shape[0]):
        nz = _knapsack_support(A[i], b[i], is_binary, tol)
        if nz is None:
            continue
        cut = _separate_row(A[i], float(b[i]), x, nz, tol)
        if cut is not None and cut[0] not in seen:
            seen.add(cut[0])
            cuts.append(cut)
            if len(cuts) >= max_cuts:
                break
    return cuts


def _separate_row(a, b, x, nz, tol):
    """Greedy violated-cover separation for one knapsack row.

    Build a cover by adding items in decreasing ``x*`` order (the items the LP
    most wants to select), then minimalize by dropping the items with smallest
    ``x*`` while the set stays a cover. Emit the cut only if violated.
    """
    order = sorted(nz, key=lambda j: -x[j])
    cover: list[int] = []
    weight = 0.0
    for j in order:
        cover.append(j)
        weight += a[j]
        if weight > b + tol:
            break
    if weight <= b + tol:
        return None  # all items fit: no cover in this row

    # Minimalize: drop least-wanted items while sum of weights stays > b.
    for j in sorted(cover, key=lambda j: x[j]):
        if len(cover) <= 2:
            break
        if weight - a[j] > b + tol:
            cover.remove(j)
            weight -= a[j]

    rhs = float(len(cover) - 1)
    if sum(x[j] for j in cover) > rhs + tol:
        return frozenset(int(j) for j in cover), rhs
    return None


def separate_clique_cuts(
    edges, x_star: np.ndarray, tol: float = 1e-6
) -> list[tuple[frozenset[int], float]]:
    """Violated clique cuts ``sum_{j in C} x_j <= 1`` from conflict-graph edges.

    Each edge ``(i, j)`` asserts binaries ``i`` and ``j`` cannot both be 1. The
    edges are greedily *merged* into larger cliques (every pair in ``C``
    conflicts), giving the valid cut ``sum_{j in C} x_j <= 1`` — much stronger
    than the pairwise edges (a triangle yields ``x_i+x_j+x_k <= 1``) and, for
    ``|C| >= 3``, separating even a symmetric interior point (``0.5*3 > 1``)
    that the pairwise/cover cuts miss. Pairwise edges are usually redundant
    with the constraints they came from; the merged cliques are where the
    value is. Returns violated cuts in the shared ``(C, rhs)`` format.
    """
    x = np.asarray(x_star, dtype=np.float64)
    adj: dict[int, set[int]] = {}
    for i, j in edges:
        i, j = int(i), int(j)
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    cuts: list[tuple[frozenset[int], float]] = []
    seen: set[frozenset[int]] = set()
    for i, j in edges:
        i, j = int(i), int(j)
        clique = {i, j}
        # Grow into a (greedy maximal) clique: add common neighbours adjacent
        # to every current member, preferring high-``x*`` items to maximize
        # the chance of a violated cut.
        candidates = sorted(adj[i] & adj[j], key=lambda k: -x[k])
        for k in candidates:
            if all(k in adj[c] for c in clique):
                clique.add(k)
        cf = frozenset(clique)
        if cf in seen:
            continue
        seen.add(cf)
        if sum(x[t] for t in clique) > 1.0 + tol:
            cuts.append((cf, 1.0))
    return cuts
