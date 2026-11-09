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

#: The unbounded sentinel of the LP layer (CLAUDE.md, INF note). A bound at or
#: beyond it is "no bound"; the test is on the BOUND, never on a product.
_COVER_INF = 1e20


def _column_box(j: int, is_binary: np.ndarray, lb, ub) -> tuple[float, float]:
    """The box of column ``j``: the caller's ``lb``/``ub`` when given, else
    ``[0, 1]`` for a binary and unbounded for anything else."""
    lo = float(lb[j]) if lb is not None else (0.0 if is_binary[j] else -np.inf)
    hi = float(ub[j]) if ub is not None else (1.0 if is_binary[j] else np.inf)
    return lo, hi


def _knapsack_support(a: np.ndarray, b: float, is_binary: np.ndarray, tol: float, lb=None, ub=None):
    """Return ``(items, cap)`` if row ``a^T x <= b`` yields a binary knapsack, else
    ``None``.

    ``items`` are the binary columns with a coefficient above ``tol`` (the cover
    candidates). Every other nonzero must be accounted for, not ignored: the cover
    argument is "every item at 1 pushes the row over ``cap``", which holds only
    against ``cap = b - min(contribution of the rest)``. So

    * a non-binary column, or a negative binary coefficient, above ``tol`` refuses
      the row (the row is not a knapsack -- unchanged behaviour);
    * a nonzero at or below ``tol`` (previously dropped silently) has its MINIMUM
      contribution over its box charged to the capacity, and refuses the row if
      that minimum is unbounded.

    #1603 (the Python twin of #1236 in ``crates/discopt-core/src/lp/cover.rs``):
    ``x0 + x1 - 1e-7 y <= 1.5`` with ``y in [0, 1e9]`` admits ``(1, 1, 5e6)``, but
    the tiny ``y`` term fell under ``tol`` and was dropped, so the separator
    emitted ``x0 + x1 <= 1`` -- an invalid cut, measured to return a wrong
    ``optimal`` (2.0 vs 3.0) through ``milp_backend="native"`` and an unsound root
    LP bound from the NLP-BB root-cut stage.
    """
    nz_all = np.flatnonzero(a)
    if nz_all.size == 0:
        return None
    items = []
    cap = float(b)
    for j in nz_all:
        aj = float(a[j])
        if abs(aj) > tol:
            if not is_binary[j] or aj <= tol:
                return None
            items.append(int(j))
            continue
        # A tiny nonzero: charge its minimum over the box to the capacity.
        lo, hi = _column_box(int(j), is_binary, lb, ub)
        pin = lo if aj > 0.0 else hi
        if not np.isfinite(pin) or abs(pin) >= _COVER_INF:
            return None
        cap -= aj * pin
    if not items or not np.isfinite(cap) or cap <= tol:
        return None
    return np.asarray(items, dtype=np.int64), cap


def separate_cover_cuts(
    A_ub,
    b_ub,
    x_star: np.ndarray,
    is_binary,
    tol: float = 1e-6,
    max_cuts: int = 64,
    lb=None,
    ub=None,
) -> list[tuple[frozenset[int], float]]:
    """Find violated minimal cover cuts for the binary-knapsack rows.

    Returns a list of ``(C, rhs)`` where the cut is
    ``sum_{j in C} x_j <= rhs`` with ``rhs = |C| - 1``. Every returned cut is
    valid; each is also violated by ``x_star`` (``sum_{j in C} x*_j > rhs``).

    ``lb``/``ub`` are the column bounds. A row's nonzeros at or below ``tol`` are
    charged to its capacity at their worst case over this box (see
    :func:`_knapsack_support`); without bounds a non-binary column is unbounded
    and such a row is refused.
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
        support = _knapsack_support(A[i], b[i], is_binary, tol, lb, ub)
        if support is None:
            continue
        nz, cap = support
        cut = _separate_row(A[i], cap, x, nz, tol)
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
