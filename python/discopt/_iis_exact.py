"""Exact (rational) feasibility decision for a small MILP subsystem.

The MILP IIS engine (:mod:`discopt._iis_milp`) confirms a finished subsystem's
infeasibility by two routes: HiGHS with presolve on, and with presolve off
(#1634). On big-M models the presolve-off route cannot answer. On MIPcc27's
``neos-5115982-kedron`` the 12-element reference IIS is two binaries forced to 0
by rows ``x - 8e6*z >= -7999999.656`` with ``ub(x) = 0.34375`` (2.5e-4 of slack,
3e-11 relative), then a 4-row LP contradiction. Presolve proves it in 4 ms;
presolve-off branch-and-bound returns ``Optimal`` with the binaries at
``1 - 3e-11`` at every ``mip_feasibility_tolerance`` down to 1e-10. Rounding them
breaks the rows, so the point is an artifact and the verdict stays undecided.
Measured on the smallset, kedron, osun and otra are all of this kind.

This module is the replacement second route for such a set, and it is free of
tolerances. It enumerates the held integer columns that occur in active rows by
depth-first search with exact fixing. A node is pruned when some row's
activity range, computed exactly from the fixed values and the remaining
columns' boxes, misses the row's bounds. At each leaf the continuous part is
decided by Fourier--Motzkin elimination over :class:`fractions.Fraction`.

The data are the engine's float64 arrays, converted exactly (``Fraction(float)``
is the double's exact value). This decides the model the engine was given,
with no rounding anywhere. A float LP is never used: a float-pruned DFS gave a
false "feasible" on otra (a row violated by 2.9e-11), and a float
"infeasible" prune would make the second route depend on the same kind of
claim it checks.

The result is ``True`` (infeasible), ``False`` (feasible: an integer assignment
and a rational continuous point exist) or ``None`` (a size or time cap was
hit). It is ``None``, never a guess, whenever the set is too large to decide
exactly.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Optional

import numpy as np

#: Largest number of integer assignments the DFS may visit (leaves + internal).
EXACT_MAX_NODES = 1 << 18
#: Fourier--Motzkin is doubly exponential in the eliminated columns; past these
#: caps the decision is left undecided.
EXACT_MAX_CONT = 12
EXACT_MAX_INEQS = 20_000


class _Undecided(Exception):
    pass


def _fm_feasible(ineqs: list[tuple[dict[int, Fraction], Fraction]]) -> bool:
    """``sum a_j x_j <= b`` for every ``(a, b)``: is there a real ``x``? Exact."""
    Ineq = tuple[dict[int, Fraction], Fraction]
    cols = sorted({j for a, _ in ineqs for j in a})
    for j in cols:
        pos: list[Ineq] = []
        neg: list[Ineq] = []
        rest: list[Ineq] = []
        for a0, b0 in ineqs:
            c = a0.get(j, 0)
            (pos if c > 0 else neg if c < 0 else rest).append((a0, b0))
        new = rest
        for ap, bp in pos:
            cp = ap[j]
            for an, bn in neg:
                cn = -an[j]
                a: dict[int, Fraction] = {}
                for k, v in ap.items():
                    if k != j:
                        a[k] = a.get(k, 0) + v * cn
                for k, v in an.items():
                    if k != j:
                        a[k] = a.get(k, 0) + v * cp
                a = {k: v for k, v in a.items() if v != 0}
                if not a:
                    if bp * cn + bn * cp < 0:
                        return False
                    continue
                new.append((a, bp * cn + bn * cp))
        if len(new) > EXACT_MAX_INEQS:
            raise _Undecided
        ineqs = new
    return all(b >= 0 for a, b in ineqs if not a)


def exact_feasible(
    A,
    rlo: np.ndarray,
    rhi: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    held: np.ndarray,
    stop_at: Optional[float] = None,
) -> Optional[bool]:
    """Exact decision of ``rlo <= A x <= rhi``, ``lb <= x <= ub``, ``x_j`` integer
    for ``j`` in ``held``. Non-finite bounds are absent. Returns True if the set
    is infeasible, False if it is feasible, None if undecided (caps or ``stop_at``).
    """
    A = A.tocsr()
    rows = [r for r in range(A.shape[0]) if np.isfinite(rlo[r]) or np.isfinite(rhi[r])]
    is_held = np.zeros(lb.size, dtype=bool)
    is_held[held] = True
    # An empty box anywhere, or a held column with no integer in its box, is
    # infeasible on its own.
    for j in range(lb.size):
        lo = lb[j] if np.isfinite(lb[j]) else None
        hi = ub[j] if np.isfinite(ub[j]) else None
        if lo is not None and hi is not None and Fraction(lo) > Fraction(hi):
            return True
        if is_held[j] and lo is not None and hi is not None:
            if np.ceil(lo) > np.floor(hi):
                return True
    F = Fraction
    row_terms: list[list[tuple[int, Fraction]]] = []
    row_lo: list[Optional[Fraction]] = []
    row_hi: list[Optional[Fraction]] = []
    used: set[int] = set()
    for r in rows:
        s, e = A.indptr[r], A.indptr[r + 1]
        terms = [(int(j), F(float(v))) for j, v in zip(A.indices[s:e], A.data[s:e]) if v != 0]
        row_terms.append(terms)
        row_lo.append(F(float(rlo[r])) if np.isfinite(rlo[r]) else None)
        row_hi.append(F(float(rhi[r])) if np.isfinite(rhi[r]) else None)
        used.update(j for j, _ in terms)
    ints = sorted(j for j in used if is_held[j])
    cont = sorted(j for j in used if not is_held[j])
    if len(cont) > EXACT_MAX_CONT:
        return None
    for j in ints:
        if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            return None  # unbounded integer: no finite enumeration
    blo = {j: (F(float(lb[j])) if np.isfinite(lb[j]) else None) for j in used}
    bhi = {j: (F(float(ub[j])) if np.isfinite(ub[j]) else None) for j in used}
    dom = {j: range(int(np.ceil(lb[j])), int(np.floor(ub[j])) + 1) for j in ints}
    # Branch first on the columns in the most rows (they prune soonest).
    deg = {j: 0 for j in ints}
    for terms in row_terms:
        for j, _ in terms:
            if j in deg:
                deg[j] += 1
    order = sorted(ints, key=lambda j: (-deg[j], j))
    fixed: dict[int, Fraction] = {}
    nodes = [0]

    def pruned() -> bool:
        """Some row's exact activity range misses its bounds."""
        for terms, lo, hi in zip(row_terms, row_lo, row_hi):
            mn: Optional[Fraction] = F(0)
            mx: Optional[Fraction] = F(0)
            for j, a in terms:
                if j in fixed:
                    v = a * fixed[j]
                    mn = None if mn is None else mn + v
                    mx = None if mx is None else mx + v
                    continue
                lo_j, hi_j = blo[j], bhi[j]
                # a * x over [lo_j, hi_j]; None stands for -inf (mn) / +inf (mx)
                small, big = (lo_j, hi_j) if a > 0 else (hi_j, lo_j)
                mn = None if (mn is None or small is None) else mn + a * small
                mx = None if (mx is None or big is None) else mx + a * big
            if hi is not None and mn is not None and mn > hi:
                return True
            if lo is not None and mx is not None and mx < lo:
                return True
        return False

    def leaf_feasible() -> bool:
        ineqs: list[tuple[dict[int, Fraction], Fraction]] = []
        for terms, lo, hi in zip(row_terms, row_lo, row_hi):
            a: dict[int, Fraction] = {}
            const = F(0)
            for j, c in terms:
                if j in fixed:
                    const += c * fixed[j]
                else:
                    a[j] = a.get(j, 0) + c
            if hi is not None:
                ineqs.append((dict(a), hi - const))
            if lo is not None:
                ineqs.append(({k: -v for k, v in a.items()}, const - lo))
        for j in cont:
            lo_j, hi_j = blo[j], bhi[j]
            if lo_j is not None:
                ineqs.append(({j: F(-1)}, -lo_j))
            if hi_j is not None:
                ineqs.append(({j: F(1)}, hi_j))
        for a, b in ineqs:
            if not a and b < 0:
                return False
        return _fm_feasible([(a, b) for a, b in ineqs if a])

    def dfs(k: int) -> bool:
        """True if some completion of ``fixed`` is feasible."""
        nodes[0] += 1
        if nodes[0] > EXACT_MAX_NODES or (stop_at is not None and time.perf_counter() >= stop_at):
            raise _Undecided
        if pruned():
            return False
        if k == len(order):
            return leaf_feasible()
        j = order[k]
        for v in dom[j]:
            fixed[j] = F(v)
            if dfs(k + 1):
                return True
            del fixed[j]
        return False

    try:
        return not dfs(0)
    except _Undecided:
        return None
