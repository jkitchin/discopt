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

import math
import time
from fractions import Fraction
from typing import Optional

import numpy as np

#: Largest number of integer assignments the DFS may visit (leaves + internal).
EXACT_MAX_NODES = 1 << 18
#: Fourier--Motzkin is doubly exponential in the eliminated columns; past these
#: caps the decision is left undecided.
EXACT_MAX_CONT = 12
#: Integer columns in active rows. The route is for small sets -- MIPcc27's
#: kedron / osun / otra hold 2 / 14 / 16 -- and a set with thousands (dives)
#: would only spend the budget on a decision it cannot reach.
EXACT_MAX_INTS = 40
EXACT_MAX_INEQS = 20_000


class _Undecided(Exception):
    pass


def _to_int_ineq(a: dict[int, Fraction], b: Fraction) -> tuple[dict[int, int], int]:
    """``a.x <= b`` multiplied by the least common denominator (a positive
    multiple, so the same half-space) and divided by the gcd of all its integers.
    No rounding: the columns need not be integer."""
    den = 1
    for q in list(a.values()) + [b]:
        den = den * q.denominator // math.gcd(den, q.denominator)
    ai = {k: int(v * den) for k, v in a.items()}
    bi = int(b * den)
    g = 0
    for v in ai.values():
        g = math.gcd(g, abs(v))
    g = math.gcd(g, abs(bi)) if g else abs(bi)
    if g > 1:
        ai = {k: v // g for k, v in ai.items()}
        bi //= g
    return ai, bi


def _fm_project(
    ineqs: list[tuple[dict[int, int], int]],
    elim: list[int],
    stop_at: Optional[float] = None,
) -> Optional[list[tuple[dict[int, int], int]]]:
    """Eliminate the columns ``elim`` from ``a.x <= b`` (integer data) by
    Fourier--Motzkin: the result holds for the other columns iff some real values
    of ``elim`` satisfy the input. ``None``: a constant row ``0 <= b < 0`` (the
    input is infeasible for every value of the other columns). Raises
    ``_Undecided`` past ``EXACT_MAX_INEQS`` or ``stop_at``, both checked while
    the pairwise combinations are formed (review of #2113: checked only after a
    full product, 20 rows x 12 columns ran 29 s past a 1 s stop).

    Kohler's rule (Chernikov 1960, Kohler 1967) keeps the growth in check: after
    ``k`` eliminations a derived inequality combining more than ``k + 1`` input
    rows is implied by the others, so dropping it is exact. Without it 13 rows
    over 4 eliminated columns passed 20,000.
    """
    # key -> the smallest set of input rows it was derived from
    cur: dict[tuple, frozenset] = {}
    for idx, (a0, b0) in enumerate(ineqs):
        cur.setdefault((tuple(sorted(a0.items())), b0), frozenset({idx}))
    for k, j in enumerate(elim, start=1):
        pos, neg = [], []
        rest: dict[tuple, frozenset] = {}
        for key, orig in cur.items():
            row = dict(key[0])
            c = row.get(j, 0)
            if c > 0:
                pos.append((row, key[1], orig))
            elif c < 0:
                neg.append((row, key[1], orig))
            else:
                rest[key] = orig
        for ap, bp, op in pos:
            if stop_at is not None and time.perf_counter() >= stop_at:
                raise _Undecided
            if len(rest) > EXACT_MAX_INEQS:
                raise _Undecided
            cp = ap[j]
            for an, bn, on in neg:
                orig = op | on
                if len(orig) > k + 1:
                    continue  # Kohler: implied by the others
                cn = -an[j]
                a: dict[int, int] = {}
                for c2, v in ap.items():
                    if c2 != j:
                        a[c2] = a.get(c2, 0) + v * cn
                for c2, v in an.items():
                    if c2 != j:
                        a[c2] = a.get(c2, 0) + v * cp
                a = {c2: v for c2, v in a.items() if v != 0}
                b = bp * cn + bn * cp
                if not a:
                    if b < 0:
                        return None
                    continue
                g = 0
                for v in a.values():
                    g = math.gcd(g, abs(v))
                g = math.gcd(g, abs(b)) if b else g
                if g > 1:
                    a = {c2: v // g for c2, v in a.items()}
                    b //= g
                key = (tuple(sorted(a.items())), b)
                if key not in rest or len(orig) < len(rest[key]):
                    rest[key] = orig
        if len(rest) > EXACT_MAX_INEQS:
            raise _Undecided
        cur = rest
    return [(dict(key[0]), key[1]) for key in cur]


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

    The continuous columns are projected out once, up front (Fourier--Motzkin
    over integer-scaled rows: ``Fraction(float)`` is exact and every double is
    dyadic, so each row scales to integers exactly). What remains is an
    equivalent system over the held integer columns alone, whose rows have finite
    activity ranges, so the search can prune: fixing a column updates only the
    rows that contain it. (Deciding the continuous part per leaf instead visited
    all 65,536 assignments of MIPcc27 otra's 16 binaries, ~19 s per set.)
    """
    A = A.tocsr()
    rows = [r for r in range(A.shape[0]) if np.isfinite(rlo[r]) or np.isfinite(rhi[r])]
    is_held = np.zeros(lb.size, dtype=bool)
    is_held[held] = True
    # Size caps first, from the structure alone (no exact arithmetic yet): the
    # engine asks before every presolve-free solve, mostly about large sets.
    if rows:
        used_cols = np.unique(
            np.concatenate([A.indices[A.indptr[r] : A.indptr[r + 1]] for r in rows])
        )
        n_int = int(np.count_nonzero(is_held[used_cols]))
        if n_int > EXACT_MAX_INTS or used_cols.size - n_int > EXACT_MAX_CONT:
            return None
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
    ineqs: list[tuple[dict[int, int], int]] = []
    used: set[int] = set()
    for r in rows:
        s, e = A.indptr[r], A.indptr[r + 1]
        terms = {int(j): F(float(v)) for j, v in zip(A.indices[s:e], A.data[s:e]) if v != 0}
        used.update(terms)
        if np.isfinite(rhi[r]):
            ineqs.append(_to_int_ineq(terms, F(float(rhi[r]))))
        if np.isfinite(rlo[r]):
            ineqs.append(_to_int_ineq({k: -v for k, v in terms.items()}, -F(float(rlo[r]))))
    ints = sorted(j for j in used if is_held[j])
    cont = sorted(j for j in used if not is_held[j])
    if len(cont) > EXACT_MAX_CONT or len(ints) > EXACT_MAX_INTS:
        return None
    for j in ints:
        if not (np.isfinite(lb[j]) and np.isfinite(ub[j])):
            return None  # unbounded integer: no finite enumeration
    for j in cont:
        if np.isfinite(lb[j]):
            ineqs.append(_to_int_ineq({j: F(-1)}, -F(float(lb[j]))))
        if np.isfinite(ub[j]):
            ineqs.append(_to_int_ineq({j: F(1)}, F(float(ub[j]))))
    # Constant rows (no columns) decide themselves.
    if any(not a and b < 0 for a, b in ineqs):
        return True
    ineqs = [(a, b) for a, b in ineqs if a]
    try:
        proj = _fm_project(ineqs, cont, stop_at)
    except _Undecided:
        return None
    if proj is None:
        return True
    # Search the integer columns. Each projected row: sum a_j x_j <= b.
    dom = {j: (int(np.ceil(lb[j])), int(np.floor(ub[j]))) for j in ints}
    n_rows = len(proj)
    rows_of: dict[int, list[tuple[int, int]]] = {j: [] for j in ints}
    lo_act = [0] * n_rows  # minimum activity over the current boxes
    for i, (a, b) in enumerate(proj):
        for j, c in a.items():
            rows_of[j].append((i, c))
            lo_j, hi_j = dom[j]
            lo_act[i] += c * (lo_j if c > 0 else hi_j)
    rhs = [b for _, b in proj]
    if any(lo_act[i] > rhs[i] for i in range(n_rows)):
        return True
    order = sorted(ints, key=lambda j: (-len(rows_of[j]), j))
    if not order:
        return False  # projected rows hold with no integer column left
    nodes = [0]

    def fix(j: int, v: int) -> bool:
        """Fix x_j = v (from its box); False if some row's minimum now exceeds it."""
        ok = True
        lo_j, hi_j = dom[j]
        for i, c in rows_of[j]:
            lo_act[i] += c * v - c * (lo_j if c > 0 else hi_j)
            if lo_act[i] > rhs[i]:
                ok = False
        return ok

    def unfix(j: int, v: int) -> None:
        lo_j, hi_j = dom[j]
        for i, c in rows_of[j]:
            lo_act[i] -= c * v - c * (lo_j if c > 0 else hi_j)

    def tick() -> None:
        nodes[0] += 1
        if nodes[0] > EXACT_MAX_NODES or (stop_at is not None and time.perf_counter() >= stop_at):
            raise _Undecided

    # Iterative DFS: stack[k] is the next value to try for column order[k].
    try:
        k = 0
        val = [dom[order[0]][0]]
        while k >= 0:
            j = order[k]
            if val[k] > dom[j][1]:
                val.pop()
                k -= 1
                if k >= 0:
                    unfix(order[k], val[k])
                    val[k] += 1
                continue
            tick()
            if fix(j, val[k]):
                if k + 1 == len(order):
                    return False  # every projected row holds: feasible
                k += 1
                val.append(dom[order[k]][0])
                continue
            unfix(j, val[k])
            val[k] += 1
    except _Undecided:
        return None
    return True
