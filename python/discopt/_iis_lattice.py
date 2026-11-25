"""Exact lattice (integer Farkas) infeasibility test for equality rows.

A system ``A_I x_I + A_C x_C = b`` with ``x_I`` integer and ``x_C`` real has no
solution iff, after projecting ``x_C`` out through a basis ``N`` of the left null
space of ``A_C`` (``N A_C = 0``), the integer system ``N A_I x_I = N b`` has none;
and that is decided exactly by unimodular column operations (a Hermite-normal-form
style elimination), the integer analogue of Farkas' lemma (Kannan and Bachem 1979;
Schrijver 1986, Cor. 4.1a). Bounds and inequality rows play no part, so a "no" is a
proof of infeasibility for any subsystem that contains these rows and holds these
integralities, whatever its bounds -- and a "yes" says nothing about the bounded
system.

Everything is integer arithmetic on Python ints: only rows whose coefficients and
right-hand side are integers enter (a row left out only weakens the test, never
makes it unsound).

Generalizes the mod-2 test the IIS engine used before (a parity contradiction is the
case ``y = comb / 2``), which also required every column of a row to be held
integer; this one treats a relaxed or continuous column as real and eliminates it.
On MIPcc27's smallset the full equality system is lattice-infeasible on enlight4,
enlight11 and neos-807454 and on no other instance; the test costs <= 0.03 s on each.
"""

from __future__ import annotations

import time
from math import gcd
from typing import Optional

import numpy as np

_INF = 1e20

#: Work cap per test (row operations plus column operations). The test returns
#: ``None`` (no conclusion) past it, so a large equality system cannot stall a probe.
#: The smallset's hits need well under 10^5.
MAX_WORK = 2_000_000


class _Budget(Exception):
    pass


def integer_rows(A, row_lo, row_hi) -> dict[int, tuple[dict[int, int], int]]:
    """``{row: ({col: coef}, rhs)}`` for every equality row with a finite rhs and
    integer coefficients and rhs: the rows the test may use. ``A`` is CSR."""
    out: dict[int, tuple[dict[int, int], int]] = {}
    lo = np.asarray(row_lo, dtype=float)
    hi = np.asarray(row_hi, dtype=float)
    for r in np.flatnonzero((lo == hi) & (np.abs(lo) < _INF)):
        b = lo[r]
        if b != round(b):
            continue
        s, e = A.indptr[r], A.indptr[r + 1]
        vals = A.data[s:e]
        if e == s or not bool(np.all(vals == np.round(vals))):
            continue
        out[int(r)] = ({int(j): int(v) for j, v in zip(A.indices[s:e], vals) if v}, int(b))
    return out


def _norm(a: dict[int, int], c: dict[int, int]) -> None:
    g = 0
    for v in a.values():
        g = gcd(g, v)
    for v in c.values():
        g = gcd(g, v)
    if g > 1:
        for k in a:
            a[k] //= g
        for k in c:
            c[k] //= g


def _left_null(rows_c: list[dict[int, int]], tick) -> list[dict[int, int]]:
    """An integer basis of the rational left null space of the matrix whose rows
    are ``rows_c``: each element maps row index -> integer weight."""
    piv: dict[int, tuple[dict[int, int], dict[int, int]]] = {}
    null: list[dict[int, int]] = []
    for i, row in enumerate(rows_c):
        a = dict(row)
        c = {i: 1}
        while a:
            j = min(a)
            if j not in piv:
                piv[j] = (a, c)
                break
            pa, pc = piv[j]
            f, g = pa[j], a[j]  # a <- f * a - g * pa eliminates column j
            tick(len(a) + len(pa) + len(c) + len(pc))
            for k in list(a):
                a[k] *= f
            for k, v in pa.items():
                nv = a.get(k, 0) - g * v
                if nv:
                    a[k] = nv
                else:
                    a.pop(k, None)
            for k in list(c):
                c[k] *= f
            for k, v in pc.items():
                nv = c.get(k, 0) - g * v
                if nv:
                    c[k] = nv
                else:
                    c.pop(k, None)
            _norm(a, c)
        else:
            null.append(c)
    return null


def _int_solvable(M: list[dict[int, int]], rhs: list[int], ncols: int, tick) -> bool:
    """Does ``M z = rhs`` have an integer solution? Column operations (unimodular)
    bring ``M`` to lower-triangular form row by row; the system is solvable iff
    each pivot divides its residual and every row without a pivot has a zero
    residual."""
    k = len(M)
    cols: list[list[int]] = [[0] * k for _ in range(ncols)]
    for i, r in enumerate(M):
        for j, v in r.items():
            cols[j][i] = v
    res = list(rhs)
    p = 0
    for i in range(k):
        act = [j for j in range(p, ncols) if cols[j][i] != 0]
        while len(act) > 1:
            act.sort(key=lambda j: abs(cols[j][i]))
            c0 = cols[act[0]]
            for j in act[1:]:
                cj = cols[j]
                q = cj[i] // c0[i]
                if q:
                    tick(k - i)
                    for t in range(i, k):
                        if c0[t]:
                            cj[t] -= q * c0[t]
            act = [j for j in act if cols[j][i] != 0]
        if act:
            j0 = act[0]
            cols[p], cols[j0] = cols[j0], cols[p]
            h = cols[p][i]
            if res[i] % h:
                return False
            z = res[i] // h
            if z:
                cp = cols[p]
                tick(k - i)
                for t in range(i, k):
                    if cp[t]:
                        res[t] -= z * cp[t]
            p += 1
        elif res[i] != 0:
            return False
    return True


def lattice_infeasible(
    rows: list[tuple[dict[int, int], int]],
    held,
    stop_at: Optional[float] = None,
    max_work: int = MAX_WORK,
) -> Optional[bool]:
    """``True``: the equality rows ``rows`` (``({col: coef}, rhs)``, integer data)
    have no solution with the columns in ``held`` integer and every other column
    real, bounds ignored -- an exact proof. ``False``: they have one. ``None``: the
    work cap or ``stop_at`` (``time.perf_counter()``) was reached first."""
    if not rows:
        return False
    work = [0]

    def tick(n: int) -> None:
        work[0] += n
        if work[0] > max_work or (
            stop_at is not None and work[0] % 4096 < n and time.perf_counter() > stop_at
        ):
            raise _Budget

    try:
        rows_c = [{j: v for j, v in r.items() if j not in held} for r, _ in rows]
        null = _left_null(rows_c, tick)
        if not null:
            return False
        icols = sorted({j for r, _ in rows for j in r if j in held})
        ix = {j: t for t, j in enumerate(icols)}
        M: list[dict[int, int]] = []
        c: list[int] = []
        for y in null:
            acc: dict[int, int] = {}
            b = 0
            for i, w in y.items():
                r, ri = rows[i]
                b += w * ri
                tick(len(r))
                for j, v in r.items():
                    t = ix.get(j)
                    if t is not None:
                        acc[t] = acc.get(t, 0) + w * v
            M.append({t: v for t, v in acc.items() if v})
            c.append(b)
        return not _int_solvable(M, c, len(icols), tick)
    except _Budget:
        return None


def lattice_core(
    rows_by_con: dict[int, tuple[dict[int, int], int]],
    held: set[int],
    orders: int = 8,
    stop_at: Optional[float] = None,
) -> Optional[tuple[set[int], set[int]]]:
    """A set of constraints and held integer columns that is minimal for the
    lattice test (dropping any one constraint or integrality makes it
    lattice-feasible), or ``None`` when the whole system is not lattice-infeasible
    or the work cap / ``stop_at`` cut every order short.

    ``rows_by_con`` maps a constraint id to its single integer equality row. The
    test is monotone (dropping a row or an integrality never makes a feasible
    system infeasible), so one row pass then one integrality pass give a minimal
    set; which minimal set depends on the order, so ``orders`` deterministic orders
    are tried (see :func:`_filter`) and the smallest result is kept."""
    cons = sorted(rows_by_con)
    if lattice_infeasible([rows_by_con[c] for c in cons], held, stop_at) is not True:
        return None
    best: Optional[tuple[set[int], set[int]]] = None
    for k in range(max(1, orders)):
        if stop_at is not None and time.perf_counter() > stop_at:
            break
        res = _filter(rows_by_con, cons, held, k, stop_at)
        if res is not None and (
            best is None or len(res[0]) + len(res[1]) < len(best[0]) + len(best[1])
        ):
            best = res
    return best


def _filter(rows_by_con, cons_all, held, k, stop_at):
    """One deletion filter with the lattice test as its oracle: rows first, all
    integralities held (cheap: nothing to project out), by halving blocks so that a
    small certificate in a large system costs O(size * log n) tests; then the
    integralities of the columns the rows touch, one at a time. Order ``k`` = 0
    takes rows by index and integralities by column index; ``k`` >= 1 shuffles both
    with ``np.random.default_rng(k)``."""
    rng = np.random.default_rng(k)

    def ok(cs, its) -> Optional[bool]:
        return lattice_infeasible([rows_by_con[c] for c in sorted(cs)], its, stop_at)

    order = list(cons_all)
    if k:
        rng.shuffle(order)
    cons = list(order)
    size = max(1, len(cons) // 2)
    while True:
        i = 0
        while i < len(cons):
            block = set(cons[i : i + size])
            rest = [c for c in cons if c not in block]
            v = ok(rest, held) if rest else False
            if v is None:
                return None
            if v:
                cons = rest
            else:
                i += size
        if size == 1:
            break
        size = max(1, size // 2)
    touched: set[int] = set()
    for c in cons:
        touched.update(rows_by_con[c][0])
    ints = set(held) & touched
    iorder = sorted(ints)
    if k:
        rng.shuffle(iorder)
    for j in iorder:
        v = ok(cons, ints - {j})
        if v is None:
            return None
        if v:
            ints.discard(j)
    return set(cons), ints
