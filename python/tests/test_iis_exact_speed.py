"""The exact decider projects continuous columns out first (``_iis_exact``).

The previous version enumerated every held-integer assignment and decided the
continuous part per leaf: on MIPcc27 otra's sets every row has an unbounded
continuous column, so nothing pruned and all 2^16 assignments were visited (~19
s per set, ~4 sets per run). Projecting the continuous columns out once
(Fourier--Motzkin with Kohler's rule, integer arithmetic) leaves rows over the
integers alone, which prune: ~10 ms per set.

The reference below is the previous implementation, verbatim; the new one must
return the same verdict on every set.
"""

import random
import time
from fractions import Fraction
from typing import Optional

import numpy as np
import scipy.sparse as sp
from discopt import _iis_exact
from discopt._iis_exact import exact_feasible

# ---- reference: the previous implementation, verbatim ----
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


def _reference_exact_feasible(
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
    if len(cont) > EXACT_MAX_CONT or len(ints) > EXACT_MAX_INTS:
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

    def visit() -> bool:
        """Count a node; True when ``fixed`` is pruned (no completion is feasible)."""
        nodes[0] += 1
        if nodes[0] > EXACT_MAX_NODES or (stop_at is not None and time.perf_counter() >= stop_at):
            raise _Undecided
        return pruned()

    def search() -> bool:
        """True if some completion of ``fixed`` is feasible. Iterative: one stack
        frame per integer column overflowed Python's recursion limit on a set
        with few continuous but thousands of integer columns (MIPcc27 dives)."""
        if visit():
            return False
        if not order:
            return leaf_feasible()
        stack = [iter(dom[order[0]])]  # stack[k]: values left for column order[k]
        while stack:
            k = len(stack) - 1
            j = order[k]
            v = next(stack[-1], None)
            if v is None:
                stack.pop()
                fixed.pop(j, None)
                continue
            fixed[j] = F(v)
            if visit():
                continue
            if k + 1 == len(order):
                if leaf_feasible():
                    return True
                continue
            stack.append(iter(dom[order[k + 1]]))
        return False

    try:
        return not search()
    except _Undecided:
        return None


# Rows of a candidate set from MIPcc27 neos-3135947-otra (7 rows, 16 binaries,
# 3 continuous columns), as the engine handed it to the decider.
_OTRA_ROWS = [
    [
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -500000.0,
        0.0,
        0.0,
        -500000.0,
        0.0,
        0.0,
    ],
    [
        -1.0,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        -0.4088882688,
        0.0,
        -500000.0,
        0.0,
        0.0,
        -500000.0,
        0.0,
    ],
    [
        -1.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        -100000.0,
        0.0,
        0.0,
        -500000.0,
        0.0,
        0.0,
        -500000.0,
    ],
    [
        1.0,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.4088882688,
        0.0,
        -500000.0,
        0.0,
        0.0,
        -500000.0,
        0.0,
    ],
    [
        1.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        100000.0,
        0.0,
        0.0,
        -500000.0,
        0.0,
        0.0,
        -500000.0,
    ],
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.408888269,
        -100000.0,
        0.0,
        -0.408888269,
        -100000.0,
    ],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
]
_OTRA_RLO = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -100000.4089, 2.0]
_OTRA_RHI = [-0.0, -0.0, -0.0, -0.0, -0.0, -100000.4089, 2.0]
_OTRA_LB = [-np.inf] + [0.0] * 12 + [-np.inf, 0.0, 0.0, -np.inf, 0.0, 0.0]
_OTRA_UB = [np.inf] + [1.0] * 12 + [np.inf, 1.0, 1.0, np.inf, 1.0, 1.0]
_OTRA_HELD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18]


def test_otra_set_is_decided_within_a_small_node_budget(monkeypatch):
    # The per-leaf search needs all 131,071 nodes here; the projection's rows
    # prune. 5,000 nodes is a budget only the new search fits in.
    monkeypatch.setattr(_iis_exact, "EXACT_MAX_NODES", 5000)
    got = exact_feasible(
        sp.csr_matrix(np.array(_OTRA_ROWS)),
        np.array(_OTRA_RLO),
        np.array(_OTRA_RHI),
        np.array(_OTRA_LB),
        np.array(_OTRA_UB),
        np.array(_OTRA_HELD),
    )
    assert got is True  # infeasible (rational enumeration agrees: all 65,536)


def test_kohler_rule_keeps_the_projection_small():
    # 13 inequalities over 4 eliminated columns passed 20,000 without it.
    A = sp.csr_matrix(
        np.array(
            [
                [-3, -1, 2, 0, 0.408888, 0.408888, 1, -0.25],
                [-3, 0.408888, 0.408888, -0.25, -1, 0.5, 0, 2],
                [2, 0, -3, 1e5, 1, 0.408888, 2, 0.408888],
                [7, 0.5, 2, -1, 2, -1e5, 0.408888, -3],
                [-1e5, -0.25, 2, 1e5, 2, -1e5, -0.25, 1e5],
            ]
        )
    )
    rlo = np.array([-np.inf, -np.inf, 3.0, -np.inf, -np.inf])
    rhi = np.array([1.33127768, 3.0, 4.74723185, -1.0, -1.0])
    lb = np.array([-1.0, -2.0, 0.0, 0.0, 0.0, -np.inf, -2.0, 0.0])
    ub = np.array([0.0, 2.0, 1.0, 2.0, 2.0, 1.0, 3.5, 1.0])
    held = np.arange(4)
    assert exact_feasible(A, rlo, rhi, lb, ub, held) is False
    assert _reference_exact_feasible(A, rlo, rhi, lb, ub, held) is False


def test_same_verdicts_as_the_previous_implementation():
    rng = random.Random(1)
    coefs = [0, 0, 1, -1, 2, -3, 0.5, -0.25, 1.5, 7, 1e5, -1e5, 0.408888]
    counts = {True: 0, False: 0, None: 0}
    for _ in range(800):
        ni, nc, m = rng.randint(0, 5), rng.randint(0, 3), rng.randint(1, 4)
        n = ni + nc
        if n == 0:
            continue
        A = sp.csr_matrix(np.array([[rng.choice(coefs) for _ in range(n)] for _ in range(m)]))
        rlo = np.array(
            [rng.choice([-np.inf, rng.randint(-5, 5), rng.uniform(-5, 5)]) for _ in range(m)]
        )
        rhi = np.array(
            [rng.choice([np.inf, rng.randint(-5, 5), rng.uniform(-5, 5)]) for _ in range(m)]
        )
        if rng.random() < 0.2:
            rhi = rlo.copy()
        lb = np.array(
            [
                float(rng.randint(-2, 0)) if j < ni else rng.choice([-np.inf, -2.0, 0.0, -1.5])
                for j in range(n)
            ]
        )
        ub = np.array(
            [
                float(rng.randint(0, 2)) if j < ni else rng.choice([np.inf, 2.0, 1.0, 3.5])
                for j in range(n)
            ]
        )
        held = np.arange(ni)
        want = _reference_exact_feasible(A, rlo, rhi, lb, ub, held)
        got = exact_feasible(A, rlo, rhi, lb, ub, held)
        assert got == want, (A.toarray(), rlo, rhi, lb, ub, ni)
        counts[got] += 1
    assert counts[True] > 100 and counts[False] > 100
