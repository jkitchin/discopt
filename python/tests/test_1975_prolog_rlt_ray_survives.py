"""#1975: RLT against an unbounded factor's lower bound does not close prolog's root ray.

#1966 pinned an exact negative-cost recession ray of prolog's root McCormick LP.
#1975 proposed the relaxation-side fix (hypothesis H): multiply each linear row that
touches only finite-box columns (prolog's shared budgets ``x2 + x3 <= 0.94 S1 -
0.244 S3`` etc.) by the lower-bound factor ``(x_k - l_k) >= 0`` of every product
factor ``x_k`` that is unbounded above. Such a row is valid with ``x_k`` open, and it
gives the LP the coupling between ``x_k*x2`` and ``x_k*x3`` that the nonlinear model
has.

The entry experiment falsified H on the probe itself. This file pins it. The rows
are built **generically** from the captured root ``MilpRelaxationModel``, and no
prolog-specific row or column is named. The exact ray search is then re-run:

* ``budget``: H as stated. Only the rows that carry a product's partner column are
  multiplied by that factor.
* ``all``: every finite-box row times every unbounded product factor. This is the
  strongest relaxation of this family.

Both leave an exact rational ray ``d >= 0`` on the FBBT box and on the root-OBBT
box, with ``A d <= 0`` on every row and ``c^T d = 3712 - fl(3340.8) - fl(371.2) =
-1.7053e-13``. The reason is structural. In the scaled limit the RLT rows bind
``x0*x_j`` and ``x1*x_j`` to two **independent** points of the bounded polytope.
The nonlinear model binds both to the same point. That shared point is the whole
margin. Measured in #1975, the nonlinear model's own asymptotic slope is
``min (3712 x7 + 5000 x8 - P) / (x7 + x8) = +0.0349`` (SCIP global, optimal, gap 0,
cross-checked by HiGHS LPs at fixed ``(x0, x1)/T``). So prolog is bounded below
over its doubles, but with a relative margin of ~1e-5 that only a relaxation keeping
that shared point can see.
"""

from __future__ import annotations

import os
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax import mccormick_lp as ML
from discopt._relax.milp_relaxation import MilpRelaxationModel
from discopt._relax.obbt import run_obbt
from discopt.tightening import fbbt_box
from scipy.optimize import linprog

pytestmark = pytest.mark.unit

PROLOG = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "prolog.nl")
EXCESS = Fraction(3340.8) + Fraction(371.2) - Fraction(3712)  # 1.7053e-13 > 0


def _capture(monkeypatch, box: str) -> tuple[MilpRelaxationModel, int]:
    """The root node LP ``solve_at_node`` builds over the FBBT (or FBBT + OBBT) box."""
    milps: list = []
    real_solve = MilpRelaxationModel.solve

    def solve_spy(self, *a, **kw):
        milps.append(self)
        return real_solve(self, *a, **kw)

    monkeypatch.setattr(MilpRelaxationModel, "solve", solve_spy)
    m = dm.from_nl(PROLOG)
    bt = fbbt_box(m)
    lb, ub = np.asarray(bt.lb, float), np.asarray(bt.ub, float)
    if box == "obbt":
        ob = run_obbt(m, lb, ub)
        assert ob.n_tightened > 0, "root OBBT tightened nothing: the OBBT arm measures FBBT"
        lb, ub = np.maximum(lb, ob.tightened_lb), np.minimum(ub, ob.tightened_ub)
    ML.MccormickLPRelaxer(m).solve_at_node(lb, ub)
    assert milps, "solve_at_node never solved a node LP: the probe measured nothing"
    return milps[0], len(lb)


def _products(A: sp.csr_matrix, n_orig: int, n_cols: int) -> dict[tuple[int, int], int]:
    """``(k, j) -> w`` for every lifted bilinear column, read off its McCormick rows."""
    Ac = A.tocsc()
    out: dict[tuple[int, int], int] = {}
    for w in range(n_orig, n_cols):
        # its envelope rows are the ones naming no other lifted column
        rows = [i for i in Ac[:, w].indices if all(j < n_orig or j == w for j in A[i].indices)]
        orig = sorted({int(j) for i in rows for j in A[i].indices if j < n_orig})
        assert len(orig) == 2, f"column {w} is not a bilinear product: {orig}"
        out[(orig[0], orig[1])] = out[(orig[1], orig[0])] = w
    return out


def _augment(M: MilpRelaxationModel, n_orig: int, variant: str):
    """``(c, A, b, lo, hi)`` of the root LP plus the RLT rows of H."""
    A = sp.csr_matrix(M._A_ub)
    b = [float(v) for v in M._b_ub]
    c = [float(v) for v in M._c]
    lo = [float(l_) for l_, _ in M._bounds]
    hi = [float(u) for _, u in M._bounds]
    n = A.shape[1]
    prod = _products(A, n_orig, n)
    finite = [np.isfinite(lo[j]) and np.isfinite(hi[j]) for j in range(n_orig)]
    factors = sorted(
        {k for k, _ in prod if np.isfinite(lo[k]) and not np.isfinite(hi[k])}
    )  # unbounded-above product factors: prolog's x0, x1
    assert factors, "no unbounded product factor: the probe measures nothing"
    budget = [
        i
        for i in range(A.shape[0])
        if all(j < n_orig and finite[j] for j in A[i].indices) and len(A[i].indices) > 1
    ]
    assert budget, "no finite-box row: the probe measures nothing"

    rows: list[tuple[dict[int, float], float]] = [
        ({int(j): float(v) for j, v in zip(A[i].indices, A[i].data)}, b[i])
        for i in range(A.shape[0])
    ]
    n_rlt = 0
    for k in factors:
        partners = {j for (kk, j) in prod if kk == k}
        for i in budget:
            a = {int(j): float(v) for j, v in zip(A[i].indices, A[i].data)}
            if variant == "budget" and not partners & a.keys():
                continue
            # (x_k - l_k)(b_i - a.x) >= 0:  sum_j a_j w_kj - b_i x_k - l_k a.x <= -l_k b_i
            lk = lo[k]
            row: dict[int, float] = {k: -b[i]}
            for j, aj in a.items():
                w = prod.get((k, j))
                if w is None:  # lift x_k*x_j: McCormick in x_k's lower bound factor only
                    w = len(c)
                    c.append(0.0)
                    lo.append(-np.inf)
                    hi.append(np.inf)
                    prod[(k, j)] = prod[(j, k)] = w
                    # (x_k - l_k)(x_j - l_j) >= 0 and (x_k - l_k)(u_j - x_j) >= 0
                    rows.append(({k: lo[j], j: lk, w: -1.0}, lk * lo[j]))
                    rows.append(({w: 1.0, k: -hi[j], j: -lk}, -lk * hi[j]))
                row[w] = row.get(w, 0.0) + aj
                row[j] = row.get(j, 0.0) - lk * aj
            rows.append((row, -lk * b[i]))
            n_rlt += 1
    assert n_rlt > 0, "no RLT row was emitted: the probe measures nothing"
    return c, rows, lo, hi, n_rlt


def _exact_ray(c, rows, lo, hi, pin: int) -> list[Fraction]:
    """An exact recession direction with ``d[pin] = 1``, or an assertion failure.

    Floating point proposes a vertex of the pinned recession cone. The vertex is
    then recomputed in rationals from its active set.
    """
    n = len(c)
    A = np.zeros((len(rows), n))
    for i, (r, _b) in enumerate(rows):
        for j, v in r.items():
            A[i, j] += v
    bounds = []
    for j in range(n):
        lo_j = 0.0 if np.isfinite(lo[j]) else None  # the cone: a finite side stays put
        hi_j = 0.0 if np.isfinite(hi[j]) else None
        bounds.append((1.0, 1.0) if j == pin else (lo_j, hi_j))
    lp = linprog(c, A_ub=A, b_ub=np.zeros(len(rows)), bounds=bounds, method="highs-ds")
    assert lp.status == 0, lp.message
    x = lp.x
    # Candidate equations, tightest first: the pin, the zero columns, then the rows
    # the float vertex holds tight. A degenerate vertex holds more than ``n``, and
    # rounding can make them inconsistent, so keep a linearly independent basis
    # (every row is re-checked exactly by the caller).
    cand: list[tuple[list[float], int]] = []
    for j in range(n):
        if j == pin or abs(x[j]) <= 1e-12:
            e = [0.0] * n
            e[j] = 1.0
            cand.append((e, 1 if j == pin else 0))
    slack = [(abs(r @ x) / (1.0 + np.abs(r).sum()), i) for i, r in enumerate(A)]
    cand += [(list(A[i]), 0) for s_, i in sorted(slack) if s_ <= 1e-9]
    forced: list[tuple[list[float], int]] = []
    for _repair in range(len(rows) + 1):
        d = _solve_basis(forced + cand, n)
        # Active-set repair: a row the float tolerance missed, violated exactly by
        # the rational point, is forced into the next basis.
        worst = max(range(len(rows)), key=lambda i: _row_value(rows[i][0], d))
        if _row_value(rows[worst][0], d) <= 0:
            return d
        forced.append((list(A[worst]), 0))
    raise AssertionError("active-set repair did not converge to an exact ray")


def _row_value(r: dict[int, float], d: list[Fraction]) -> Fraction:
    return sum((Fraction(a) * d[j] for j, a in r.items()), Fraction(0))


def _solve_basis(cand: list[tuple[list[float], int]], n: int) -> list[Fraction]:
    """The rational point of the first ``n`` linearly independent candidate equations."""
    basis: list[tuple[list[float], int]] = []
    for e, h in cand:
        if np.linalg.matrix_rank(np.array([r for r, _ in basis] + [e])) > len(basis):
            basis.append((e, h))
        if len(basis) == n:
            break
    assert len(basis) == n, "the float vertex's active set does not determine a point"
    M = [[Fraction(v) for v in r] + [Fraction(h)] for r, h in basis]
    for col in range(n):  # Gauss-Jordan on the square rational system
        p = next(k for k in range(col, n) if M[k][col] != 0)
        M[col], M[p] = M[p], M[col]
        inv = 1 / M[col][col]
        M[col] = [v * inv for v in M[col]]
        for k in range(n):
            if k != col and M[k][col] != 0:
                f = M[k][col]
                M[k] = [a - f * bb for a, bb in zip(M[k], M[col])]
    return [M[i][n] for i in range(n)]


@pytest.mark.parametrize("box", ["fbbt", "obbt"])
@pytest.mark.parametrize("variant", ["budget", "all"])
def test_rlt_rows_leave_an_exact_negative_cost_ray(monkeypatch, box, variant):
    M, n_orig = _capture(monkeypatch, box)
    c, rows, lo, hi, n_rlt = _augment(M, n_orig, variant)
    assert n_rlt == (2 if variant == "budget" else 10), n_rlt
    d = _exact_ray(c, rows, lo, hi, pin=7)

    checks = 0
    for j, v in enumerate(d):  # stays in the cone: a finite side is never crossed
        if np.isfinite(lo[j]):
            assert v >= 0, (j, v)
        if np.isfinite(hi[j]):
            assert v <= 0, (j, v)
        checks += 1
    for r, _b in rows:  # A d <= 0 on EVERY row, exactly
        assert sum(Fraction(a) * d[j] for j, a in r.items()) <= 0
        checks += 1
    assert checks == len(d) + len(rows)
    cost = sum(Fraction(cj) * dj for cj, dj in zip(c, d))
    assert cost < 0, cost
    # ... and the cost is the model's binary pricing excess, as on the bare LP (#1966).
    assert cost == -EXCESS * d[7]
