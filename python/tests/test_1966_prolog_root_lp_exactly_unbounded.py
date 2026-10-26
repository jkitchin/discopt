"""#1966: prolog's root McCormick LP is exactly unbounded below over its double data.

#1966 asked for ``exact_ns_bound`` to converge on this LP. It cannot, and no dual
correction can: the LP has a rational recession direction ``d >= 0``, supported on
columns open above, with ``A d = 0`` exactly and ``c^T d = -1.7053e-13 < 0``. A
finite lower bound for it would be false. The direction comes from the model's own
data. Rows 12-13 let ``x18 <= fl(3340.8) x7`` and ``x19 <= fl(371.2) x7``, the
objective prices ``x7`` at ``3712``, and in binary ``fl(3340.8) + fl(371.2)`` exceeds
``3712`` by ``1.7053e-13`` (in decimal the two are equal and the ray has zero cost).

The pre-#1956 certificate (``prolog`` "optimal" with ``gap_certified=True``) came
from a Neumaier-Shcherbina bound that rounded column 7's reduced cost to ``0.0``. It
was never a proof. This file pins the ray so that no later change "fixes"
``exact_ns_bound`` into certifying it.
"""

from __future__ import annotations

import os
from fractions import Fraction

import discopt.modeling as dm
import discopt.solvers.lp_milp_highs as L
import numpy as np
import pytest
from discopt._relax import mccormick_lp as ML
from discopt.tightening import fbbt_box
from scipy.optimize import linprog

pytestmark = pytest.mark.unit

PROLOG = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "prolog.nl")


def _root_lp(monkeypatch):
    """The ``(y, std-form)`` pairs the verified route hands ``exact_ns_bound`` for the
    root node LP, plus the node result.

    The node relaxation is captured while ``solve_at_node`` builds it and then handed
    to the verified route directly, so the probe measures the LP itself whatever
    status the node engine reports (``uncertified``, or ``iteration_limit`` once the
    native routes gate ``optimal`` on an NS bound, #1953).
    """
    monkeypatch.setenv("DISCOPT_FREE_COLUMN_LP_CERT", "1")
    from discopt._relax.milp_relaxation import MilpRelaxationModel

    milps: list = []
    real_solve = MilpRelaxationModel.solve

    def solve_spy(self, *a, **kw):
        milps.append(self)
        return real_solve(self, *a, **kw)

    monkeypatch.setattr(MilpRelaxationModel, "solve", solve_spy)
    m = dm.from_nl(PROLOG)
    bt = fbbt_box(m)
    res = ML.MccormickLPRelaxer(m).solve_at_node(bt.lb, bt.ub)
    assert milps, "solve_at_node never solved a node LP: the probe measured nothing"

    seen: list = []
    real = L.exact_ns_bound

    def spy(y, sf, **kw):
        out = real(y, sf, **kw)
        seen.append((np.asarray(y).copy(), sf, out))
        return out

    monkeypatch.setattr(L, "exact_ns_bound", spy)
    verified = ML._verified_free_column_bound(milps[0], 30.0)
    assert seen, "the verified route never reached exact_ns_bound: the probe measured nothing"
    return res, verified, seen


def _exact_null_vector(cols: list[int], sf, pin: int) -> list[Fraction]:
    """The rational solution of ``A[:, cols] d = 0`` with ``d[pin] = 1`` (1-D nullspace)."""
    A = sf.A.tocsc()
    rows = sorted({int(i) for j in cols for i in A[:, j].indices})
    free = [j for j in cols if j != pin]
    # Augmented system M z = rhs over the touched rows: A_free z = -A_pin.
    M = [[Fraction(float(A[i, j])) for j in free] + [-Fraction(float(A[i, pin]))] for i in rows]
    n = len(free)
    r = 0
    piv_cols = []
    for c in range(n):
        p = next((k for k in range(r, len(M)) if M[k][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][c]
        M[r] = [v * inv for v in M[r]]
        for k in range(len(M)):
            if k != r and M[k][c] != 0:
                f = M[k][c]
                M[k] = [a - f * b for a, b in zip(M[k], M[r])]
        piv_cols.append(c)
        r += 1
    assert len(piv_cols) == n, "the support is not a 1-D nullspace"
    assert all(row[n] == 0 for row in M[r:]), "A[:, cols] d = 0 has no solution with d[pin] = 1"
    d = dict(zip(free, (M[k][n] for k in range(n))))
    d[pin] = Fraction(1)
    return [d[j] for j in cols]


def test_the_root_lp_has_an_exact_negative_cost_recession_ray(monkeypatch):
    _, _, seen = _root_lp(monkeypatch)
    _y, sf, _out = seen[0]
    A = sf.A.tocsc()
    open_hi = np.flatnonzero(sf.xu >= L.READBACK_LIMIT)
    assert not np.any(sf.xl <= -L.READBACK_LIMIT), "prolog's open sides are all upper"
    assert 7 in open_hi

    # Floating point only proposes the support; everything below is exact.
    bounds = [(0, None)] * len(open_hi)
    bounds[list(open_hi).index(7)] = (1, 1)
    lp = linprog(sf.c[open_hi], A_eq=A[:, open_hi], b_eq=np.zeros(sf.m), bounds=bounds)
    assert lp.status == 0, lp.message
    support = [int(open_hi[i]) for i in np.flatnonzero(lp.x > 1e-12)]
    d = _exact_null_vector(support, sf, pin=7)

    assert all(v >= 0 for v in d), "the direction leaves the box (an open side only)"
    for i in range(sf.m):  # A d = 0 on EVERY row, exactly
        row = A[i, support].toarray().ravel()
        assert sum(Fraction(float(a)) * v for a, v in zip(row, d)) == 0
    cost = sum(Fraction(float(sf.c[j])) * v for j, v in zip(support, d))
    assert cost < 0, cost
    assert float(cost) == pytest.approx(-1.7053025658242404e-13, rel=1e-6)
    # ... and that is exactly the binary excess of the model's two pricing coefficients.
    assert Fraction(3340.8) + Fraction(371.2) - Fraction(3712) == -cost


def test_exact_ns_bound_abstains_and_the_root_lp_stays_uncertified(monkeypatch):
    """Abstaining is the only correct answer on an exactly unbounded LP."""
    res, verified, seen = _root_lp(monkeypatch)
    assert all(out[0] is None for _y, _sf, out in seen), [out for *_, out in seen]
    assert verified == (None, None), verified
    assert res.status != "optimal" and res.lower_bound is None, res
