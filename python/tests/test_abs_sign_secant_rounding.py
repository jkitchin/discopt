"""The ``abs`` secant and ``sign`` rows must contain the graph exactly.

The ``abs`` secant used ``slope = (|hi| - |lo|)/(hi - lo)``, ``a = |lo| - slope*lo``.
When ``|lo| >> hi > 0`` the slope rounds to ``-1`` and ``a`` to ``0`` (true
intercept ``~2*hi``), so with ``w >= x`` the row forced ``x <= 0``.
``maximize x + 0.5*|x|`` on ``[-1e15, 0.05]`` certified ``0`` as optimal, though
``x = 0.05`` gives ``0.075``. Neither the abs secant nor the ``sign`` rows were
outward-rounded.
"""

from __future__ import annotations

from fractions import Fraction

import discopt._relax.uniform_relax as U
import discopt.modeling as dm
import numpy as np
import pytest

_ABS_BOXES = [
    (-1e15, 0.05),
    (-4.58e-11, 4.67e8),
    (-465312.5, 6.65e-6),
    (-0.05, 1e15),
    (-3.0, 7.0),
]


def _graph_points_satisfy_rows(build, f, lo, hi):
    m = dm.Model()
    x = m.continuous("x", lb=lo, ub=hi)
    m.minimize(build(x))
    r = U.build_uniform_relaxation(m, box=(np.array([lo]), np.array([hi])))
    A = r.model._A_ub.toarray()
    b = r.model._b_ub
    assert A.shape[1] == 2, A.shape  # x and the single aux column
    n = 0
    for t in [lo, hi, 0.0, 0.5 * (lo + hi), lo + 1e-3 * (hi - lo)]:
        if not lo <= t <= hi:
            continue
        w = f(t)
        for row, rhs in zip(A, b):
            lhs = Fraction(float(row[0])) * Fraction(t) + Fraction(float(row[1])) * Fraction(w)
            assert lhs <= Fraction(float(rhs)), (lo, hi, t, row, rhs, float(lhs - Fraction(rhs)))
            n += 1
    return n


@pytest.mark.parametrize("lo,hi", _ABS_BOXES)
def test_abs_rows_contain_graph_exactly(lo, hi):
    assert _graph_points_satisfy_rows(dm.abs, abs, lo, hi) >= 9


@pytest.mark.parametrize("c", [1e6, 1e15])
def test_sign_rows_contain_graph_exactly_with_large_constant(c):
    lo, hi = c - 1.0, c + 1.0

    def sgn(t):
        d = Fraction(t) - Fraction(c)
        return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)

    assert _graph_points_satisfy_rows(lambda x: dm.sign(x - c), sgn, lo, hi) >= 8


def test_abs_wide_box_maximize_is_not_falsely_certified():
    m = dm.Model()
    x = m.continuous("x", lb=-1e15, ub=0.05)
    m.maximize(x + 0.5 * dm.abs(x))
    r = m.solve(time_limit=60)
    # x = 0.05 is feasible with objective 0.075; a certified bound below it is false.
    assert r.bound >= 0.075 - 1e-6, (r.status, r.objective, r.bound)
    if r.status == "optimal":
        assert r.objective == pytest.approx(0.075, abs=1e-6)
