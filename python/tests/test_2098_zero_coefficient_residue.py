"""#2098: an exactly-zero like-term sum keeps no float residue.

``_sum_is_benign`` used to judge a coefficient whose exact value is zero against
the row's largest exact coefficient, and keep a residue below ``2**-26`` of that
scale. In ``((t + x) - t) - x + y == k`` with ``t = 2**53 * x`` the residue is
``-1`` on ``x``: the row reads ``-x + y == k``, ``y <= k`` forces ``x == 0``, and
the certified bound 0 excludes every feasible ``x > 0`` (truth 10). Each model here
certified 0 on ``a624e0f`` (``k = 1e6`` happened to pass: the constant set the scale).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

M = 2.0**53

pytestmark = pytest.mark.smoke


def _assert_certified_ten(r):
    assert r.objective == pytest.approx(10.0, abs=1e-6)
    if r.gap_certified:
        assert r.bound >= 10.0 - 1e-6


@pytest.mark.parametrize("k", [1e6, 1e9, 1e13])
def test_zero_sum_next_to_a_large_constant(k):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=k)
    t = M * x
    m.subject_to(((t + x) - t) - x + y == k)  # exactly y == k: x is free
    m.maximize(x)
    _assert_certified_ten(m.solve(time_limit=30, threads=1))


def test_zero_sum_next_to_a_large_coefficient():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=1)
    t = M * x
    m.subject_to(1e13 * y + ((t + x) - t) - x == 1e13)  # exactly y == 1
    m.maximize(x)
    _assert_certified_ten(m.solve(time_limit=30, threads=1))
