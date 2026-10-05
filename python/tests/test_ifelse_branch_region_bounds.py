"""``if_else`` encloses each branch over the region where it is selected.

#1043's own repro, ``if_else(x >= 0, exp(x) - 1, log(-x + 3)) <= 1`` over
``x in [-10, 10]``, raised on a default solve: ``log(-x + 3)`` has no finite
whole-box enclosure, so the aux fell back to the +/-9.999e19 box, which the hull
refuses. The else branch is only ever evaluated where ``x <= 0``.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import _condition_box


def test_aux_bounds_come_from_each_branch_region():
    m = dm.Model("ifb")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    w = m.if_else(x >= 0, dm.exp(x) - 1, dm.log(-x + 3))
    lo, hi = float(w.lb), float(w.ub)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= 0.0 and hi >= math.exp(10.0) - 1 - 1e-6
    # sound: every value either branch takes in its own region is enclosed
    n = 0
    for xv in np.linspace(-10, 10, 2001):
        val = math.exp(xv) - 1 if xv >= 0 else math.log(-xv + 3)
        assert lo - 1e-12 <= val <= hi + 1e-9 * (1 + abs(val))
        n += 1
    assert n == 2001


def test_condition_box_is_sound_and_declines_nonlinear():
    m = dm.Model("cb")
    x = m.continuous("x", lb=-4.0, ub=6.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    box = _condition_box(x + 2 * y <= 1)
    assert box is not None and float(box[x].hi) == pytest.approx(1.0)
    assert y not in box or float(box[y].hi) <= 2.0
    rng = np.random.default_rng(0)
    checks = 0
    for _ in range(2000):
        xv, yv = rng.uniform(-4, 6), rng.uniform(0, 2)
        if xv + 2 * yv <= 1:
            assert float(box[x].lo) <= xv <= float(box[x].hi)
            checks += 1
    assert checks > 0
    assert _condition_box(x * y <= 1) is None
    assert _condition_box(None) is None


def test_the_1043_repro_solves():
    m = dm.Model("issue1043")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.maximize(x)
    m.subject_to(dm.if_else(x >= 0, dm.exp(x) - 1, dm.log(-x + 3)) <= 1.0)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(math.log(2.0), abs=1e-6)
