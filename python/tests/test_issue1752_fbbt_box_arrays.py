"""#1752: ``discopt.tightening.fbbt_box`` tightens the elements of array variables.

The block FBBT engine keeps one interval per variable block, so a row on one
element (``x[0] <= 2``) could not tighten anything and the array model came back
untouched while the same model over scalars was tightened.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.tightening import fbbt_box

pytestmark = pytest.mark.smoke


def _model(array: bool):
    m = dm.Model("f")
    if array:
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        a, b, c = x[0], x[1], x[2]
    else:
        a = m.continuous("a", lb=0, ub=10)
        b = m.continuous("b", lb=0, ub=10)
        c = m.continuous("c", lb=0, ub=10)
    m.subject_to(a <= 2)
    m.subject_to(a + b <= 7)
    m.subject_to(c <= 4)
    m.minimize(a + b + c)
    return m


@pytest.mark.parametrize("array", [True, False])
def test_array_matches_scalar(array):
    res = fbbt_box(_model(array))
    assert not res.infeasible
    np.testing.assert_allclose(res.lb, [0, 0, 0])
    np.testing.assert_allclose(res.ub, [2, 7, 4])
    assert res.n_tightened == 3


def test_box_keeps_every_feasible_point():
    m = _model(True)
    res = fbbt_box(m)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, (20000, 3))
    feas = pts[(pts[:, 0] <= 2) & (pts[:, 0] + pts[:, 1] <= 7) & (pts[:, 2] <= 4)]
    assert feas.shape[0] > 100
    assert np.all(feas >= res.lb) and np.all(feas <= res.ub)


def test_element_row_proves_infeasible():
    m = dm.Model("inf")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] >= 5)
    m.subject_to(x[0] <= 3)
    m.minimize(dm.sum(x))
    assert fbbt_box(m).infeasible
