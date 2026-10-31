"""``if_else``'s auxiliary box is re-derived at every solve and save (#2061 review).

``w = if_else(c, a, b)`` gets ``w``'s bounds from an interval enclosure of the two
branches over the box at declaration. Widening a variable a branch reads
afterwards left that box in place, so ``maximize(w)`` on ``x in [0, 1]`` widened to
``x <= 10`` was certified ``optimal`` at 1 against a true 10. This is the same
stale-derived-bound defect the review found in the complementarity lowerings.
"""

from __future__ import annotations

import copy
import pickle

import discopt.modeling as dm
import pytest
from discopt.serialize import dumps, loads

pytestmark = pytest.mark.smoke


def _model():
    m = dm.Model("ifw")
    x = m.continuous("x", lb=0.0, ub=1.0)
    w = m.if_else(x >= 0.5, x, 2 * x)
    m.maximize(w)
    return m, x, w


def test_widening_after_declaration_does_not_cut_the_optimum():
    m, x, w = _model()
    x.ub = 10.0
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(10.0, abs=1e-6)


def test_a_second_solve_still_sees_the_widening():
    # The solve boundary restores every box on exit; the refresh must not mistake
    # that restored box for a bound the user set on w.
    m, x, w = _model()
    x.ub = 10.0
    for _ in range(2):
        r = m.solve(time_limit=30)
        assert r.objective == pytest.approx(10.0, abs=1e-6)


def test_solve_leaves_the_callers_box_alone_and_refresh_is_idempotent():
    m, x, w = _model()
    before = (float(w.lb), float(w.ub))
    assert m._refresh_if_else_bounds(context="t") == 0
    m.solve(time_limit=30)
    assert (float(w.lb), float(w.ub)) == before


def test_bounds_the_user_set_on_w_are_kept():
    # w.ub = 0.75 is a model constraint, not a derivation: a refresh after a
    # widening intersects with it instead of replacing it.
    m, x, w = _model()
    w.ub = 0.75
    x.ub = 10.0
    r = m.solve(time_limit=30)
    assert r.objective == pytest.approx(0.75, abs=1e-6)


@pytest.mark.parametrize("how", ["deepcopy", "pickle"])
def test_widening_a_copy_refreshes_the_copy(how):
    m, x, w = _model()
    m2 = copy.deepcopy(m) if how == "deepcopy" else pickle.loads(pickle.dumps(m))
    x2 = next(v for v in m2._variables if v.name == "x")
    x2.ub = 10.0
    r = m2.solve(time_limit=30)
    assert r.objective == pytest.approx(10.0, abs=1e-6)


def test_save_writes_the_refreshed_box():
    m, x, w = _model()
    x.ub = 10.0
    m2 = loads(dumps(m))
    w2 = next(v for v in m2._variables if v.name == w.name)
    assert float(w2.ub) == pytest.approx(10.0)
    assert m2.solve(time_limit=30).objective == pytest.approx(10.0, abs=1e-6)


def test_load_recovers_the_derivation_and_refuses_a_widening():
    m, x, w = _model()
    m2 = loads(dumps(m))
    assert len(m2._if_else_auxes) == 1 and m2._if_else_auxes[0].last is None
    x2 = next(v for v in m2._variables if v.name == "x")
    # Narrowing: the saved box still encloses the branches. (x <= 0.4 keeps the
    # solve off the boundary x == 0.5, where both disjuncts hold and w = 2x = 1.)
    x2.ub = 0.4
    assert m2.solve(time_limit=30).objective == pytest.approx(0.8, abs=1e-6)
    x2.ub = 10.0
    with pytest.raises(ValueError, match="was read from a file"):
        m2.solve(time_limit=30)
    with pytest.raises(ValueError, match="was read from a file"):
        dumps(m2)


def test_nested_if_else_refreshes_in_declaration_order():
    m = dm.Model("nest")
    x = m.continuous("x", lb=0.0, ub=1.0)
    w1 = m.if_else(x >= 0.5, x, 0.0)
    w2 = m.if_else(x >= 0.25, w1, 0.0)
    m.maximize(w2)
    x.ub = 4.0
    r = m.solve(time_limit=30)
    assert r.objective == pytest.approx(4.0, abs=1e-6)
