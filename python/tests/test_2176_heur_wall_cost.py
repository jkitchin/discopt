"""#2176: ``DISCOPT_HEUR_WALL_COST`` -- the improver contingent priced by measured wall.

The #1344 contingent admits an improver-role call while
``spent + cost <= offset + quot * nodes * weight``. ``cost`` is a fixed number per
heuristic (RINS 5, local branching 10, enumeration 12). On GDPlib batch_processing
hull, 6 RINS calls took 54.8 s against 11 s for the tree's 159 nodes, yet were charged
about 30 nodes' worth. ON keeps the formula and measures both sides. One unit is the
search's non-improver wall per node; a call costs its heuristic's mean measured wall.
The first call keeps the fixed-cost test. Default OFF; with no ``walls`` in the state,
every function behaves exactly as before.
"""

from __future__ import annotations

import pytest
from discopt import solver as S

pytestmark = pytest.mark.smoke

_KNOBS = dict(success_gain=3.0, offset=0.0, quot=0.5)


class _Tree:
    def __init__(self, nodes, incumbent=(None, 1.0)):
        self._nodes, self._inc = nodes, incumbent

    def incumbent(self):
        return self._inc

    def stats(self):
        return {"total_nodes": self._nodes}


def _wall_state(t0=0.0, wall=0.0, walls=None, **kw):
    st = {"calls": 0, "found": 0, "cost": 0.0, "walls": walls or {}, "wall": wall, "t0": t0}
    st.update(kw)
    return st


def test_flag_is_default_off(monkeypatch):
    monkeypatch.delenv("DISCOPT_HEUR_WALL_COST", raising=False)
    assert S._heur_wall_cost_enabled() is False
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "1")
    assert S._heur_wall_cost_enabled() is True


def test_without_walls_the_key_changes_nothing():
    """Flag OFF: the state has no ``walls``, so ``key``/``wall`` are inert."""
    st = {"calls": 0, "found": 0, "cost": 0.0}
    S._record_improver_run(st, 5.0, False, key="rins", wall=9.0)
    assert st == {"calls": 1, "found": 0, "cost": 5.0}
    tree = _Tree(100)  # contingent 0.5 * 100 * 3 * 1/2 = 75 units
    assert S._improver_within_contingent(
        70.0, budget_on=True, tree=tree, heur_state=st, key="rins", **_KNOBS
    )


def test_first_call_uses_the_fixed_cost_test():
    st = _wall_state()
    tree = _Tree(10)  # contingent 15 units; nothing measured for "rins" yet
    assert S._improver_within_contingent(
        5.0, budget_on=True, tree=tree, heur_state=st, key="rins", now=100.0, **_KNOBS
    )
    assert not S._improver_within_contingent(
        15.1, budget_on=True, tree=tree, heur_state=st, key="rins", now=100.0, **_KNOBS
    )


def test_an_expensive_measured_heuristic_is_refused():
    """The batch_processing shape: a 9 s RINS against a tree at ~0.07 s per node.

    159 nodes in 11 s of search wall; one 9 s call already recorded. Fixed-cost test:
    5 + 5 <= 0.5 * 159 * 3 * (0+1)/(1+1) = 119 units -> admitted. In seconds the
    contingent is 119.25 * (11 / 159) = 8.25 s, and 9 + 9 exceeds it -> refused.
    """
    st = _wall_state(t0=0.0)
    S._record_improver_run(st, 5.0, False, key="rins", wall=9.0)
    tree = _Tree(159)
    now = 11.0 + 9.0  # 11 s of search plus the 9 s the call took
    assert S._improver_within_contingent(  # the fixed-cost test alone admits it
        5.0, budget_on=True, tree=tree, heur_state={**st, "walls": None}, **_KNOBS
    )
    assert not S._improver_within_contingent(
        5.0, budget_on=True, tree=tree, heur_state=st, key="rins", now=now, **_KNOBS
    )


def test_a_cheap_measured_heuristic_is_admitted():
    """A 0.05 s call against the same tree fits the same contingent easily."""
    st = _wall_state(t0=0.0)
    S._record_improver_run(st, 5.0, False, key="rins", wall=0.05)
    tree = _Tree(159)
    assert S._improver_within_contingent(
        5.0, budget_on=True, tree=tree, heur_state=st, key="rins", now=11.05, **_KNOBS
    )


def test_improver_wall_is_not_counted_as_search_wall():
    """The unit is non-improver wall per node, so heuristics cannot buy their own budget."""
    st = _wall_state(t0=0.0)
    for _ in range(4):
        S._record_improver_run(st, 5.0, False, key="rins", wall=100.0)
    tree = _Tree(1000)
    # 10 s of search, 400 s of RINS: per node 0.01 s; contingent 0.5*1000*3/5*0.01 = 3 s.
    assert not S._improver_within_contingent(
        5.0, budget_on=True, tree=tree, heur_state=st, key="rins", now=410.0, **_KNOBS
    )


def test_recording_keeps_walls_per_heuristic():
    st = _wall_state()
    S._record_improver_run(st, 5.0, True, key="rins", wall=2.0)
    S._record_improver_run(st, 10.0, False, key="lbranch", wall=1.0)
    S._record_improver_run(st, 5.0, False, key="rins", wall=4.0)
    assert st["walls"] == {"rins": [2.0, 4.0], "lbranch": [1.0]}
    assert st["wall"] == pytest.approx(7.0)
    assert (st["calls"], st["found"], st["cost"]) == (3, 1, 20.0)


def test_the_finder_role_is_still_never_gated():
    st = _wall_state()
    S._record_improver_run(st, 5.0, False, key="rins", wall=1e6)
    assert S._improver_within_contingent(
        5.0,
        budget_on=True,
        tree=_Tree(1, incumbent=None),
        heur_state=st,
        key="rins",
        now=1.0,
        **_KNOBS,
    )
