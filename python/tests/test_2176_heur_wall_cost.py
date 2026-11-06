"""#2176: ``DISCOPT_HEUR_WALL_COST`` -- the improver contingent priced by measured wall.

The #1344 contingent admits an improver-role call while
``spent + cost <= offset + quot * nodes * weight``. ``cost`` is a fixed number per
heuristic (RINS 5, local branching 10, enumeration 12). On GDPlib batch_processing
hull, 6 RINS calls took 54.8 s against 11 s for the tree's 159 nodes, yet were charged
about 30 nodes' worth. ON keeps the formula and measures both sides:

* one unit is the loop's node-processing wall (``rust_time + jax_time``, which excludes
  the root heuristics) per processed node;
* a call costs its heuristic's mean measured wall, timed from
  ``_improver_clock_start`` at the call;
* a first call keeps the fixed-cost test.

Default ON since 2026-11-06 (#2176 panel); ``=0`` drops ``walls`` from the state and every
function behaves exactly as before.
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


def _on_state(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "1")
    return S._new_heur_state()


def _measured(st, key, wall):
    """Record one call of ``key`` that took ``wall`` seconds."""
    st["started"][key] = 0.0
    real = S.time.perf_counter
    try:
        S.time.perf_counter = lambda: wall
        S._record_improver_run(st, 5.0, False, key=key)
    finally:
        S.time.perf_counter = real


def test_flag_is_default_on(monkeypatch):
    monkeypatch.delenv("DISCOPT_HEUR_WALL_COST", raising=False)
    assert S._heur_wall_cost_enabled() is True  # graduated 2026-11-06 (#2176)
    assert set(S._new_heur_state()) == {"calls", "found", "cost", "walls", "wall", "started"}
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "0")
    assert S._heur_wall_cost_enabled() is False
    assert S._new_heur_state() == {"calls": 0, "found": 0, "cost": 0.0}


def test_off_state_ignores_the_clock_and_the_key(monkeypatch):
    """Flag OFF (``=0``): no ``walls``, so the clock is a no-op and the test is fixed-cost."""
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "0")
    st = S._new_heur_state()
    S._improver_clock_start(st, "rins")
    S._record_improver_run(st, 5.0, False, key="rins")
    assert st == {"calls": 1, "found": 0, "cost": 5.0}
    tree = _Tree(100)  # contingent 0.5 * 100 * 3 * 1/2 = 75 units
    assert S._improver_within_contingent(
        70.0, budget_on=True, tree=tree, heur_state=st, key="rins", node_wall=1.0, **_KNOBS
    )


def test_first_call_uses_the_fixed_cost_test(monkeypatch):
    st = _on_state(monkeypatch)
    tree = _Tree(10)  # contingent 15 units; nothing measured for "rins" yet
    kw = dict(budget_on=True, tree=tree, heur_state=st, key="rins", node_wall=1.0, **_KNOBS)
    assert S._improver_within_contingent(5.0, **kw)
    assert not S._improver_within_contingent(15.1, **kw)


def test_an_expensive_measured_heuristic_is_refused(monkeypatch):
    """The batch_processing shape: a 9 s RINS against 11 s of node wall over 159 nodes.

    Fixed-cost test: 5 + 5 <= 0.5 * 159 * 3 * (0+1)/(1+1) = 119.25 units -> admitted.
    In seconds the contingent is 119.25 * (11 / 159) = 8.25 s; 9 + 9 exceeds it.
    """
    st = _on_state(monkeypatch)
    _measured(st, "rins", 9.0)
    tree = _Tree(159)
    fixed = {k: v for k, v in st.items() if k not in ("walls", "wall", "started")}
    assert S._improver_within_contingent(  # the fixed-cost test alone admits it
        5.0, budget_on=True, tree=tree, heur_state=fixed, **_KNOBS
    )
    assert not S._improver_within_contingent(
        5.0, budget_on=True, tree=tree, heur_state=st, key="rins", node_wall=11.0, **_KNOBS
    )


def test_a_cheap_measured_heuristic_is_admitted(monkeypatch):
    st = _on_state(monkeypatch)
    _measured(st, "rins", 0.05)
    assert S._improver_within_contingent(
        5.0, budget_on=True, tree=_Tree(159), heur_state=st, key="rins", node_wall=11.0, **_KNOBS
    )


def test_the_unit_is_the_node_wall_passed_in_not_elapsed_time(monkeypatch):
    """Root heuristics (an LOA proposer, pumps) are outside ``node_wall``.

    Review on jkitchin/discopt#2175: deriving the unit from elapsed time made a 60 s
    root LOA count as tree wall, so at 3 nodes one "node" cost ~20 s and a 9 s RINS
    looked cheap. With the node wall passed in (0.3 s for 3 nodes), it is refused.
    """
    st = _on_state(monkeypatch)
    _measured(st, "rins", 9.0)
    tree = _Tree(3)
    kw = dict(budget_on=True, tree=tree, heur_state=st, key="rins", **_KNOBS)
    assert not S._improver_within_contingent(5.0, node_wall=0.3, **kw)
    assert S._improver_within_contingent(5.0, node_wall=60.3, **kw)  # what elapsed time gave


def test_without_a_node_wall_the_fixed_cost_test_stands(monkeypatch):
    st = _on_state(monkeypatch)
    _measured(st, "rins", 9.0)
    assert S._improver_within_contingent(
        5.0, budget_on=True, tree=_Tree(159), heur_state=st, key="rins", **_KNOBS
    )


def test_a_record_without_a_running_clock_charges_no_wall(monkeypatch):
    """Only the span from ``_improver_clock_start`` is billed (review on #2175)."""
    st = _on_state(monkeypatch)
    S._record_improver_run(st, 5.0, False, key="rins")
    assert st["walls"] == {} and st["wall"] == 0.0
    assert (st["calls"], st["cost"]) == (1, 5.0)


def test_recording_keeps_walls_per_heuristic(monkeypatch):
    st = _on_state(monkeypatch)
    _measured(st, "rins", 2.0)
    _measured(st, "lbranch", 1.0)
    _measured(st, "rins", 4.0)
    assert st["walls"] == {"rins": [2.0, 4.0], "lbranch": [1.0]}
    assert st["wall"] == pytest.approx(7.0)
    assert st["started"] == {}


def test_the_finder_role_is_still_never_gated(monkeypatch):
    st = _on_state(monkeypatch)
    _measured(st, "rins", 1e6)
    assert S._improver_within_contingent(
        5.0,
        budget_on=True,
        tree=_Tree(1, incumbent=None),
        heur_state=st,
        key="rins",
        node_wall=1e-3,
        **_KNOBS,
    )
