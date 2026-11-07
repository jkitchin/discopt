"""#2216: under ``deterministic=True`` the improver governor must not price by wall.

#2176 (``DISCOPT_HEUR_WALL_COST``, default ON since 2026-11-06) admits an
improver-role call (RINS, local branching, binary-seed enumeration) when
``spent_wall + expected_wall <= contingent * node_wall / nodes``: both sides are
measured seconds. That is a **role-2** gate by #912's definition. It decides how
much work runs, and its answer is a function of machine speed and load. The
other role-2 gates were already neutralized under ``deterministic`` (#1116,
#1187, #2120); this one was added after them and was not.

Measured on ``cvxnonsep_nsig30`` at ``deterministic=True, max_nodes=2000,
time_limit=120``, ``PYTHONHASHSEED=0``, four repeats in one process:

* flag ON: 39 wall-priced decisions per solve, **four different decision
  sequences** (21/21/21/20 calls admitted), and several decisions within 0.3 % of
  their threshold (lhs/rhs 0.9971, 0.9989);
* flag OFF: one decision sequence in all four repeats;
* the two incumbents the issue reports, 130.6287126437295 (ON on this machine)
  and 130.62871265397885 (OFF), at the same 165 nodes and dual bound
  130.61841392487696. Which local-branching calls run picks between them.

Under ``deterministic`` the governor now uses the fixed-cost test, so it decides
exactly as ``DISCOPT_HEUR_WALL_COST=0`` does. Off ``deterministic`` the flag is
unchanged.
"""

from __future__ import annotations

import dataclasses

import discopt.solver_tuning as solver_tuning
import pytest
from discopt import solver as S

pytestmark = pytest.mark.smoke

_KNOBS = dict(success_gain=3.0, offset=0.0, quot=0.5)


class _Tree:
    def __init__(self, nodes):
        self._nodes = nodes

    def incumbent(self):
        return (None, 1.0)

    def stats(self):
        return {"total_nodes": self._nodes}


def _tuning(deterministic):
    return solver_tuning.set_current(
        dataclasses.replace(solver_tuning.SolverTuning(), deterministic=deterministic)
    )


def _run_governor(wall_per_call):
    """Admit/refuse sequence for 6 RINS calls that each take ``wall_per_call`` s.

    Returns the verdicts and how many were priced by wall. The node wall is fixed
    at 1 s over 100 nodes, so the fixed-cost test admits all six (contingent 75
    units at success weight 1/2 down to 3/14, cost 5 each); the wall-priced test
    refuses a call that costs more than about 0.75 s / 0.32 s.
    """
    st = S._new_heur_state()
    tree = _Tree(100)
    verdicts, priced = [], 0
    for _ in range(6):
        walls = st.get("walls")
        priced += bool(walls and walls.get("rins"))
        ok = S._improver_within_contingent(
            5.0, budget_on=True, tree=tree, heur_state=st, key="rins", node_wall=1.0, **_KNOBS
        )
        verdicts.append(ok)
        if ok:
            S._improver_clock_start(st, "rins")
            started = st.get("started")
            if started is not None:  # emulate a call that took ``wall_per_call``
                started["rins"] -= wall_per_call
            S._record_improver_run(st, 5.0, False, key="rins")
    return verdicts, priced


def test_deterministic_state_has_no_timing_fields(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "1")
    token = _tuning(True)
    try:
        assert S._new_heur_state() == {"calls": 0, "found": 0, "cost": 0.0}
    finally:
        solver_tuning.reset_current(token)


def test_flag_still_applies_off_deterministic(monkeypatch):
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "1")
    token = _tuning(False)
    try:
        assert {"walls", "wall", "started"} <= set(S._new_heur_state())
    finally:
        solver_tuning.reset_current(token)


def test_deterministic_decisions_do_not_depend_on_measured_wall(monkeypatch):
    """The #2216 defect in miniature: before the fix the verdicts below moved with
    the measured wall of each call. Under ``deterministic`` they must not."""
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "1")
    token = _tuning(False)
    try:
        fast, priced_fast = _run_governor(0.01)
        slow, priced_slow = _run_governor(5.0)
    finally:
        solver_tuning.reset_current(token)
    # Anti-vacuity (CLAUDE.md §6): off deterministic the wall test is consulted and
    # does move with the measured wall, so the comparison below can fail.
    assert priced_fast > 0 and priced_slow > 0
    assert fast != slow

    token = _tuning(True)
    try:
        det_fast, p1 = _run_governor(0.01)
        det_slow, p2 = _run_governor(5.0)
    finally:
        solver_tuning.reset_current(token)
    assert p1 == p2 == 0, "a deterministic solve priced an improver by wall"
    assert det_fast == det_slow
    # ... and matches the fixed-cost governor (``DISCOPT_HEUR_WALL_COST=0``).
    monkeypatch.setenv("DISCOPT_HEUR_WALL_COST", "0")
    token = _tuning(False)
    try:
        fixed, _ = _run_governor(5.0)
    finally:
        solver_tuning.reset_current(token)
    assert det_fast == fixed
