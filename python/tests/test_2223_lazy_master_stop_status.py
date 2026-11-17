"""Issue #2223 defect C: a stopped Rust lazy master must not report INFEASIBLE.

``solve_milp_with_lazy_cuts`` maps ``terminate_callback`` onto the driver's
``AfterSelect`` debug checkpoint. That checkpoint fires after ``export_batch`` has
popped the batch, and a ``Stop`` there used to break out without putting those
nodes back: with no incumbent yet the tree looked empty and the driver returned
``Infeasible`` -- a false infeasibility certificate for a search that never looked
at the popped boxes. The OA caller masks it today (it reads its own
``callback_terminated`` first), but the engine's verdict was wrong on its own.

Measured before the fix on the toy below (the issue's probe8): both stop modes
returned ``SolveStatus.INFEASIBLE`` with ``bound=None``; after it they return a
limit status with the valid bound ``-4.0``.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus

ms = pytest.importorskip("discopt.solvers.milp_simplex")

# min -x - y, x, y integer in [0, 3], x + y <= 4.5; lazy x + y <= 3. Optimum -3.
_C = np.array([-1.0, -1.0])
_A = np.array([[1.0, 1.0]])
_B = np.array([4.5])
_OPT = -3.0


def _solve(stop_rule):
    vetoes = [0]

    def lazy(x):
        if x[0] + x[1] > 3.0 + 1e-9:
            vetoes[0] += 1
            return [(np.array([1.0, 1.0]), 3.0)]
        return None

    def term(_snap):
        return stop_rule(vetoes[0])

    return ms.solve_milp_with_lazy_cuts(
        _C,
        A_ub=_A,
        b_ub=_B,
        bounds=[(0, 3), (0, 3)],
        integrality=np.array([1, 1]),
        time_limit=10,
        lazy_callback=lazy,
        terminate_callback=term,
    )


def test_unstopped_control_is_optimal():
    r = _solve(lambda _v: False)
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(_OPT, abs=1e-9)
    assert r.callback_stats["terminate_calls"] >= 1
    assert r.callback_stats["terminated"] is False


@pytest.mark.parametrize(
    "rule",
    [lambda _v: True, lambda v: v > 0],
    ids=["stop_at_first_checkin", "stop_after_first_veto"],
)
def test_stop_without_incumbent_is_not_infeasible(rule):
    r = _solve(rule)
    stats = r.callback_stats
    # The probe fired: the check-in ran and the stop was honoured (CLAUDE.md §6).
    assert stats["terminate_calls"] >= 1
    assert stats["terminated"] is True
    assert r.objective is None, "the stop must land before any incumbent"
    assert r.status != SolveStatus.INFEASIBLE, (
        "a terminate_callback stop was reported as an infeasibility proof"
    )
    assert r.status != SolveStatus.OPTIMAL
    assert r.bound is None or r.bound <= _OPT + 1e-9
