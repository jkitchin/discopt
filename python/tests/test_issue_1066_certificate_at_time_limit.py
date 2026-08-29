"""#1066: a closed gap on a valid bound is a certificate even at the wall.

``solve_lp_nlp_bb`` decided its exit status from *how the loop ended* and only
then, on two of the branches, looked at the gap. A run cut off by the clock was
reported ``feasible`` even when the master's own dual bound already certified the
incumbent -- measured on ``squfl015-060`` at default settings: dual bound
366.6218167383147 against incumbent 366.62181673996474, a relative gap of 4.5e-14
against the 1e-4 default, reported ``feasible``.

The decision now lives in :func:`_lp_nlp_bb_exit_status`, which these tests drive
directly at every combination its inputs can take.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.oa import _lp_nlp_bb_exit_status, solve_lp_nlp_bb

TOL = 1e-4


def _status(**over):
    kw = dict(
        converged_early=False,
        callback_terminated=False,
        master_status=SolveStatus.OPTIMAL,
        has_incumbent=True,
        master_bound_valid=True,
        gap=0.5,
        gap_tolerance=TOL,
        hook_stopped_before_wall=False,
    )
    kw.update(over)
    return _lp_nlp_bb_exit_status(**kw)


def test_a_closed_gap_at_the_wall_is_optimal():
    """The regression: the clock ended the loop, but the certificate stands."""
    status, reason = _status(callback_terminated=True, gap=4.5e-14)
    assert status == "optimal"
    # The search really did stop on the clock and the trace keeps saying so --
    # status answers "is it proven", the reason answers "why did the loop end".
    assert reason == "time_limit"


def test_an_open_gap_at_the_wall_is_still_feasible():
    """The kill criterion: the upgrade must not fire on an unconverged run."""
    status, reason = _status(callback_terminated=True, gap=0.5)
    assert status == "feasible"
    assert reason == "time_limit"


def test_a_closed_gap_on_an_invalid_bound_is_not_a_certificate():
    """``master_bound_valid`` is the whole warrant for the upgrade."""
    assert _status(callback_terminated=True, gap=0.0, master_bound_valid=False)[0] == "feasible"


def test_no_bound_at_all_is_not_a_certificate():
    assert _status(callback_terminated=True, gap=None)[0] == "feasible"


def test_no_incumbent_at_the_wall_stays_time_limit():
    status, reason = _status(callback_terminated=True, has_incumbent=False, gap=None)
    assert status == "time_limit"
    assert reason == "time_limit"


def test_the_hook_is_named_when_it_stopped_before_the_wall():
    status, reason = _status(callback_terminated=True, hook_stopped_before_wall=True, gap=0.5)
    assert (status, reason) == ("feasible", "termination_hook")


def test_the_early_exit_keeps_its_own_reason():
    status, reason = _status(converged_early=True, gap=0.0)
    assert (status, reason) == ("optimal", "gap_tolerance")


@pytest.mark.parametrize(
    "master_status,expected",
    [
        (SolveStatus.INFEASIBLE, "infeasible"),
        (SolveStatus.TIME_LIMIT, "feasible"),
        (SolveStatus.ITERATION_LIMIT, "feasible"),
        (SolveStatus.OPTIMAL, "feasible"),
    ],
)
def test_master_status_branches_with_an_open_gap(master_status, expected):
    """Every non-callback branch keeps the meaning it had before the split."""
    assert _status(master_status=master_status, gap=0.5)[0] == expected


def test_an_iteration_limit_without_an_incumbent_is_reported_as_such():
    assert (
        _status(master_status=SolveStatus.ITERATION_LIMIT, has_incumbent=False, gap=None)[0]
        == "iteration_limit"
    )


def test_no_feasible_point_when_nothing_ended_the_loop():
    assert (
        _status(master_status=SolveStatus.ERROR, has_incumbent=False, gap=None)[0]
        == "no_feasible_point"
    )


def test_the_driver_actually_consults_the_helper(monkeypatch):
    """Anti-vacuity (CLAUDE.md §6): a helper nothing calls proves nothing.

    Every assertion above is on a pure function. This one solves a real convex
    MINLP through ``solve_lp_nlp_bb`` and fails if the driver reached its exit
    without asking.
    """
    calls: list[dict] = []
    import discopt.solvers.oa as oa

    real = oa._lp_nlp_bb_exit_status

    def spy(**kw):
        calls.append(dict(kw))
        return real(**kw)

    monkeypatch.setattr(oa, "_lp_nlp_bb_exit_status", spy)

    m = dm.Model("convex_minlp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.binary("y")
    m.subject_to(x >= 2 * y)
    m.subject_to(x + y >= 1.5)
    m.minimize((x - 3) ** 2 + y)
    result = solve_lp_nlp_bb(m, time_limit=30.0, gap_tolerance=TOL)

    assert len(calls) == 1, f"the driver exited without consulting the helper: {calls}"
    assert calls[0]["gap_tolerance"] == TOL
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-6)
