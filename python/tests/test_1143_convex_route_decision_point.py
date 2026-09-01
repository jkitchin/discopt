"""Regression (#1143): the convex-MINLP route must not spend half the budget to
learn it is going to abstain.

The #1066 guard cannot fire before ``_CONVEX_ROUTE_BUDGET_FRACTION`` (half the
limit) by construction, so an abstaining route always costs at least that much --
on the cert panel routinely 3-8x the whole unrouted solve. The #1143 policy moves
the checkpoint to ``_CONVEX_ROUTE_DECISION_POINT_FRACTION`` and makes arriving
there *be* the verdict.

Why that is sound rather than a smaller constant: across 24 classified instances
the route either certifies fast or not at all. Every row it certifies, it
certifies within 2.96 s of a 60 s limit; every abstain costs >= 28 s. The
decision therefore does not need a model property (the root-gap gate is
falsified -- the won and abstained gaps overlap completely) or a gap trend (which
kept ``cvxnonsep_nsig30`` running 28.2 s while improving the whole way and still
failing). It needs to be taken early.

These tests pin the policy's *shape*, not wall-clock numbers: the flag's default,
both verdict branches, the untouched legacy path, and the caller override. The
wall-clock evidence lives in the PR and in ``docs/dev/performance-plan.md``.
"""

from __future__ import annotations

import pytest
from discopt.solver import (
    _CONVEX_ROUTE_BUDGET_FRACTION,
    _CONVEX_ROUTE_DECISION_POINT_FLOOR_S,
    _CONVEX_ROUTE_DECISION_POINT_FRACTION,
    _CONVEX_ROUTE_FALLBACK_FLOOR_S,
    _convex_route_decision_point_enabled,
    _route_progress_guard_options,
    _RouteProgressGuard,
)

_FLAG = "DISCOPT_CONVEX_ROUTE_DECISION_POINT"


@pytest.fixture
def flag(monkeypatch):
    """Set/clear the #1143 flag for one test."""

    def _set(on: bool | None):
        if on is None:
            monkeypatch.delenv(_FLAG, raising=False)
        else:
            monkeypatch.setenv(_FLAG, "1" if on else "0")

    return _set


# --------------------------------------------------------------------------- #
# The flag itself
# --------------------------------------------------------------------------- #


def test_the_policy_is_default_on_after_graduation(flag):
    """CLAUDE.md §5: bound-changing, so it shipped default-off and graduated on the
    panel recorded in `_convex_route_decision_point_enabled` -- cert-clean (0 lost,
    1 gained, 0 objective violations) and net-positive (3 faster, 1 slower)."""
    flag(None)
    assert _convex_route_decision_point_enabled() is True


def test_the_opt_out_restores_the_1066_policy(flag):
    """The `=0` escape hatch and the legacy path are kept intact (§5)."""
    flag(False)
    assert _convex_route_decision_point_enabled() is False
    g = _RouteProgressGuard(60.0)
    assert g.decision_point is False
    assert g.checkpoint == pytest.approx(60.0 * _CONVEX_ROUTE_BUDGET_FRACTION)


def test_the_flag_is_read_per_call_not_cached(flag):
    flag(None)
    assert _convex_route_decision_point_enabled() is True
    flag(True)
    assert _convex_route_decision_point_enabled() is True
    flag(False)
    assert _convex_route_decision_point_enabled() is False


# --------------------------------------------------------------------------- #
# Where the checkpoint lands
# --------------------------------------------------------------------------- #


def test_the_opt_out_keeps_the_1066_half_budget_checkpoint(flag):
    flag(False)
    g = _RouteProgressGuard(60.0)
    assert g.decision_point is False
    assert g.checkpoint == pytest.approx(60.0 * _CONVEX_ROUTE_BUDGET_FRACTION)


def test_flag_on_moves_the_checkpoint_to_the_decision_point(flag):
    """The defect in one line: 30 s of a 60 s budget spent to learn nothing."""
    flag(True)
    g = _RouteProgressGuard(60.0)
    assert g.decision_point is True
    assert g.checkpoint == pytest.approx(60.0 * _CONVEX_ROUTE_DECISION_POINT_FRACTION)
    assert g.checkpoint < 60.0 * _CONVEX_ROUTE_BUDGET_FRACTION


def test_the_fallback_floor_still_applies_under_the_new_policy(flag):
    """Below the floor a fallback cannot do anything useful, so handing over early
    would trade a routed result for nothing. The floor is not weakened here."""
    flag(True)
    g = _RouteProgressGuard(2.0)
    assert g.checkpoint == pytest.approx(min(2.0, _CONVEX_ROUTE_FALLBACK_FLOOR_S))


def test_an_explicit_check_fraction_still_wins(flag):
    """A caller that names its own fraction is not overridden by the flag."""
    flag(True)
    g = _RouteProgressGuard(60.0, check_fraction=0.25)
    assert g.decision_point is False
    assert g.checkpoint == pytest.approx(15.0)


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_reaching_the_decision_point_is_the_verdict(flag):
    """Even a HEALTHY, steadily-improving gap trend hands over.

    This is the discriminator against the #1066 policy, and it is the
    ``cvxnonsep_nsig30`` shape: its gap improves the whole way (0.298 -> 0.0013)
    and it still never certifies, so "is it improving?" is the wrong question --
    answering it correctly still costs half the budget.
    """
    flag(True)
    g = _RouteProgressGuard(60.0)
    g.history = [(0.5, 0.90), (1.0, 0.45), (2.0, 0.20), (5.0, 0.05)]  # improving fast
    stop, reason = g._verdict(6.0)
    assert stop is True
    assert "decision point" in reason and "#1143" in reason


def test_a_converged_route_is_left_to_its_own_termination(flag):
    """The convergence branch is preserved: stopping a route whose gap has closed
    would discard a result that is about to be returned anyway."""
    flag(True)
    g = _RouteProgressGuard(60.0)
    g.history = [(0.5, 0.9), (5.0, 0.0)]
    assert g._verdict(6.0) == (False, None)


def test_the_hook_still_does_not_fire_before_the_checkpoint(flag):
    """The decision point moves *when* the verdict is taken, not whether the hook
    respects it."""
    flag(True)
    g = _RouteProgressGuard(60.0)
    ctx = {
        "elapsed": 1.0,
        "relative_gap": 0.5,
        "current_dual_bound": 1.0,
        "current_primal_bound": 2.0,
    }
    assert g(ctx) is False  # 1.0 s < the 6.0 s decision point
    assert g.fired_at is None
    ctx = dict(ctx, elapsed=6.5)
    assert g(ctx) is True
    assert g.fired_at == pytest.approx(6.5)


# --------------------------------------------------------------------------- #
# The legacy path is untouched
# --------------------------------------------------------------------------- #


def test_flag_off_still_keeps_a_route_that_is_improving(flag):
    flag(False)
    g = _RouteProgressGuard(60.0)
    g.history = [(20.0, 0.90), (29.0, 0.10)]
    assert g._verdict(30.0) == (False, None)


def test_flag_off_still_stops_a_route_that_has_stalled(flag):
    flag(False)
    g = _RouteProgressGuard(60.0)
    g.history = [(20.0, 0.90), (29.0, 0.89)]
    stop, reason = g._verdict(30.0)
    assert stop is True
    assert "improvement" in reason


def test_flag_off_still_stops_when_there_is_no_trend_to_judge(flag):
    flag(False)
    g = _RouteProgressGuard(60.0)
    g.history = [(29.0, 0.5)]
    stop, reason = g._verdict(30.0)
    assert stop is True
    assert "no trend to judge" in reason


# --------------------------------------------------------------------------- #
# The master check-in has to follow, or the hook is never called to decide
# --------------------------------------------------------------------------- #


def test_the_master_checkin_deadline_follows_the_decision_point(flag):
    """Without this the policy is inert on exactly the instances it targets.

    ``fac2``'s hook is called at 0.06-0.16 s and then NOT AGAIN until 30.09 s: its
    first master runs the whole way and only returns at the check-in deadline. A
    lower checkpoint with the old deadline would never be evaluated.
    """
    flag(True)
    opts, guard = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
    assert guard is not None
    assert opts["master_checkin_deadline"] == pytest.approx(guard.checkpoint)
    assert opts["master_checkin_deadline"] == pytest.approx(
        60.0 * _CONVEX_ROUTE_DECISION_POINT_FRACTION
    )

    flag(False)
    opts_off, guard_off = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
    assert opts_off["master_checkin_deadline"] == pytest.approx(
        60.0 * _CONVEX_ROUTE_BUDGET_FRACTION
    )


def test_a_caller_supplied_hook_still_declines_the_guard(flag):
    """Two hooks racing to stop the same loop is not a contract worth having."""
    flag(True)
    opts, guard = _route_progress_guard_options(
        {"termination_hook": lambda ctx: False}, method_key="oa", time_limit=60.0
    )
    assert guard is None


# --------------------------------------------------------------------------- #
# The absolute floor: a fraction alone is unsound on a short budget
# --------------------------------------------------------------------------- #

#: The slowest route win measured over 24 classified instances (``syn20m02m``).
#: A decision point at or below this cuts inside the win distribution.
_SLOWEST_MEASURED_WIN_S = 2.96


def test_the_decision_point_never_cuts_inside_the_win_distribution(flag):
    """The regression witness, and the reason the floor exists.

    A route's win time is ABSOLUTE -- ``syn20m02m`` certifies in ~2.9 s at every
    limit -- while a fraction SHRINKS with the limit. At a 20 s budget the bare
    10% gives 2.0 s and ``syn20m02m`` went optimal 3.16 s -> **feasible 20.12 s**.
    The 61-instance in-repo panel could not see it: it holds no ``syn``/``rsyn``
    instance, which is the class the route exists for.
    """
    flag(True)
    for limit in (20.0, 30.0, 60.0, 600.0):
        g = _RouteProgressGuard(limit)
        assert g.checkpoint > _SLOWEST_MEASURED_WIN_S, (limit, g.checkpoint)


def test_the_floor_binds_on_a_short_budget(flag):
    flag(True)
    g = _RouteProgressGuard(20.0)
    assert g.checkpoint == pytest.approx(_CONVEX_ROUTE_DECISION_POINT_FLOOR_S)
    # still well under the policy it replaces
    assert g.checkpoint < 20.0 * _CONVEX_ROUTE_BUDGET_FRACTION


def test_the_fraction_binds_on_a_long_budget(flag):
    flag(True)
    for limit in (60.0, 600.0):
        g = _RouteProgressGuard(limit)
        assert g.checkpoint == pytest.approx(limit * _CONVEX_ROUTE_DECISION_POINT_FRACTION)


def test_the_policy_never_spends_more_than_the_one_it_replaces(flag):
    """On a budget too short for the floor the cap wins, so the behavior degrades
    to the #1066 half-budget split rather than to something worse."""
    flag(True)
    for limit in (2.0, 4.0, 8.0, 10.0, 20.0, 60.0):
        on = _RouteProgressGuard(limit).checkpoint
        flag(False)
        off = _RouteProgressGuard(limit).checkpoint
        flag(True)
        assert on <= off + 1e-9, (limit, on, off)
