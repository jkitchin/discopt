"""#1066: the auto-route's progress budget, extended to the single-tree method.

The #1059 route hands a convex MINLP to the MIP-NLP family and keeps a reserve of
the caller's time limit for the fallback. #1066 replaced that fixed 50% wall with
:class:`~discopt.solver._RouteProgressGuard`, a ``termination_hook`` that returns
budget the route has not earned -- but only ``solve_oa`` accepted such a hook, so
once the route was retargeted at ``lp_nlp_bb`` the guard declined and the fixed
wall came back.

Both directions of that wall were measured on the reporter panel (default
settings, 60 s, arms back to back, 2026-08-29):

* ``rsyn0820m02m`` certifies at **37.5 s** and the 30 s wall cut it: ``optimal``
  became ``feasible``.
* ``squfl015-060`` is certified by the **fallback**, not the route. Handing the
  route the whole limit turned that row from ``optimal`` into ``feasible``.

So neither a bigger budget nor a smaller one is the answer, and the discriminator
has to be progress -- which means ``lp_nlp_bb`` needs a check-in. The HiGHS master
has exactly one honest one: the restart between trees.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solver import _route_progress_guard_options, _RouteProgressGuard
from discopt.solvers import SolveStatus

highspy = pytest.importorskip("highspy", reason="the HiGHS master is an opt-in extra")

from discopt.solvers.milp_highs import solve_milp_with_lazy_cuts  # noqa: E402

# max x0 + 2 x1 s.t. x0 + x1 <= 3.5, integer, x >= 0 -- the same toy the cut-row
# regression uses, so a separator veto is the only thing driving restarts.
_C = np.array([-1.0, -2.0])
_A_UB = np.array([[1.0, 1.0]])
_B_UB = np.array([3.5])
_INTEGRALITY = np.array([1, 1])
_BOUNDS = [(0.0, None), (0.0, None)]


def _veto_until(threshold: float):
    """Separate ``x1 <= t`` for a shrinking ``t`` -- one restart per call."""

    def _sep(x):
        if x[1] > threshold + 1e-6:
            return [(np.array([0.0, 1.0]), threshold)]
        return None

    return _sep


@pytest.mark.smoke
def test_the_highs_master_consults_a_terminate_callback_at_every_restart():
    """It used to raise ``NotImplementedError`` for any ``terminate_callback``."""
    seen: list[dict] = []

    res = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=_veto_until(1.0),
        terminate_callback=lambda snap: seen.append(dict(snap)) or False,
    )
    assert res.status == SolveStatus.OPTIMAL
    assert seen, "the callback never fired -- there was no check-in to budget by"
    assert len(seen) == res.callback_stats["terminate_polls"], (seen, res.callback_stats)
    assert sum(1 for s in seen if s["context"] == "restart") == res.callback_stats["restarts"]
    for snap in seen:
        assert snap["context"] in ("restart", "interrupt")
        assert snap["elapsed"] >= 0.0
        # The bound is the master's own, which is what a progress judgement reads.
        assert "dual_bound" in snap
    assert res.callback_stats["terminated"] is False
    assert res.callback_stats["terminate_context"] is None


@pytest.mark.smoke
def test_a_callback_that_says_stop_stops_the_loop_and_says_so():
    res = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=_veto_until(1.0),
        terminate_callback=lambda _snap: True,
    )
    assert res.callback_stats["terminated"] is True
    assert res.callback_stats["terminate_context"] == "restart"
    # The tree that just finished was solved to optimality but its incumbent was
    # vetoed, so the cut it needs is the one we are declining to add. Reporting
    # OPTIMAL would certify a master that is missing that row.
    assert res.status != SolveStatus.OPTIMAL
    assert res.callback_stats["restarts"] == 1


@pytest.mark.smoke
def test_stopping_early_still_returns_a_valid_bound_and_an_accepted_point():
    """Cutting the loop short must not turn a bound into a wrong one."""
    stopped = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=_veto_until(1.0),
        terminate_callback=lambda _snap: True,
    )
    full = solve_milp_with_lazy_cuts(
        _C,
        _A_UB,
        _B_UB,
        bounds=_BOUNDS,
        integrality=_INTEGRALITY,
        lazy_callback=_veto_until(1.0),
    )
    assert full.status == SolveStatus.OPTIMAL
    assert full.objective is not None
    # Minimization sense inside the master: a dual bound never exceeds the optimum.
    assert stopped.bound is not None
    assert stopped.bound <= full.objective + 1e-9, (stopped.bound, full.objective)
    if stopped.x is not None:
        # Only ever a point the separator accepted.
        assert stopped.x[1] <= 1.0 + 1e-6


@pytest.mark.smoke
def test_a_raising_callback_surfaces_rather_than_reading_as_keep_going():
    def _boom(_snap):
        raise ZeroDivisionError("instrument broke")

    with pytest.raises(ZeroDivisionError, match="instrument broke"):
        solve_milp_with_lazy_cuts(
            _C,
            _A_UB,
            _B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=_veto_until(1.0),
            terminate_callback=_boom,
        )


def _toy_convex_minlp():
    """A convex MINLP whose master really does restart (7 times, measured).

    Three big-M-linked facilities with a convex quadratic operating cost. The
    shape matters: a model whose first master point the separator *accepts*
    never restarts, so a hook installed on it is never called and every
    assertion about the context it receives is vacuous (CLAUDE.md §6).
    Optimum 9.0 at ``x = (2, 2, 2)``, all three open.
    """
    import discopt.modeling as dm

    m = dm.Model("issue_1066_hook")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(3)]
    ys = [m.binary(f"y{i}") for i in range(3)]
    for xi, yi in zip(xs, ys):
        m.subject_to(xi <= 5.0 * yi)
    m.subject_to(sum(xs) >= 4.0)
    m.minimize(sum((xi - 2.0) ** 2 for xi in xs) + sum(3.0 * yi for yi in ys))
    return m


# --------------------------------------------------------------------------
# The route wiring: the guard now installs on ``lp_nlp_bb`` too.
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_guard_installs_on_the_single_tree_method_it_now_routes_to():
    options, guard = _route_progress_guard_options(None, method_key="lp_nlp_bb", time_limit=60.0)
    assert isinstance(guard, _RouteProgressGuard)
    assert options is not None
    assert options["termination_hook"] is guard
    # OA needs a soft master deadline to get its hook called at all; the single
    # tree checks in at each restart, and there is no separate master to deadline.
    # Passing one would be a kwarg ``solve_lp_nlp_bb`` refuses.
    assert "master_checkin_deadline" not in options


@pytest.mark.smoke
def test_the_guard_still_declines_for_a_method_that_cannot_call_it():
    options, guard = _route_progress_guard_options(None, method_key="ecp", time_limit=60.0)
    assert guard is None
    assert options is None


@pytest.mark.smoke
def test_the_route_never_installs_its_own_hook_over_the_callers():
    mine = object()
    options, guard = _route_progress_guard_options(
        {"termination_hook": mine}, method_key="lp_nlp_bb", time_limit=60.0
    )
    assert guard is None
    assert options is not None and options["termination_hook"] is mine


@pytest.mark.smoke
def test_the_simplex_master_refuses_a_hook_it_could_never_call():
    """A budget built on a hook that cannot fire is a fiction (CLAUDE.md §3)."""
    from discopt.solvers.oa import solve_lp_nlp_bb

    m = _toy_convex_minlp()

    with pytest.raises(NotImplementedError, match="no check-in point"):
        solve_lp_nlp_bb(m, time_limit=5.0, milp_solver="simplex", termination_hook=lambda _c: False)


@pytest.mark.smoke
def test_the_hook_sees_the_bounds_a_progress_judgement_needs():
    """And the call count is reported, so 'never fired' cannot read as 'said go'."""
    from discopt.solvers.oa import solve_lp_nlp_bb

    m = _toy_convex_minlp()

    seen: list[dict] = []
    res = solve_lp_nlp_bb(
        m,
        time_limit=30.0,
        milp_solver="highs",
        termination_hook=lambda ctx: seen.append(dict(ctx)) or False,
    )
    stats = (res.mip_nlp_trace or {})["summary"]["callback_stats"]
    assert stats["termination_hook_calls"] == len(seen), (stats, len(seen))
    assert stats["termination_hook_calls"] > 0, (
        "the hook never got a check-in on this model, so this test asserts nothing "
        "about the context it is supposed to receive (CLAUDE.md §6)"
    )
    for ctx in seen:
        assert ctx["event"] == "termination"
        assert ctx["elapsed"] >= 0.0
        assert ctx["is_minimization"] is True
        assert "current_dual_bound" in ctx
        assert "current_primal_bound" in ctx
        assert "relative_gap" in ctx
        assert ctx["restarts"] >= 1
    assert res.status == "optimal", res.status
    assert res.objective == pytest.approx(9.0, abs=1e-4)


@pytest.mark.smoke
def test_a_hook_that_says_stop_ends_the_solve_and_is_named_as_the_reason():
    """The clock and the hook share a callback; the reason must not conflate them."""
    from discopt.solvers.oa import solve_lp_nlp_bb

    res = solve_lp_nlp_bb(
        _toy_convex_minlp(),
        time_limit=300.0,  # nowhere near reached, so "time_limit" would be a lie
        milp_solver="highs",
        termination_hook=lambda _ctx: True,
    )
    trace = res.mip_nlp_trace or {}
    assert trace["termination_reason"] == "termination_hook"
    assert trace["summary"]["callback_stats"]["terminated"] is True
    assert res.status != "optimal"
    # A stopped route still owes the caller a sound bound.
    if res.bound is not None and res.objective is not None:
        assert res.bound <= res.objective + 1e-6


def _tree_that_lasts():
    """A MILP whose *tree* really runs, plus a separator that restarts it.

    The two-variable toy above never enters the branch-and-bound loop -- HiGHS
    finishes it in presolve -- so it exercises the restart check-in and nothing
    else. This one (45 binaries, seeded, ~0.15 s) produces both kinds of check-in
    in the same solve: one restart and, unthrottled, a few hundred in-tree polls.
    """
    rng = np.random.default_rng(7)
    n = 45
    c = -rng.integers(5, 60, size=n).astype(float)
    a_ub = rng.integers(1, 40, size=(6, n)).astype(float)
    b_ub = a_ub.sum(axis=1) * 0.32
    state = {"k": 0}

    def _sep(x):
        """Veto any incumbent that opens too many items, three times over."""
        if state["k"] < 3 and x.sum() > 12 - state["k"]:
            state["k"] += 1
            return [(np.ones(n), 12.0 - state["k"])]
        return None

    return {
        "c": c,
        "A_ub": a_ub,
        "b_ub": b_ub,
        "bounds": [(0.0, 1.0)] * n,
        "integrality": np.ones(n, dtype=int),
        "lazy_callback": _sep,
    }


@pytest.mark.smoke
@pytest.mark.parametrize("fixture", ["presolved", "tree"])
def test_installing_a_hook_must_not_change_the_answer(fixture):
    """The sticky-flag regression, and the reason the poll writes it on every path.

    HiGHS hands every callback the same ``HighsCallbackInput``. The separator sets
    ``user_interrupt`` to request a restart; if the interrupt poll then returns
    without restating the flag, the *rebuilt* tree is interrupted before it has
    done anything. Measured on the two-variable model: 7 restarts / optimal 9.0
    with no hook became 1 restart / feasible 15.0 with one. A hook that never says
    stop has to be invisible.
    """
    if fixture == "presolved":
        kw = dict(
            c=_C,
            A_ub=_A_UB,
            b_ub=_B_UB,
            bounds=_BOUNDS,
            integrality=_INTEGRALITY,
            lazy_callback=_veto_until(1.0),
        )
        hooked_kw = dict(kw, lazy_callback=_veto_until(1.0))
    else:
        kw = _tree_that_lasts()
        hooked_kw = _tree_that_lasts()  # the separator is stateful; give each a fresh one

    plain = solve_milp_with_lazy_cuts(**kw)
    hooked = solve_milp_with_lazy_cuts(
        **hooked_kw,
        terminate_callback=lambda _snap: False,
        terminate_poll_s=0.0,  # poll as hard as possible: the worst case for stickiness
    )
    assert hooked.status == plain.status
    assert hooked.objective == pytest.approx(plain.objective)
    assert hooked.callback_stats["restarts"] == plain.callback_stats["restarts"]
    assert hooked.callback_stats["lazy_cuts"] == plain.callback_stats["lazy_cuts"]


@pytest.mark.smoke
def test_the_in_tree_poll_fires_and_the_interval_is_what_bounds_it():
    """Restarts alone are not a clock; the in-tree poll is what makes one."""
    unthrottled = solve_milp_with_lazy_cuts(
        **_tree_that_lasts(),
        terminate_callback=lambda _snap: False,
        terminate_poll_s=0.0,
    )
    throttled = solve_milp_with_lazy_cuts(
        **_tree_that_lasts(),
        terminate_callback=lambda _snap: False,
        terminate_poll_s=3600.0,  # longer than this solve, so only restarts check in
    )
    assert throttled.callback_stats["terminate_polls"] == throttled.callback_stats["restarts"]
    assert (
        unthrottled.callback_stats["terminate_polls"] > throttled.callback_stats["terminate_polls"]
    ), "the interval did not bound anything -- the in-tree poll never fired"


@pytest.mark.smoke
def test_a_stop_from_inside_the_tree_is_honoured_and_attributed():
    calls = []

    def _stop_on_the_first_in_tree_look(snap):
        calls.append(snap["context"])
        return snap["context"] == "interrupt"

    res = solve_milp_with_lazy_cuts(
        **_tree_that_lasts(),
        terminate_callback=_stop_on_the_first_in_tree_look,
        terminate_poll_s=0.0,
    )
    assert "interrupt" in calls, "the in-tree poll never fired, so this asserts nothing"
    assert res.callback_stats["terminated"] is True
    assert res.callback_stats["terminate_context"] == "interrupt"
    assert res.status != SolveStatus.OPTIMAL


@pytest.mark.smoke
def test_a_negative_poll_interval_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        solve_milp_with_lazy_cuts(
            **_tree_that_lasts(),
            terminate_callback=lambda _snap: False,
            terminate_poll_s=-1.0,
        )


def _separator_outlives_the_certificate():
    """A convex MINLP whose separator keeps cutting long after the gap is closed.

    Eight symmetric on/off units, four of which must open: the master enumerates
    equivalent integer assignments and the separator vetoes each one, so the
    restart loop keeps going (75 restarts, measured) well past the instant the
    master's dual bound meets the NLP incumbent. Optimum 0.008 with four units at
    ``x = 0.5``... the point is not the number, it is that the certificate becomes
    available early and the separator does not stop.
    """
    import discopt.modeling as dm

    m = dm.Model("issue_1066_certificate")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=3.0) for i in range(8)]
    ys = [m.binary(f"y{i}") for i in range(8)]
    for xi, yi in zip(xs, ys):
        m.subject_to(xi <= 3.0 * yi)
    m.subject_to(sum(xs) >= 4.0)
    m.minimize(sum((xi - 0.5) ** 2 for xi in xs) + 0.001 * sum(ys))
    return m


@pytest.mark.smoke
def test_the_single_tree_stops_when_its_own_bound_certifies_the_incumbent():
    """It used to run on until the separator ran dry, the clock ran out, or a hook fired.

    ``rsyn0820m02m`` is the reporter row this cost: bound 1092.1600 against
    incumbent 1092.0911 -- gap 6.3e-5, inside the 1e-4 default -- at 5.1 s of a
    60 s limit, then five more trees and ``feasible`` at the wall.
    """
    res = _separator_outlives_the_certificate().solve(time_limit=60)
    stats = res.mip_nlp_trace["summary"]["callback_stats"]

    assert stats["dual_bound_observations"] > 0, (
        "no check-in carried a dual bound, so the early exit was never asked "
        "anything -- this test would pass on a driver that cannot certify at all"
    )
    assert stats["converged_early"] is True
    assert stats["restarts"] > 1, "the separator ran dry on its own; nothing was cut short"
    assert res.mip_nlp_trace["termination_reason"] == "gap_tolerance"
    assert str(res.status) in ("SolveStatus.OPTIMAL", "optimal")
    # The certificate has to hold on the reported numbers, not just internally.
    assert res.bound <= res.objective + 1e-6 + 1e-4 * abs(res.objective)


@pytest.mark.smoke
def test_the_early_exit_stays_out_of_the_way_when_the_separator_finishes_first():
    """Not every model needs it, and on those it must change nothing."""
    res = _toy_convex_minlp().solve(time_limit=30)
    stats = res.mip_nlp_trace["summary"]["callback_stats"]
    assert stats["dual_bound_observations"] > 0
    assert stats["converged_early"] is False
    assert stats["terminated"] is False
    assert res.objective == pytest.approx(9.0, abs=1e-6)
