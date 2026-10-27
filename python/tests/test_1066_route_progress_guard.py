"""#1066: the convex-MINLP auto-route is budgeted by *progress*, not by a clock.

#1059 cut the auto-route off at a fixed half of the caller's limit so there was
always a remainder to fall back with. That wall is blind in both directions, and
the #1066 reporter panel (15 convex MINLPLib instances, default settings, 60 s;
600 s on two rows) measured both failures:

* it cuts off a route that is converging -- ``syn40m``'s relative gap falls
  0.770 -> 0.391 -> 0.147 and OA certifies the oracle optimum 67.713256 at
  43.5 s, but the wall lands at 30 s, so the panel reported ``feasible``
  58.210 and neither path certified;
* it lets a dead route hold the budget to the wall -- ``portfol_classical050_1``
  sits at relative gap 0.0033823 from 49.4 s to 59.9 s, identical bounds, and
  ``alan`` holds a gap of 1.0 flat for all 51 of its iterations.

Every trace in these tests is real, copied from the measured run
(``scratchpad/issue1066/charac.json``); the two extrapolated points are labelled
where they appear. Replaying the guard over all 15 recorded traces: certified
7 -> **8**, total wall 1388.2 s -> **1376.4 s**, no row left without a dual
bound, nothing regressed.
"""

from pathlib import Path

import pytest
from discopt.modeling.core import Model, from_nl
from discopt.solver import (
    _CONVEX_ROUTE_BOUND_STALL_FRACTION,
    _CONVEX_ROUTE_BUDGET_FRACTION,
    _CONVEX_ROUTE_MIN_GAP_IMPROVEMENT,
    _convex_route_progress_guard_enabled,
    _route_progress_guard_options,
    _RouteProgressGuard,
)
from discopt.solvers.oa import _master_time_budget, solve_oa

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
ROUTE_ENV = "DISCOPT_CONVEX_MINLP_ROUTE"
GUARD_ENV = "DISCOPT_CONVEX_ROUTE_GUARD"

pytest.importorskip("highspy", reason="the OA master needs a MILP backend")

#: #1143 graduated an earlier *decision point* to default-on, which replaces the
#: half-budget checkpoint and the gap-trend verdict this module is about. That is
#: not a conflict: §5 requires the superseded policy to be KEPT INTACT AND TESTED
#: behind its opt-out, and this module is that test. Pin it explicitly for the
#: whole module rather than letting these assertions read whichever policy happens
#: to be the default -- a test of the legacy path must select the legacy path.
DECISION_POINT_ENV = "DISCOPT_CONVEX_ROUTE_DECISION_POINT"


@pytest.fixture(autouse=True)
def _pin_the_1066_policy(monkeypatch):
    monkeypatch.setenv(DECISION_POINT_ENV, "0")


def _load(name: str) -> Model:
    path = NL_DIR / f"{name}.nl"
    assert path.exists(), f"missing corpus instance {path}"
    m = from_nl(str(path))
    m._convexity_time_budget = 10.0
    return m


def _ctx(elapsed, gap, *, bounded=True):
    """One OA ``termination_hook`` context, shaped like the real one.

    ``bounded=False`` is the OA driver's pre-dual-bound state: it reports
    ``relative_gap=1.0`` with ``current_dual_bound=None``, a placeholder rather
    than a measurement, and the guard has to tell the two apart.

    The bounds are consistent with ``gap``: the incumbent is pinned at 1.0 and
    the dual bound carries the whole gap, as on an OA route whose incumbent is
    already optimal. The #2013 bound-stall rule reads the bounds, so a fixture
    whose bound never moves while its gap shrinks would be a different trace.
    """
    return {
        "elapsed": float(elapsed),
        "relative_gap": gap,
        "current_dual_bound": (1.0 - (gap or 0.0)) if bounded else None,
        "current_primal_bound": 1.0 if bounded else None,
    }


def _geometric(*, window_ratio, limit=60.0, step_s=1.0):
    """A trace whose gap shrinks by ``window_ratio`` across each trailing window.

    One point every ``step_s`` seconds out to ``limit``, decaying geometrically,
    so the ratio between the newest gap and the oldest one still inside the
    window is exactly ``window_ratio`` -- which is the quantity the guard tests.
    """
    window = limit * _CONVEX_ROUTE_BUDGET_FRACTION * 0.5
    per_step = window_ratio ** (step_s / window)
    points, gap, elapsed = [], 0.5, step_s
    while elapsed <= limit:
        points.append((elapsed, gap))
        gap *= per_step
        elapsed += step_s
    return points


def _replay(guard, points):
    """Feed ``(elapsed, gap)`` points until the guard fires. Returns the time."""
    for elapsed, gap in points:
        if guard(_ctx(elapsed, gap)):
            return elapsed
    return None


# The measured OA gap traces, ``(elapsed, relative_gap)``. ``None`` gaps are the
# iterations that had no finite dual bound yet.
SYN40M = [
    (0.06, None),
    (4.06, 0.7703621580824017),
    (10.77, 0.39143864188149563),
    (24.6, 0.14672035380698015),
]
RSYN0840M = [(0.14, None), (54.82, 1.014735195337869)]
PORTFOL_TAIL = [
    (30.5, 0.0033823455725941615),
    (35.4, 0.0033823455725941615),
    (44.1, 0.0033823455725941615),
    (49.36, 0.0033823455725941615),
    (54.41, 0.0033823455725941615),
    (59.87, 0.0033823455725941615),
]


@pytest.mark.unit
class TestGuardVerdicts:
    """What the guard does with the traces that motivated it."""

    def test_it_never_fires_before_the_checkpoint(self):
        """Behaviour up to the old wall must be exactly the old wall's.

        ``alan`` is the case: a gap pinned at 1.0 from the first iteration. The
        guard has every reason to stop it and must not, because stopping before
        the checkpoint would take budget the fixed split had already granted.
        """
        guard = _RouteProgressGuard(60.0)
        flat = [(t / 10.0, 1.0) for t in range(1, 300)]  # 0.1s .. 29.9s
        assert _replay(guard, flat) is None
        assert guard.call_count == len(flat)
        assert guard.fired_at is None

    def test_a_converging_route_keeps_the_budget_past_the_old_wall(self):
        """``syn40m``: the certificate the 30 s wall threw away.

        The first four points are measured; OA's next iteration certified at
        43.5 s. The 35 s point continues that trend (0.147 -> 0.05) and stands
        in for it -- the guard must let the route reach it.
        """
        guard = _RouteProgressGuard(60.0)
        assert _replay(guard, [*SYN40M, (35.0, 0.05)]) is None
        assert guard.fired_at is None

    def test_a_stalled_route_is_stopped_at_the_checkpoint(self):
        """``portfol_classical050_1``: identical bounds, 154 cuts an iteration."""
        guard = _RouteProgressGuard(60.0)
        warmup = [(t / 2.0, 0.0033823455725941615) for t in range(1, 60)]
        assert _replay(guard, [*warmup, *PORTFOL_TAIL]) == pytest.approx(30.5)
        assert "improvement" in guard.reason

    def test_one_observation_is_not_a_trend(self):
        """``rsyn0840m``: the OA loop manages **two** iterations in 60 s.

        One finite gap, no trend. Reading that as progress would hand the whole
        budget to a loop that cannot iterate and leave the fallback nothing --
        on this instance that is the difference between reporting an incumbent
        of 151.97 and reporting one of -11.41.
        """
        guard = _RouteProgressGuard(60.0)
        assert _replay(guard, RSYN0840M) == pytest.approx(54.82)
        assert "no trend to judge" in guard.reason

    def test_improvement_below_the_threshold_does_not_earn_the_budget(self):
        """The threshold is per *window*, not per iteration.

        A trace that shrinks its gap by 10% across the trailing window has not
        met the 25% the route must show, however many iterations it took.
        """
        assert _replay(_RouteProgressGuard(60.0), _geometric(window_ratio=0.90)) == pytest.approx(
            30.0
        )

    def test_improvement_above_the_threshold_does_earn_it(self):
        assert _replay(_RouteProgressGuard(60.0), _geometric(window_ratio=0.50)) is None

    def test_the_threshold_is_the_documented_constant(self):
        """Straddle it: just inside keeps the budget, just outside loses it."""
        margin = 0.02
        keeps = 1.0 - _CONVEX_ROUTE_MIN_GAP_IMPROVEMENT - margin
        loses = 1.0 - _CONVEX_ROUTE_MIN_GAP_IMPROVEMENT + margin
        assert _replay(_RouteProgressGuard(60.0), _geometric(window_ratio=keeps)) is None
        assert _replay(_RouteProgressGuard(60.0), _geometric(window_ratio=loses)) == pytest.approx(
            30.0
        )

    def test_progress_older_than_the_window_does_not_count(self):
        """Improvement has to be recent. A route that converged early and then
        stopped is a stalled route, not a converging one."""
        early = [(1.0, 1.0), (2.0, 0.1)]
        late = [(t * 1.0, 0.1) for t in range(3, 40)]
        guard = _RouteProgressGuard(60.0)
        assert _replay(guard, [*early, *late]) == pytest.approx(30.0)
        assert "trailing" in guard.reason

    def test_a_closed_gap_is_left_to_the_loops_own_test(self):
        guard = _RouteProgressGuard(60.0)
        assert _replay(guard, [(t * 1.0, 0.0) for t in range(1, 40)]) is None

    def test_a_placeholder_gap_with_no_dual_bound_is_not_an_observation(self):
        """``relative_gap=1.0`` with ``current_dual_bound=None`` is the OA
        driver's placeholder. Counting it would fabricate a trend out of two
        iterations that never produced a bound."""
        guard = _RouteProgressGuard(60.0)
        assert guard(_ctx(1.0, 1.0, bounded=False)) is False
        assert guard(_ctx(20.0, 1.0, bounded=False)) is False
        assert guard.history == []
        assert guard(_ctx(31.0, 1.0, bounded=False)) is True
        assert "0 finite dual-bound observation(s)" in guard.reason

    def test_the_checkpoint_matches_the_split_it_replaces(self):
        assert _RouteProgressGuard(60.0).checkpoint == pytest.approx(
            60.0 * _CONVEX_ROUTE_BUDGET_FRACTION
        )

    def test_a_tiny_limit_keeps_the_fallback_floor(self):
        """Below the floor the checkpoint is the whole limit, exactly as the
        fixed split behaved -- there is no useful fallback to reserve for."""
        assert _RouteProgressGuard(0.4).checkpoint == pytest.approx(0.4)

    def test_it_returns_real_bools(self):
        """``_validate_external_termination`` rejects anything else."""
        guard = _RouteProgressGuard(60.0)
        assert guard(_ctx(1.0, 0.5)) is False
        assert guard(_ctx(40.0, 0.5)) is True


#: ``cvxnonsep_pcon30`` under ``DISCOPT_HIGHS_MASTER_LP_CUTS=1`` at 30 s (#2013),
#: measured ``(elapsed, dual_bound, primal_bound)`` at the master restarts: the
#: dual bound never leaves -36.156283 while incumbents close the gap 0.275 ->
#: 0.011. The #1066 gap test passes at every check and the guard never fires; the
#: solve ended ``feasible``. With the rule the guard fires at the first restart
#: past the 15 s checkpoint and the fallback certifies -35.986842 at 24.2 s. (The
#: trace keeps every 10th check-in; times are rounded to 0.01 s.)
PCON30_LP_CUTS = [
    (0.11, -36.15628293609323, -26.205452783208024),
    (3.69, -36.15628290334783, -30.309674567899048),
    (6.31, -36.15628290334783, -31.202434348179253),
    (7.52, -36.15628290334783, -32.51191469133501),
    (9.41, -36.15628290334783, -33.92346217335541),
    (11.05, -36.15628290334783, -34.95204068797755),
    (13.31, -36.15628290334783, -34.95204068797755),
    (14.99, -36.15628290334783, -35.221484270984604),
    (16.18, -36.15628290334783, -35.221484270984604),
    (19.16, -36.15628290334783, -35.42008379981217),
    (21.97, -36.15628290334783, -35.679637809096604),
    (25.06, -36.15628290334783, -35.745393635266645),
]


def _bound_ctx(elapsed, dual, primal):
    """A minimisation context with real bounds; the gap is OA's relative gap."""
    return {
        "elapsed": float(elapsed),
        "relative_gap": abs(primal - dual) / max(abs(primal), 1e-10),
        "current_dual_bound": float(dual),
        "current_primal_bound": float(primal),
    }


def _replay_bounds(guard, points):
    for point in points:
        if guard(_bound_ctx(*point)):
            return point[0]
    return None


@pytest.mark.unit
class TestBoundStall:
    """#2013: incumbents alone cannot earn the route its budget."""

    STALL_ENV = "DISCOPT_CONVEX_ROUTE_BOUND_STALL"

    def test_pcon30_is_handed_over_at_the_checkpoint(self, monkeypatch):
        monkeypatch.setenv(self.STALL_ENV, "1")
        guard = _RouteProgressGuard(30.0)
        assert _replay_bounds(guard, PCON30_LP_CUTS) == pytest.approx(16.18)
        assert "only incumbents are closing the gap" in guard.reason

    def test_without_the_rule_the_gap_test_keeps_pcon30_running(self, monkeypatch):
        """The failure the rule exists for: this is the default before #2013."""
        monkeypatch.setenv(self.STALL_ENV, "0")
        guard = _RouteProgressGuard(30.0)
        assert _replay_bounds(guard, PCON30_LP_CUTS) is None

    def test_a_moving_bound_keeps_the_budget(self, monkeypatch):
        """The same incumbents, with a dual bound that closes its share of the gap."""
        monkeypatch.setenv(self.STALL_ENV, "1")
        moving = [(t, dual + 0.01 * t, primal) for t, dual, primal in PCON30_LP_CUTS]
        guard = _RouteProgressGuard(30.0)
        assert _replay_bounds(guard, moving) is None

    def test_movement_is_judged_against_the_gap(self, monkeypatch):
        """Straddle the documented fraction of the absolute gap."""
        monkeypatch.setenv(self.STALL_ENV, "1")
        window = _RouteProgressGuard(30.0).window

        def trace(rate):
            # The incumbent halves its distance to 10 across each window, so the
            # relative gap passes the #1066 test; the dual bound rises at ``rate``.
            return [
                (float(t), 10.0 + rate * t, 10.0 + 2.0 * 0.5 ** (t / window)) for t in range(31)
            ]

        def window_move_over_threshold(rate):
            # What the guard computes at the 15 s checkpoint, re-derived here.
            pts = trace(rate)
            now = 15.0
            duals = [d for t, d, _ in pts if now - window <= t <= now]
            _, d_now, p_now = pts[int(now)]
            threshold = _CONVEX_ROUTE_BOUND_STALL_FRACTION * (p_now - d_now)
            return (max(duals) - min(duals)) / threshold

        unit = 1e-4
        scale = 1.0 / window_move_over_threshold(unit)  # rate*scale*unit sits on the line
        over, under = 1.1 * scale * unit, 0.9 * scale * unit
        assert window_move_over_threshold(over) > 1.0 > window_move_over_threshold(under)
        assert _replay_bounds(_RouteProgressGuard(30.0), trace(over)) is None
        assert _replay_bounds(_RouteProgressGuard(30.0), trace(under)) == pytest.approx(15.0)

    def test_it_never_acts_before_the_checkpoint(self, monkeypatch):
        monkeypatch.setenv(self.STALL_ENV, "1")
        guard = _RouteProgressGuard(30.0)
        early = [p for p in PCON30_LP_CUTS if p[0] < guard.checkpoint]
        assert _replay_bounds(guard, early) is None
        assert guard.call_count == len(early)

    def test_the_decision_point_policy_is_untouched(self, monkeypatch):
        """#1143 consults no trend, so the rule has nothing to act on there."""
        monkeypatch.setenv(self.STALL_ENV, "1")
        monkeypatch.setenv(DECISION_POINT_ENV, "1")
        guard = _RouteProgressGuard(30.0)
        assert guard.decision_point
        assert _replay_bounds(guard, PCON30_LP_CUTS) == pytest.approx(5.0, abs=1.4)
        assert "decision point" in guard.reason


@pytest.mark.unit
class TestGuardInstallation:
    def test_it_installs_on_an_oa_route(self, monkeypatch):
        monkeypatch.setenv(GUARD_ENV, "1")
        options, guard = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
        assert isinstance(guard, _RouteProgressGuard)
        assert options["termination_hook"] is guard

    def test_the_flag_off_restores_the_fixed_split(self, monkeypatch):
        monkeypatch.setenv(GUARD_ENV, "0")
        options, guard = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
        assert guard is None
        assert options is None

    def test_a_method_without_the_hook_declines(self, monkeypatch):
        monkeypatch.setenv(GUARD_ENV, "1")
        _, guard = _route_progress_guard_options(None, method_key="shot", time_limit=60.0)
        assert guard is None

    def test_a_callers_own_termination_hook_wins(self, monkeypatch):
        """Two hooks racing to stop the same loop is not a contract worth
        having, and the caller asked first."""
        monkeypatch.setenv(GUARD_ENV, "1")
        mine = {"termination_hook": lambda ctx: False}
        options, guard = _route_progress_guard_options(mine, method_key="oa", time_limit=60.0)
        assert guard is None
        assert options["termination_hook"] is mine["termination_hook"]

    def test_the_callers_mapping_is_not_mutated(self, monkeypatch):
        monkeypatch.setenv(GUARD_ENV, "1")
        mine = {"milp_solver": "highs"}
        options, guard = _route_progress_guard_options(mine, method_key="oa", time_limit=60.0)
        assert guard is not None
        assert mine == {"milp_solver": "highs"}
        assert options["milp_solver"] == "highs"


@pytest.mark.unit
class TestRouteBudgetWiring:
    """What ``solve_mip_nlp`` actually receives."""

    @staticmethod
    def _capture(monkeypatch):
        seen = {}
        import discopt.solvers.mip_nlp as mn

        real = mn.solve_mip_nlp

        def spy(model, **kw):
            seen["time_limit"] = kw.get("time_limit")
            seen["options"] = kw.get("mip_nlp_options")
            return real(model, **kw)

        monkeypatch.setattr(mn, "solve_mip_nlp", spy)
        return seen

    def test_the_guarded_route_gets_the_whole_limit_and_the_hook(
        self, monkeypatch, forwarded_budget
    ):
        """This is the #1066 change: no wall, a guard instead."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv(GUARD_ENV, "1")
        seen = self._capture(monkeypatch)
        _load("gbd").solve(time_limit=20.0)
        assert seen, "solve_mip_nlp was never called -- the router did not fire"
        assert seen["time_limit"] == forwarded_budget(20.0)
        hook = seen["options"]["termination_hook"]
        assert isinstance(hook, _RouteProgressGuard)
        assert hook.checkpoint == forwarded_budget(20.0, _CONVEX_ROUTE_BUDGET_FRACTION)

    def test_the_flag_off_reproduces_the_1059_split_exactly(self, monkeypatch, forwarded_budget):
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv(GUARD_ENV, "0")
        seen = self._capture(monkeypatch)
        _load("gbd").solve(time_limit=20.0)
        assert seen["time_limit"] == forwarded_budget(20.0, _CONVEX_ROUTE_BUDGET_FRACTION)
        assert seen["options"] is None

    def test_an_explicit_choice_is_never_guarded(self, monkeypatch):
        """An explicit ``solver='mip-nlp'`` has no fallback to reserve for, so
        there is nothing for the guard to hand the budget back to."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        monkeypatch.setenv(GUARD_ENV, "1")
        seen = self._capture(monkeypatch)
        _load("gbd").solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=20.0)
        from discopt.modeling.core import last_solve_prework_seconds

        # The whole limit less only ``Model.solve``'s own pre-work.
        assert seen["time_limit"] == pytest.approx(20.0 - last_solve_prework_seconds())
        assert last_solve_prework_seconds() < 1.0
        assert seen["options"] is None or "termination_hook" not in seen["options"]


@pytest.mark.slow
class TestGuardEndToEnd:
    def test_alan_still_recovers_its_certificate_through_the_fallback(self, monkeypatch):
        """The #1059 regression the whole budget split exists to prevent.

        ``alan``'s OA gap is 1.0 flat, so the guard must take the budget off it
        at the checkpoint and leave the fallback enough to certify 2.925.
        """
        monkeypatch.setenv(ROUTE_ENV, "1")
        monkeypatch.setenv(GUARD_ENV, "1")
        result = _load("alan").solve(time_limit=60.0)
        assert result.gap_certified is True
        assert result.objective == pytest.approx(2.925, abs=1e-3)


class TestMasterCheckinDeadline:
    """The guard can only budget by progress if OA gets back to the top of the loop.

    OA's first master expands to fill whatever budget it is given
    (``_MASTER_NO_INCUMBENT_BUDGET_FRAC``), so widening the route from half the
    limit to all of it also widened the master. Measured on ``rsyn0840m`` at 60 s:
    the guard's second call landed at 55.1 s, so abandoning there left the spatial
    fallback ~5 s and the row's incumbent collapsed from 151.97 to -11.41 on a
    maximization model. ``master_checkin_deadline`` is how the route tells OA when
    it intends to look.
    """

    def test_the_deadline_binds_only_when_it_is_tighter(self):
        budget = _master_time_budget
        # No incumbent: the pre-existing fixed-NLP reserve, untouched.
        assert budget(60.0, has_incumbent=False) == pytest.approx(54.0)
        assert budget(60.0, has_incumbent=True) == pytest.approx(60.0)
        # A deadline inside that budget wins, with or without an incumbent.
        assert budget(60.0, has_incumbent=False, checkin_remaining=30.0) == pytest.approx(30.0)
        assert budget(60.0, has_incumbent=True, checkin_remaining=30.0) == pytest.approx(30.0)
        # A deadline outside it changes nothing.
        assert budget(60.0, has_incumbent=False, checkin_remaining=100.0) == pytest.approx(54.0)

    def test_a_deadline_already_past_does_not_starve_the_master(self):
        """After the checkpoint the clamp must lift, not hand out a negative budget.

        The route sets one absolute deadline for the whole solve, so every
        iteration after the checkpoint sees ``checkin_remaining <= 0``. Treating
        that as a limit would make every subsequent master unsolvable -- which is
        the opposite of the intent, since past the checkpoint the guard has
        already decided to let the route keep running.
        """
        for past in (0.0, -1.0, -45.0):
            assert _master_time_budget(
                30.0, has_incumbent=True, checkin_remaining=past
            ) == pytest.approx(30.0)
            assert _master_time_budget(
                30.0, has_incumbent=False, checkin_remaining=past
            ) == pytest.approx(27.0)

    def test_the_route_installs_the_deadline_at_the_checkpoint(self, monkeypatch):
        monkeypatch.setenv(GUARD_ENV, "1")
        options, guard = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
        assert guard is not None
        assert options["master_checkin_deadline"] == pytest.approx(guard.checkpoint)
        # The checkpoint is where the old fixed split cut, so the reserve the
        # fallback used to get is exactly the reserve it still gets.
        assert guard.checkpoint == pytest.approx(60.0 * _CONVEX_ROUTE_BUDGET_FRACTION)

    def test_declining_the_guard_installs_no_deadline(self, monkeypatch):
        """The opt-out must not leave a lone master cap behind.

        A ``master_checkin_deadline`` with no ``termination_hook`` to read it is
        a master truncation for nobody -- exactly the falsified intervention in
        ``docs/dev/performance-plan.md`` §22.2.
        """
        monkeypatch.setenv(GUARD_ENV, "0")
        options, guard = _route_progress_guard_options(
            {"add_slack": True}, method_key="oa", time_limit=60.0
        )
        assert guard is None
        assert "master_checkin_deadline" not in (options or {})

    def test_oa_refuses_a_deadline_with_no_reader(self):
        model = _load("alan")
        with pytest.raises(ValueError, match="only has an effect alongside termination_hook"):
            solve_oa(model, time_limit=5.0, master_checkin_deadline=2.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_oa_refuses_a_nonsense_deadline(self, bad):
        model = _load("alan")
        with pytest.raises(ValueError, match="finite positive number of seconds"):
            solve_oa(
                model,
                time_limit=5.0,
                master_checkin_deadline=bad,
                termination_hook=lambda ctx: False,
            )

    def test_mip_nlp_accepts_the_option(self):
        """It must survive the option whitelist, or the route's install is rejected."""
        from discopt.solvers.mip_nlp import _OA_OPTION_KEYS

        assert "master_checkin_deadline" in _OA_OPTION_KEYS


class TestGraduatedDefault:
    """The guard is default-on as of the 2026-08-22 panel; the opt-out still works."""

    def test_the_guard_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv(GUARD_ENV, raising=False)
        assert _convex_route_progress_guard_enabled() is True
        options, guard = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
        assert guard is not None
        assert options["termination_hook"] is guard

    @pytest.mark.parametrize("off", ["0"])
    def test_the_opt_out_restores_the_fixed_split(self, monkeypatch, off):
        """``=0`` must keep working, and must not leave a stray master cap behind."""
        monkeypatch.setenv(GUARD_ENV, off)
        assert _convex_route_progress_guard_enabled() is False
        options, guard = _route_progress_guard_options(None, method_key="oa", time_limit=60.0)
        assert guard is None
        assert options is None
