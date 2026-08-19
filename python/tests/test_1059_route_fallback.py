"""#1059: the convex-MINLP auto-route must certify or fall back, never strand.

The §5 graduation panel refused ``DISCOPT_CONVEX_MINLP_ROUTE`` for a specific,
reproducible reason: on three in-repo instances the routed OA path terminated
*without a dual bound*, and on ``alan`` — a 9-variable convex MIQP — it burned
the entire 180 s budget and returned an uncertified 3.000 where the default
spatial path certifies 2.925 in 0.18 s (3/3 reps). A route is a decision, not a
commitment; these tests pin the decision being revisited.

Two mechanisms, tested separately:

* the auto-route is **budgeted** to a fraction of the caller's limit, so there
  is a remainder left to recover with (an explicit ``solver="mip-nlp"`` is a
  caller's choice and keeps the whole budget);
* an auto-routed result that did not certify **falls back** to the default path
  with the remainder, seeded by whatever the route did find, and says so on
  ``SolveResult.algorithm_route``.
"""

import logging
from pathlib import Path

import pytest
from discopt.modeling.core import Model, SolveResult, from_nl
from discopt.solver import (
    _CONVEX_ROUTE_BUDGET_FRACTION,
    _merge_route_and_fallback,
    _route_is_better,
    _route_result_is_certified,
)

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
ROUTE_ENV = "DISCOPT_CONVEX_MINLP_ROUTE"

pytest.importorskip("highspy", reason="the OA master needs a MILP backend")


def _load(name: str) -> Model:
    path = NL_DIR / f"{name}.nl"
    assert path.exists(), f"missing corpus instance {path}"
    m = from_nl(str(path))
    m._convexity_time_budget = 10.0
    return m


GAP_TOL = 1e-4


class _Res:
    """Minimal stand-in with just the fields the predicate reads."""

    def __init__(self, status=None, gap_certified=False, x=None, gap=0.0):
        self.status = status
        self.gap_certified = gap_certified
        self.x = x
        self.gap = gap


@pytest.mark.unit
class TestCertifiedPredicate:
    """``feasible`` with no bound is the failure the fallback exists for."""

    def test_certified_result_is_certified(self):
        assert _route_result_is_certified(_Res("optimal", gap_certified=True), GAP_TOL) is True

    def test_feasible_without_a_certificate_is_not(self):
        assert _route_result_is_certified(_Res("feasible", gap_certified=False), GAP_TOL) is False

    def test_optimal_without_gap_certified_is_not(self):
        """``optimal`` is a claim; ``gap_certified`` is the proof. Trust the proof."""
        assert _route_result_is_certified(_Res("optimal", gap_certified=False), GAP_TOL) is False

    def test_infeasible_is_a_certificate(self):
        assert _route_result_is_certified(_Res("infeasible"), GAP_TOL) is True

    def test_unbounded_is_a_certificate(self):
        assert _route_result_is_certified(_Res("unbounded"), GAP_TOL) is True

    def test_none_is_not(self):
        assert _route_result_is_certified(None, GAP_TOL) is False

    def test_a_valid_but_wide_gap_is_not_finished(self):
        """The #1059 defect: ``gap_certified`` means *valid*, not *closed*.

        Measured on ``syn40m`` at ``time_limit=60`` with the route on: OA came
        back ``status="feasible"``, ``gap_certified=True``, ``gap=0.808``,
        ``bound=290.611`` against ``obj=55.713`` -- and the predicate called it
        a certificate, so the solver returned after 30.17 s and discarded the
        29.8 s it had reserved for the fallback.
        """
        wide = _Res("feasible", gap_certified=True, gap=0.808289)
        assert _route_result_is_certified(wide, GAP_TOL) is False

    def test_a_gap_inside_tolerance_is_finished(self):
        assert (
            _route_result_is_certified(
                _Res("optimal", gap_certified=True, gap=GAP_TOL / 2), GAP_TOL
            )
            is True
        )

    def test_a_missing_or_infinite_gap_is_not_finished(self):
        """No gap at all is the ``alan``/``clay0303hfsg``/``tls2`` failure."""
        assert (
            _route_result_is_certified(_Res("feasible", gap_certified=True, gap=None), GAP_TOL)
            is False
        )
        assert (
            _route_result_is_certified(
                _Res("feasible", gap_certified=True, gap=float("inf")), GAP_TOL
            )
            is False
        )


def _sr(obj=None, bound=None, gap_certified=False, status="feasible", gap=None):
    """A real ``SolveResult``, so the merge is tested against the real dataclass."""
    return SolveResult(
        status=status, objective=obj, bound=bound, gap_certified=gap_certified, gap=gap, x={}
    )


@pytest.mark.unit
class TestBetterInSense:
    def test_maximize_prefers_larger(self):
        assert _route_is_better(2.0, 1.0, True) is True
        assert _route_is_better(1.0, 2.0, True) is False

    def test_minimize_prefers_smaller(self):
        assert _route_is_better(1.0, 2.0, False) is True
        assert _route_is_better(2.0, 1.0, False) is False

    def test_no_incumbent_loses_to_any_finite_value(self):
        assert _route_is_better(1.0, None, False) is True
        assert _route_is_better(None, 1.0, False) is False
        assert _route_is_better(None, None, False) is False

    def test_nonfinite_counts_as_no_incumbent(self):
        assert _route_is_better(float("nan"), 1.0, False) is False
        assert _route_is_better(1.0, float("inf"), True) is True


@pytest.mark.unit
class TestRouteFallbackMerge:
    """The "never worse than the route" contract, made true by construction.

    ``_route_incumbent_seed`` used to carry this contract by warm-starting the
    fallback with the routed point. Measured on ``rsyn0840m`` at a 60 s limit,
    that made things *worse*, not better: the default path alone returns 70.351
    in the same 30 s the fallback had, while the seeded fallback returned
    -11.413 -- the warm start pinned the incumbent and the primal heuristics
    then searched its neighbourhood. Suppressing only the seed recovered 70.351
    exactly. The merge below replaces that mechanism.
    """

    def test_the_better_incumbent_wins_on_maximize(self):
        route, fb = _sr(obj=-11.413), _sr(obj=70.351)
        assert _merge_route_and_fallback(route, fb, True) is fb

    def test_the_route_wins_when_the_fallback_is_worse(self):
        route, fb = _sr(obj=55.713), _sr(obj=33.197)
        assert _merge_route_and_fallback(route, fb, True) is route

    def test_the_better_incumbent_wins_on_minimize(self):
        route, fb = _sr(obj=1.0), _sr(obj=2.0)
        assert _merge_route_and_fallback(route, fb, False) is route

    def test_a_fallback_with_no_incumbent_never_displaces_the_route(self):
        route, fb = _sr(obj=5.0), _sr(obj=None)
        assert _merge_route_and_fallback(route, fb, True) is route

    def test_the_tighter_certified_bound_is_kept_when_the_fallback_wins(self):
        """Measured on ``syn40m``: OA proves 290.61 in 30 s where the fallback's
        own bound after the remaining 30 s is 1206.26 -- 4x looser on a maximize
        model, where the bound is an upper bound and smaller is tighter."""
        route = _sr(obj=55.713, bound=290.611, gap_certified=True)
        fb = _sr(obj=70.0, bound=1206.261)
        merged = _merge_route_and_fallback(route, fb, True)
        assert merged is fb
        assert merged.bound == pytest.approx(290.611)
        assert merged.gap == pytest.approx((290.611 - 70.0) / 70.0)

    def test_a_looser_bound_never_replaces_a_tighter_one(self):
        route = _sr(obj=1.0, bound=1000.0, gap_certified=True)
        fb = _sr(obj=1.0, bound=100.0, gap_certified=True)
        merged = _merge_route_and_fallback(route, fb, True)
        assert merged.bound == pytest.approx(100.0)

    def test_an_uncertified_fallback_bound_is_not_adopted_by_the_route(self):
        """An uncertified bound is not a proof and must not be reported as one."""
        route = _sr(obj=5.0, bound=10.0, gap_certified=True)
        fb = _sr(obj=1.0, bound=6.0, gap_certified=False)
        merged = _merge_route_and_fallback(route, fb, True)
        assert merged is route
        assert merged.bound == pytest.approx(10.0)

    def test_an_uncertified_route_bound_is_not_adopted_by_the_fallback(self):
        route = _sr(obj=1.0, bound=6.0, gap_certified=False)
        fb = _sr(obj=5.0, bound=10.0, gap_certified=True)
        merged = _merge_route_and_fallback(route, fb, True)
        assert merged is fb
        assert merged.bound == pytest.approx(10.0)

    def test_a_closed_certified_fallback_is_never_displaced(self, caplog):
        """A certificate outranks a better number.

        If the fallback proved optimality, no feasible point can beat it; a
        route objective that appears to is an inconsistency, not an answer, and
        swapping the proof out for it would be a silent certificate loss.
        """
        route = _sr(obj=9.0, bound=None, gap_certified=False)
        fb = _sr(obj=5.0, bound=5.0, gap_certified=True, gap=0.0, status="optimal")
        with caplog.at_level(logging.WARNING, logger="discopt.solver"):
            merged = _merge_route_and_fallback(route, fb, True)
        assert merged is fb
        assert merged.gap_certified is True
        assert any("should be impossible" in r.message for r in caplog.records), (
            "the inconsistency must be surfaced, not silently swallowed"
        )

    def test_a_closed_certified_route_still_wins_on_objective(self):
        """The guard is one-directional: two certificates compare normally."""
        route = _sr(obj=9.0, bound=9.0, gap_certified=True, gap=0.0, status="optimal")
        fb = _sr(obj=5.0, bound=5.0, gap_certified=True, gap=0.0, status="optimal")
        assert _merge_route_and_fallback(route, fb, True) is route

    def test_minimize_bound_direction(self):
        """On minimize the bound is a LOWER bound: tighter means larger."""
        route = _sr(obj=10.0, bound=8.0, gap_certified=True)
        fb = _sr(obj=9.0, bound=2.0)
        merged = _merge_route_and_fallback(route, fb, False)
        assert merged is fb
        assert merged.bound == pytest.approx(8.0)

    def test_missing_sides_pass_through(self):
        fb = _sr(obj=1.0)
        assert _merge_route_and_fallback(None, fb, True) is fb
        route = _sr(obj=1.0)
        assert _merge_route_and_fallback(route, None, True) is route


@pytest.mark.unit
class TestRouteBudget:
    """An auto-route is budgeted; an explicit choice is not."""

    @staticmethod
    def _capture(monkeypatch):
        seen = {}
        import discopt.solvers.mip_nlp as mn

        real = mn.solve_mip_nlp

        def spy(model, **kw):
            seen["time_limit"] = kw.get("time_limit")
            return real(model, **kw)

        monkeypatch.setattr(mn, "solve_mip_nlp", spy)
        return seen

    def test_auto_route_gets_a_fraction_of_the_limit(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        seen = self._capture(monkeypatch)
        _load("gbd").solve(time_limit=20.0)
        assert seen, "solve_mip_nlp was never called -- the router did not fire"
        assert seen["time_limit"] == pytest.approx(20.0 * _CONVEX_ROUTE_BUDGET_FRACTION)

    def test_explicit_mip_nlp_keeps_the_whole_limit(self, monkeypatch):
        """The caller chose the algorithm; there is no fallback to reserve for."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        seen = self._capture(monkeypatch)
        _load("gbd").solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=20.0)
        assert seen, "solve_mip_nlp was never called"
        assert seen["time_limit"] == pytest.approx(20.0)


@pytest.mark.slow
class TestFallbackEndToEnd:
    """``alan``: the panel's sharpest regression, and the reason for this change.

    Reference optimum 2.925 (``minlplib.solu``). Before the fallback the routed
    arm returned an uncertified 3.000 after spending the whole budget.
    """

    def test_alan_recovers_its_certificate_through_the_fallback(self, monkeypatch):
        monkeypatch.setenv(ROUTE_ENV, "1")
        result = _load("alan").solve(time_limit=60.0)
        assert result.gap_certified is True
        assert result.objective == pytest.approx(2.925, abs=1e-3)

    def test_the_fallback_is_visible_on_the_result(self, monkeypatch):
        """A fallback solve must not read as one that was never routed."""
        monkeypatch.setenv(ROUTE_ENV, "1")
        result = _load("alan").solve(time_limit=60.0)
        assert result.algorithm_route is not None
        assert "mip-nlp/oa" in result.algorithm_route
        assert "fell back" in result.algorithm_route

    def test_route_off_is_untouched(self, monkeypatch):
        """The default path keeps its own answer and its empty route field."""
        monkeypatch.delenv(ROUTE_ENV, raising=False)
        result = _load("alan").solve(time_limit=60.0)
        assert result.algorithm_route is None
        assert result.gap_certified is True
        assert result.objective == pytest.approx(2.925, abs=1e-3)
