"""#1059: OA must not certify a bound it has already crossed, nor spin on a stalled cut loop.

Both defects were found by the #1059 graduation panel, which routes
convexity-certified MINLPs through the MIP-NLP/OA family and so put OA on the
default path for models that had never reached it.
"""

import logging

import discopt.solvers.oa as oa
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.solver import _bound_crosses_objective, _merge_route_and_fallback


def _sr(obj=None, bound=None, gap_certified=False, status="feasible", gap=None):
    return SolveResult(
        status=status, objective=obj, bound=bound, gap_certified=gap_certified, gap=gap, x={}
    )


@pytest.mark.unit
class TestBoundCrossesObjective:
    """The shared predicate behind the merge guard."""

    def test_the_fac2_crossing_is_caught(self):
        """Measured on ``fac2`` (MINLPLib, optimum 331837498.2).

        OA returned ``bound=331845337.4396878`` with ``gap_certified=True``
        against the merged incumbent ``331837498.1769339`` -- a lower bound
        7839 above the true optimum and 176 above its own incumbent.
        """
        assert _bound_crosses_objective(331845337.4396878, 331837498.1769339, False) is True

    def test_a_valid_minimize_bound_is_not_a_crossing(self):
        assert _bound_crosses_objective(331837497.5456339, 331837498.1769339, False) is False

    def test_rounding_at_scale_is_not_a_crossing(self):
        """1 ulp at |obj| ~ 3e8 must not read as a broken certificate."""
        obj = 331837498.1769339
        assert _bound_crosses_objective(obj + 1e-6, obj, False) is False

    def test_maximize_direction_is_inverted(self):
        # A maximize bound is an UPPER bound: below the incumbent is the crossing.
        assert _bound_crosses_objective(5.0, 9.0, True) is True
        assert _bound_crosses_objective(9.0, 5.0, True) is False

    def test_a_missing_objective_cannot_be_crossed(self):
        assert _bound_crosses_objective(1.0, None, False) is False
        assert _bound_crosses_objective(1.0, float("nan"), False) is False


@pytest.mark.unit
class TestMergeRefusesACrossingBound:
    def test_the_losers_crossing_bound_is_not_adopted(self):
        """The #1059 merge must not publish a bound past the winner's incumbent.

        Route and fallback tie on objective, so the fallback wins; the route is
        the loser and carries a ``gap_certified`` bound that is numerically
        "tighter" but crosses the incumbent. Adopting it is what produced
        ``bound > objective`` on ``fac2``.
        """
        route = _sr(obj=331837498.1769339, bound=331845337.4396878, gap_certified=True)
        fallback = _sr(obj=331837498.1769339, bound=331837497.5456339, gap_certified=True)
        merged = _merge_route_and_fallback(route, fallback, False)
        assert merged is fallback
        assert merged.bound == pytest.approx(331837497.5456339), (
            "the crossing bound must not be adopted"
        )
        assert merged.bound <= merged.objective, "bound <= incumbent is not negotiable"

    def test_a_non_crossing_tighter_bound_is_still_adopted(self):
        """The guard must not disable the bound merge it protects."""
        route = _sr(obj=100.0, bound=95.0, gap_certified=True)
        fallback = _sr(obj=100.0, bound=90.0, gap_certified=True)
        merged = _merge_route_and_fallback(route, fallback, False)
        assert merged is fallback
        assert merged.bound == pytest.approx(95.0)


@pytest.mark.unit
class TestBoundsMoved:
    """The no-progress predicate behind the OA stall abort."""

    def test_a_frozen_pair_has_not_moved(self):
        """``alan``'s signature: ~7500 iterations at LB=0.0, UB=3.0."""
        assert oa._bounds_moved((0.0, 3.0), (0.0, 3.0)) is False

    def test_real_progress_moves(self):
        """``cvxnonsep_nsig30`` genuinely converges and must never be aborted."""
        assert oa._bounds_moved((130.479973, 185.909159), (130.479973, 161.840224)) is True

    def test_an_ulp_at_scale_is_not_movement(self):
        """Otherwise the counter resets forever on large-objective models, which
        is exactly where a stalled loop is most expensive."""
        assert oa._bounds_moved((3.318e8, 3.319e8), (3.318e8 + 1e-3, 3.319e8)) is False

    def test_a_missing_bound_counts_as_movement(self):
        """Unknown is not evidence of a stall; never abort on it."""
        assert oa._bounds_moved((None, 3.0), (0.0, 3.0)) is True
        assert oa._bounds_moved((0.0, 3.0), (0.0, None)) is True


@pytest.mark.slow
class TestOnRealInstances:
    def test_fac2_route_never_reports_a_bound_past_its_incumbent(self, monkeypatch, caplog):
        """End-to-end #1059 regression: ``fac2`` through the auto-route.

        Before the fix this returned ``bound=331845337.44`` against
        ``objective=331837498.18`` with ``status="optimal"``.
        """
        from pathlib import Path

        from discopt.modeling.core import from_nl

        nl = Path("python/tests/data/minlplib_nl/fac2.nl")
        if not nl.exists():
            pytest.skip(f"{nl} not in the in-repo corpus")
        # The crossing is reached through the auto-route (OA); without the flag
        # this exercises the default spatial path and proves nothing (§6).
        monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", "1")
        model = from_nl(str(nl))
        with caplog.at_level(logging.WARNING, logger="discopt.solvers.oa"):
            result = model.solve(time_limit=60)
        assert result.objective is not None
        assert result.bound is not None
        assert result.bound <= result.objective + 1e-8 * abs(result.objective), (
            f"bound {result.bound!r} crossed incumbent {result.objective!r}"
        )
        assert result.bound <= 331837498.2 + 1e-4 * 331837498.2, (
            "the reported lower bound must not exceed the MINLPLib optimum"
        )

    def test_alan_does_not_spend_the_route_budget_on_a_stalled_cut_loop(self, monkeypatch):
        """``alan`` went 0.162 s -> 30.061 s under the route, for an identical
        answer, because OA's cut loop never moved either bound and had no exit.
        """
        from pathlib import Path

        from discopt.modeling.core import from_nl

        nl = Path("python/tests/data/minlplib_nl/alan.nl")
        if not nl.exists():
            pytest.skip(f"{nl} not in the in-repo corpus")
        monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", "1")
        model = from_nl(str(nl))
        result = model.solve(time_limit=60)
        assert result.objective == pytest.approx(2.925, abs=1e-6)
        assert result.gap_certified is True
        assert result.wall_time < 10.0, (
            f"the stalled OA loop was not abandoned: {result.wall_time:.2f}s "
            "(pre-fix 30.06s, post-fix 0.41s)"
        )


@pytest.mark.unit
def test_no_progress_limit_is_generous_enough_for_a_real_plateau():
    """``fac2`` sits at LB=2.6e8 for two iterations before a cut binds and it
    jumps to the optimum on the third. The abort must be nowhere near that."""
    assert oa._OA_NO_PROGRESS_ITERATIONS >= 25
    assert np.isfinite(oa._OA_NO_PROGRESS_ITERATIONS)
