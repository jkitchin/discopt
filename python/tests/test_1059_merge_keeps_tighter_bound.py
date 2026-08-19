"""#1059: the route/fallback merge must keep the tighter dual bound and report
the work both halves actually did.

The auto-route runs the MIP-NLP family on half the budget, then hands the
remainder to the default spatial path and returns the better of the two. Two
defects in that merge were measured on ``squfl025-040`` at a 60 s limit:

===================== ======== ======== =======
side                   obj      bound    nodes
===================== ======== ======== =======
route (OA, 30.2 s)      423.98    76.87        0
fallback (B&B, 29.8 s) 1139.53   127.40      991
===================== ======== ======== =======

1. The merge adopted the loser's bound only when the loser was
   ``gap_certified``. A time-limited B&B is never ``gap_certified`` yet its
   global dual bound is valid -- and the *non-routed* path reports precisely
   that bound. So the merge published 76.87 and discarded 127.40, making the
   route strictly worse on the dual side than no route at all.
2. ``node_count`` came from the winner alone, so a 61 s solve that explored 991
   nodes reported 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solver import _merge_route_and_fallback


class _R:
    """Minimal stand-in for the fields the merge touches."""

    def __init__(self, objective, bound, *, gap_certified=False, node_count=0, status="feasible"):
        self.objective = objective
        self.bound = bound
        self.gap_certified = gap_certified
        self.node_count = node_count
        self.status = status
        self.gap = None


@pytest.mark.unit
class TestTighterBoundIsKept:
    def test_uncertified_fallback_bound_is_adopted_when_tighter(self):
        """The squfl025-040 shape: route wins the incumbent, fallback the bound."""
        route = _R(423.98, 76.87, gap_certified=True, node_count=0)
        fallback = _R(1139.53, 127.40, gap_certified=False, node_count=991)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.objective == pytest.approx(423.98)
        assert out.bound == pytest.approx(127.40), "the tighter valid bound was discarded"

    def test_maximize_sense_takes_the_smaller_upper_bound(self):
        route = _R(-70.0, 2176.14, gap_certified=True, node_count=3)
        fallback = _R(-211.0, 860.84, gap_certified=False, node_count=447)
        out = _merge_route_and_fallback(route, fallback, is_maximize=True)
        assert out.objective == pytest.approx(-70.0)
        assert out.bound == pytest.approx(860.84)

    def test_a_looser_loser_bound_is_ignored(self):
        route = _R(1.0, 0.9, gap_certified=True, node_count=0)
        fallback = _R(2.0, 0.1, gap_certified=False, node_count=50)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.bound == pytest.approx(0.9)

    def test_a_crossing_loser_bound_is_still_refused(self):
        """Soundness outranks tightness: a bound past the incumbent is broken."""
        route = _R(100.0, 50.0, gap_certified=True, node_count=0)
        fallback = _R(500.0, 300.0, gap_certified=False, node_count=10)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.bound == pytest.approx(50.0), "a bound above the incumbent must not be published"

    def test_a_non_finite_loser_bound_is_refused(self):
        route = _R(10.0, 1.0, gap_certified=True, node_count=0)
        fallback = _R(20.0, np.inf, gap_certified=False, node_count=7)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.bound == pytest.approx(1.0)

    def test_the_gap_is_recomputed_from_the_adopted_bound(self):
        route = _R(400.0, 76.87, gap_certified=True, node_count=0)
        fallback = _R(1100.0, 127.40, gap_certified=False, node_count=991)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.gap == pytest.approx(abs(127.40 - 400.0) / 400.0)


@pytest.mark.unit
class TestWorkIsReported:
    def test_node_count_sums_both_halves(self):
        route = _R(423.98, 76.87, gap_certified=True, node_count=0)
        fallback = _R(1139.53, 127.40, gap_certified=False, node_count=991)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.node_count == 991, "991 nodes of real work were reported as 0"

    def test_node_count_sums_when_the_fallback_wins(self):
        route = _R(900.0, 10.0, gap_certified=True, node_count=4)
        fallback = _R(100.0, 20.0, gap_certified=False, node_count=750)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.objective == pytest.approx(100.0)
        assert out.node_count == 754

    def test_a_missing_node_count_on_one_side_is_not_fatal(self):
        route = _R(5.0, 1.0, gap_certified=True, node_count=None)
        fallback = _R(9.0, 2.0, gap_certified=False, node_count=33)
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.node_count == 33


@pytest.mark.unit
class TestUnchangedContracts:
    def test_a_certified_fallback_still_outranks_a_better_route_number(self):
        route = _R(1.0, 0.0, gap_certified=False, node_count=0)
        fallback = _R(2.0, 2.0, gap_certified=True, node_count=5)
        fallback.gap = 0.0
        out = _merge_route_and_fallback(route, fallback, is_maximize=False)
        assert out.objective == pytest.approx(2.0)

    def test_a_missing_side_passes_the_other_through(self):
        r = _R(1.0, 0.0, node_count=3)
        assert _merge_route_and_fallback(None, r, is_maximize=False) is r
        assert _merge_route_and_fallback(r, None, is_maximize=False) is r
