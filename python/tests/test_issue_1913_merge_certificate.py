"""#1913: the route/fallback merge must not throw away the fallback's certificate.

``_merge_route_and_fallback`` protects a certified fallback from a route objective
that looks better, but it tested "certified" with ``_gap_is_closed``, whose gap
threshold is a fixed 1e-6, far below the default relative ``gap_tolerance`` of 1e-4.
On ``smallinvDAXr2b200-220`` the fallback certified optimal at a relative gap of
1.7e-5, the uncertified route's objective was 1.2e-10 better, the route won, and
the solve reported ``feasible`` with a closed gap. ``_result_certifies`` reads the
producer's own verdict (``status == "optimal"`` with ``gap_certified``).

When the route's verified point lies inside the fallback's certified gap, it is
reported under that certificate. The bound is at most the route objective, which
is at most the fallback objective, so both the absolute and the relative gap
against it are smaller. A route objective beyond the certified bound is
impossible, and the certified fallback is kept.
"""

from __future__ import annotations

import pytest
from discopt.modeling.core import SolveResult
from discopt.solver import _merge_route_and_fallback, _result_certifies

pytestmark = pytest.mark.unit


def _R(objective, bound, *, gap_certified=False, status="feasible", termination=None):
    r = SolveResult(status=status, objective=objective, bound=bound, gap_certified=gap_certified)
    r.termination = termination
    return r


# smallinvDAXr2b200-220, measured (minimize; MINLPLib optimum 156.6042679)
ROUTE_OBJ, ROUTE_BOUND = 156.6042678865059, 156.58534306610653
FB_OBJ, FB_BOUND = 156.60426788662008, 156.6016245725951


def test_result_certifies_reads_the_producers_verdict():
    assert _result_certifies(_R(FB_OBJ, FB_BOUND, gap_certified=True, status="optimal"))
    # A valid but open gap is not a certificate (#1059's "certified is not closed").
    assert not _result_certifies(_R(FB_OBJ, 100.0, gap_certified=True, status="feasible"))
    assert not _result_certifies(_R(FB_OBJ, FB_BOUND, gap_certified=False, status="optimal"))
    assert not _result_certifies(None)


def test_a_certified_fallback_is_not_lost_to_a_route_objective_a_hair_better():
    route = _R(ROUTE_OBJ, ROUTE_BOUND)
    fb = _R(FB_OBJ, FB_BOUND, gap_certified=True, status="optimal", termination="gap")
    out = _merge_route_and_fallback(route, fb, is_maximize=False)
    assert out.status == "optimal" and out.gap_certified, (out.status, out.gap_certified)
    assert out.bound == pytest.approx(FB_BOUND)  # the certified bound is reported
    assert out.bound <= out.objective  # certificate invariant, minimize


def test_a_route_point_inside_the_certified_gap_is_reported_under_it():
    """smallinvDAXr5b200-220: the fallback certified 156.6169; the route held the
    optimum 156.6043. The better verified point is reported, still certified."""
    route = _R(156.6042678865059, 156.585)
    fb = _R(156.61690883644175, 156.60140466056933, gap_certified=True, status="optimal")
    out = _merge_route_and_fallback(route, fb, is_maximize=False)
    assert out.objective == pytest.approx(156.6042678865059)
    assert out.status == "optimal" and out.gap_certified
    assert out.bound == pytest.approx(156.60140466056933) and out.bound <= out.objective


def test_maximize_mirror():
    route = _R(-ROUTE_OBJ, -ROUTE_BOUND)
    fb = _R(-FB_OBJ, -FB_BOUND, gap_certified=True, status="optimal")
    out = _merge_route_and_fallback(route, fb, is_maximize=True)
    assert out.status == "optimal" and out.gap_certified
    assert out.bound >= out.objective  # an upper bound, maximize


def test_a_route_objective_past_the_certified_bound_keeps_the_fallback():
    """Impossible for a feasible point: something is inconsistent, so the proof wins."""
    route = _R(150.0, 140.0)  # below the fallback's certified lower bound 156.6016
    fb = _R(FB_OBJ, FB_BOUND, gap_certified=True, status="optimal")
    out = _merge_route_and_fallback(route, fb, is_maximize=False)
    assert out is fb and out.objective == pytest.approx(FB_OBJ)


def test_an_uncertified_fallback_does_not_certify_the_route():
    route = _R(ROUTE_OBJ, ROUTE_BOUND)
    fb = _R(FB_OBJ + 1.0, 120.0, gap_certified=False, status="feasible")
    out = _merge_route_and_fallback(route, fb, is_maximize=False)
    assert out is route and not out.gap_certified and out.status == "feasible"
