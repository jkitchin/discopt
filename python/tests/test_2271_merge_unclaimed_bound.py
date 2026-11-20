"""#2271: the route/fallback merge installed a bound its producer did not claim.

``_merge_route_and_fallback`` keeps the tighter dual bound from either side. It
gated that on finiteness, tightness and not crossing the incumbent, but not on
the loser's ``bound_valid``. So a bound its own producer declined to claim valid
could replace a proved one under a live certificate, and the result read
``gap_certified=True`` beside ``bound_valid=False`` -- a certificate resting on a
bound nobody proved, which the ``SolveResult.bound_valid`` contract forbids.

Both shapes below are the nightly adversary's setter traces, numbers verbatim
(discopt 0.12.0 wheel). On ``main`` neither instance reaches the merge any more
(both certify in ~1 s by another route, measured), so the merge is tested
directly: the defect is in the merge, not in the instances.
"""

from __future__ import annotations

import itertools

import pytest
from discopt.modeling.core import SolveResult
from discopt.solver import _merge_route_and_fallback


def _res(status, objective, bound, *, certified, valid, source=None):
    # The bound goes in at construction: ``__post_init__`` revokes a certificate
    # that has no bound, so setting it afterwards would build an uncertified input
    # and the tests would pass without exercising the merge they target.
    r = SolveResult(status=status, objective=objective, bound=bound, gap_certified=certified)
    r._set_bound(bound, valid=valid, source=source if valid else None)
    assert (r.gap_certified, r.bound_valid) == (certified, valid), "fixture did not build"
    return r


def _assert_consistent(r):
    assert not (r.gap_certified and not r.bound_valid), (
        f"certified gap on an unclaimed bound: status={r.status} bound={r.bound} "
        f"gap_certified={r.gap_certified} bound_valid={r.bound_valid}"
    )


@pytest.mark.unit
def test_tloss_lift_race_shape_keeps_the_proved_bound():
    """``_ipx_lift_race`` on ``tloss``: the lifted slice timed out with an
    unclaimed 16.3, the unlifted slice certified 16.299999999982692."""
    lifted = _res("time_limit", 16.3, 16.3, certified=False, valid=False)
    unlifted = _res("optimal", 16.3, 16.299999999982692, certified=True, valid=True,
                    source="bnb_tree")  # fmt: skip
    out = _merge_route_and_fallback(lifted, unlifted, is_maximize=False)
    _assert_consistent(out)
    assert out.status == "optimal" and out.gap_certified
    assert out.bound == 16.299999999982692, "the unclaimed 16.3 replaced the proved bound"
    assert out.bound_valid is True and out.bound_source == "bnb_tree"


@pytest.mark.unit
def test_cvxnonsep_oa_shape_keeps_the_proved_bound():
    """The issue's second instance, ``cvxnonsep_nsig30r``: OA returned
    ``feasible`` with an unclaimed 156.4198391794836, NLP-BB certified
    156.41298071074038 (different caller, same merge)."""
    oa = _res("feasible", 156.4266874638832, 156.4198391794836, certified=False, valid=False)
    nlpbb = _res("optimal", 156.4266874638832, 156.41298071074038, certified=True, valid=True,
                 source="bnb_tree")  # fmt: skip
    out = _merge_route_and_fallback(oa, nlpbb, is_maximize=False)
    _assert_consistent(out)
    assert out.gap_certified
    assert out.bound == 156.41298071074038


@pytest.mark.unit
def test_an_unclaimed_bound_may_still_replace_an_unclaimed_one():
    """Nothing is asserted on either side, so tighter is the only preference left;
    the gate must not throw that away."""
    route = _res("feasible", 10.0, 1.0, certified=False, valid=False)
    fallback = _res("feasible", 20.0, 5.0, certified=False, valid=False)
    out = _merge_route_and_fallback(route, fallback, is_maximize=False)
    assert out.bound == 5.0 and out.bound_valid is False


@pytest.mark.unit
def test_no_merge_publishes_a_certificate_on_an_unclaimed_bound():
    """Every combination of side, sense, certificate and claim, with the loser's
    bound tighter than the winner's and inside the incumbent. The published
    result must be self-consistent, and a claimed bound may only be one that
    some input claimed."""
    checked = 0
    for is_max, route_better, r_cert, f_cert, r_valid, f_valid in itertools.product(
        (False, True), (False, True), (False, True), (False, True), (False, True), (False, True)
    ):
        if (r_cert and not r_valid) or (f_cert and not f_valid):
            continue  # __post_init__ rule 1: a certificate is a claim
        s = -1.0 if is_max else 1.0  # reported sense: work in "min" then flip
        r_obj, f_obj = (s * 10.0, s * 10.5) if route_better else (s * 10.5, s * 10.0)
        r_bnd, f_bnd = (s * 9.0, s * 9.9) if route_better else (s * 9.9, s * 9.0)
        route = _res("optimal" if r_cert else "feasible", r_obj, r_bnd, certified=r_cert,
                     valid=r_valid, source="lp_dual")  # fmt: skip
        fb = _res("optimal" if f_cert else "feasible", f_obj, f_bnd, certified=f_cert,
                  valid=f_valid, source="bnb_tree")  # fmt: skip
        claimed = {b for b, v in ((r_bnd, r_valid), (f_bnd, f_valid)) if v}
        out = _merge_route_and_fallback(route, fb, is_maximize=is_max)
        _assert_consistent(out)
        if out.bound_valid:
            assert out.bound in claimed, (is_max, route_better, r_cert, f_cert, r_valid, f_valid)
        checked += 1
    assert checked == 36, checked  # the sweep ran (CLAUDE.md §6)
