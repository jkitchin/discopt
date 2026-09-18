"""#1320: the HiGHS MILP route certified when the NS-safe root cross-check RAN
but was inconclusive.

#1309 applied "a certificate whose safety net never ran is not a certificate" to
the branch where the root cross-check is *skipped* for want of budget. The same
principle was not applied when the check runs and settles nothing -- ``solve_lp_std``
comes back ``time_limit`` or ``error`` (a HiGHS ``kUnknown``, or a
sentinel-magnitude readback refusal), and the code only recorded
``milp/root_check_inconclusive`` before returning the result unchanged:
``optimal``/``infeasible``, ``gap_certified=True``, ``root_bound=None``.

Part 2 of the issue: even when the #1309 skip branch *did* downgrade, the result
still sent mixed signals -- ``gap=0.0``, ``bound_valid=True`` and a route label
reading "verified HiGHS route" all continued to say "certified" beside an
uncertified status. A downgraded result now reports a consistent state.

The starved-budget race is reproduced deterministically with the same fake-clock
technique as #1309's own test, rather than a wall-clock guess a faster or slower
runner could dodge either way.
"""

import itertools
from unittest.mock import patch

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import HighsOutcome, StdForm, solve_milp_std

TL = 10.0


def _starved_clock():
    """Budget intact at the pre-solve gate (so HiGHS really runs and returns its
    own status), a sliver left at the root-check gate (so the check RUNS rather
    than being skipped), and gone by the time ``solve_lp_std`` could finish it."""
    return itertools.chain([0.0, 1e-6, TL - 1e-9, TL - 1e-9, TL - 1e-9], itertools.repeat(TL + 1))


def _knapsack_sf() -> StdForm:
    """max x0 + x1 s.t. x0 + x1 <= 10, x0, x1 integer -- HiGHS returns kOptimal."""
    return StdForm.from_arrays(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def _infeasible_sf() -> StdForm:
    """2x == 1 with x integer -- HiGHS returns kInfeasible."""
    return StdForm.from_arrays(
        c=np.array([1.0]),
        A=np.array([[2.0]]),
        b=np.array([1.0]),
        xl=np.array([0.0]),
        xu=np.array([10.0]),
        int_idx=np.array([0]),
    )


def test_inconclusive_root_check_downgrades_optimal_to_uncertified_feasible():
    """kOptimal + an inconclusive root check must not claim ``gap_certified``.

    It degrades to an honest ``feasible`` that keeps the incumbent -- the point and
    its objective are still usable, they are just not proven optimal.
    """
    with patch("time.perf_counter", side_effect=_starved_clock()):
        out = solve_milp_std(_knapsack_sf(), time_limit=TL, gap_tolerance=1e-4, max_nodes=1000)

    # The check RAN (this is the #1320 arm, not #1309's skip) and settled nothing.
    assert out.stats.get("milp/root_check_ran") == 1.0
    assert out.stats.get("milp/root_check_inconclusive") == 1.0
    assert out.stats.get("milp/root_check_skipped") is None

    assert not out.gap_certified
    assert out.status == "feasible"
    assert out.x is not None
    assert out.objective == -10.0


def test_inconclusive_root_check_downgrades_infeasible_to_error():
    """kInfeasible has no incumbent to fall back on, so an unverifiable
    infeasibility claim becomes ``error`` rather than a certified ``infeasible``."""
    with patch("time.perf_counter", side_effect=_starved_clock()):
        out = solve_milp_std(_infeasible_sf(), time_limit=TL, gap_tolerance=1e-4, max_nodes=1000)

    assert out.stats.get("milp/root_check_inconclusive") == 1.0
    assert not out.gap_certified
    assert out.status == "error"


def test_downgraded_result_reports_no_proven_bound():
    """#1320 part 2: a downgraded result must not keep signalling "certified".

    The unchecked tree bound gives way to the NS-safe root bound -- ``None`` here,
    since producing one is exactly what the check failed to do -- so nothing
    downstream can read a proven zero gap off it.
    """
    with patch("time.perf_counter", side_effect=_starved_clock()):
        out = solve_milp_std(_knapsack_sf(), time_limit=TL, gap_tolerance=1e-4, max_nodes=1000)

    assert out.root_bound is None
    assert out.bound is None, "an unverified tree bound must not be reported as a bound"
    assert out.labels.get("milp/certificate") == "declined"
    assert out.labels.get("milp/bound_provenance") == "none"


def test_skipped_root_check_also_reports_no_proven_bound():
    """The #1309 skip branch gets the same consistency treatment (#1320 part 2):
    before this, it downgraded the *status* but left ``bound`` (hence ``gap=0.0``
    and ``bound_valid=True``) in place."""

    def skipped_clock():
        # Budget intact at the pre-solve gate, long gone by the root-check gate.
        return itertools.chain([0.0, 1e-6], itertools.repeat(1e6))

    with patch("time.perf_counter", side_effect=skipped_clock()):
        out = solve_milp_std(_knapsack_sf(), time_limit=TL, gap_tolerance=1e-4, max_nodes=1000)

    assert out.stats.get("milp/root_check_skipped") == 1.0
    assert out.status == "feasible"
    assert not out.gap_certified
    assert out.bound is None
    assert out.labels.get("milp/certificate") == "declined"


def test_end_to_end_downgrade_is_consistent_through_model_solve():
    """Through ``Model.solve`` the whole result must agree with itself: an
    uncertified status, no gap, no valid bound, and a route label that does not
    call itself "verified".

    Driven by the condition the issue reports on natural models -- the root LP
    coming back ``error`` (a HiGHS ``kUnknown``, or a sentinel-magnitude readback
    refusal) -- rather than by a fake clock. ``Model.solve`` makes an unpredictable
    number of ``perf_counter`` calls before reaching ``solve_milp_std``, so a fixed
    clock sequence here would be a probe that silently stops measuring the arm it
    is named after (CLAUDE.md, measurement discipline #6).
    """
    m = dm.Model("knap")
    m.integer("x0", lb=0, ub=20)
    m.integer("x1", lb=0, ub=20)
    x0, x1 = m._variables
    m.subject_to(x0 + x1 <= 10)
    m.maximize(x0 + x1)

    calls = []

    def inconclusive_root_lp(sf, **kwargs):
        calls.append(sf)
        return HighsOutcome("error", message="kUnknown", highs_status="kUnknown")

    with patch.object(lp_milp_highs, "solve_lp_std", inconclusive_root_lp):
        r = m.solve(time_limit=30.0)

    assert calls, "the root cross-check never ran, so this asserts nothing"
    assert not r.gap_certified, "certified a result whose root cross-check settled nothing"
    assert r.status != "optimal"
    assert r.gap != 0.0
    assert not r.bound_valid
    assert r.algorithm_route is not None
    assert "unverified HiGHS route" in r.algorithm_route


def test_conclusive_root_check_still_certifies():
    """Regression fence: with an ordinary budget the root check runs, concludes,
    and the same MILP is certified exactly as before -- the fix only touches the
    inconclusive arm."""
    out = solve_milp_std(_knapsack_sf(), time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/root_check_inconclusive") is None
    assert out.status == "optimal"
    assert out.gap_certified
    assert out.objective == -10.0


def test_certified_route_label_still_says_verified():
    """The label demotion is confined to declined certificates: a normal certified
    solve keeps the route's "verified" wording."""
    m = dm.Model("knap2")
    m.integer("x0", lb=0, ub=20)
    m.integer("x1", lb=0, ub=20)
    x0, x1 = m._variables
    m.subject_to(x0 + x1 <= 10)
    m.maximize(x0 + x1)

    r = m.solve(time_limit=30.0)
    assert r.status == "optimal"
    assert r.gap_certified
    assert r.algorithm_route is not None
    assert "verified HiGHS route" in r.algorithm_route
    assert "unverified" not in r.algorithm_route


@pytest.mark.parametrize("sf_fn", [_knapsack_sf, _infeasible_sf])
def test_no_time_limit_never_downgrades(sf_fn):
    """``remaining()`` is always ``None`` without a time limit, so neither the skip
    nor the inconclusive branch may fire."""
    out = solve_milp_std(sf_fn(), time_limit=None, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/root_check_skipped") is None
    assert out.stats.get("milp/root_check_inconclusive") is None
    assert out.gap_certified


# ── accepted cost of the decertify rule ───────────────────────────────────


def test_integral_infeasibility_under_an_unbounded_root_lp_is_certified():
    """Round-4 review, finding 3. The trade this pinned is no longer taken (#1337).

    ``2x == 1`` with ``x`` integer has no solution, but the infeasibility is
    purely INTEGRAL: the root LP relaxation is perfectly happy at ``x = 0.5``,
    and here it is even unbounded (``min -y`` with ``y in [0, 1e20]``). So the
    NS-safe root cross-check came back ``unbounded`` -- which settles nothing --
    and the decertify rule refused to pass HiGHS's raw ``kInfeasible`` through.

    The gap this docstring called "real and worth closing" is closed the way it
    named: not by trusting the label, but by asking the root LP the question the
    branch actually cares about. What defeated the check was the OBJECTIVE, so the
    re-check drops it -- same LP, same box, zero objective -- and the relaxation
    answers ``feasible``, leaving the ``kInfeasible`` claim standing on exactly the
    footing the ordinary ``lp.status in ("optimal", "feasible")`` path already
    gives it.

    The assertion is TIGHTENED rather than relaxed: ``error`` was acceptable while
    the check could not be run, and is not acceptable now that it can. The §1 line
    is kept alongside it.
    """
    m = dm.Model("integral_infeasible_unbounded_root")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(2 * x == 1)
    y = m.continuous("y", lb=0, ub=1e20)
    m.minimize(-y)
    r = m.solve(time_limit=60)
    # The §1 line: never a WRONG certified status.
    assert r.status != "optimal"
    assert r.status != "unbounded"
    assert r.status == "infeasible", (
        f"the feasibility-only root re-check (#1337) must decide this, got {r.status!r}"
    )
