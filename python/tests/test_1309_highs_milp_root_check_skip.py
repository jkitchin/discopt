"""#1309: the HiGHS MILP route's NS-safe root cross-check can be skipped by an
ordinary time-budget race (the wall-clock budget is exhausted between HiGHS's
own solve finishing and the post-solve gate that decides whether to run the
root LP). Before the fix, ``solve_milp_std`` still returned
``status="optimal"``/``"infeasible"`` with ``gap_certified=True`` in that case,
even though the check meant to catch a wrong tree bound at the root (#1295)
never ran -- a certificate claimed but not verified (CLAUDE.md #1).

The race is reproduced deterministically (not via a wall-clock guess that a
faster/slower CI runner could dodge either way) by patching ``time.perf_counter``
to a fixed sequence: a normal small elapsed time at the pre-solve budget gate
(so HiGHS actually runs and returns kOptimal with a real time budget), then a
huge jump for every call from the post-solve gate this issue is about onward
(including ``done()``'s own trailing call for ``wall_time``, on whichever
return path is taken).
"""

import itertools
from unittest.mock import patch

import numpy as np
from discopt.solvers.lp_milp_highs import StdForm, solve_milp_std


def _fake_clock():
    """t0=0.0, pre-solve gate elapsed=1e-6 (budget intact), everything from the
    post-solve gate onward elapsed=1e6 (budget long gone) -- including the
    ``done()`` wrapper's own trailing ``time.perf_counter()`` call for
    ``wall_time``, whichever return path is taken."""
    return itertools.chain([0.0, 1e-6], itertools.repeat(1e6))


def _knapsack_sf() -> StdForm:
    # max x0 + x1 s.t. x0 + x1 <= 10, x0, x1 in {0..1e20} integer.
    # HiGHS solves this to kOptimal essentially instantly.
    return StdForm.from_arrays(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def test_root_check_skip_downgrades_optimal_to_uncertified_feasible():
    """A kOptimal result whose NS-safe root check was skipped must not claim
    gap_certified=True -- it degrades to an honest, uncertified 'feasible'
    while keeping the found incumbent (the point/objective are still usable,
    just not proven optimal)."""
    sf = _knapsack_sf()
    with patch("time.perf_counter", side_effect=_fake_clock()):
        out = solve_milp_std(sf, time_limit=10.0, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/root_check_skipped") == 1.0
    assert not out.gap_certified
    assert out.status != "optimal"
    # The incumbent must survive the downgrade rather than being thrown away.
    assert out.status == "feasible"
    assert out.x is not None
    assert out.objective == -10.0


def test_root_check_runs_and_certifies_with_an_ordinary_time_limit():
    """Sanity check: with a normal time budget, the same MILP is certified
    exactly as before -- the fix only changes the starved-budget race."""
    sf = _knapsack_sf()
    out = solve_milp_std(sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/root_check_skipped") is None
    assert out.status == "optimal"
    assert out.gap_certified
    assert out.objective == -10.0


def test_root_check_skip_with_no_time_limit_never_fires():
    """No time_limit means remaining() is always None, so the skip branch
    (and the downgrade) must never trigger regardless of how the solve took."""
    sf = _knapsack_sf()
    out = solve_milp_std(sf, time_limit=None, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/root_check_skipped") is None
    assert out.status == "optimal"
    assert out.gap_certified
