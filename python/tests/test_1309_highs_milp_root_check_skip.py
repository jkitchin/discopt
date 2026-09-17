"""#1309: the HiGHS MILP route's NS-safe root cross-check can be skipped by an
ordinary time-budget race (the wall-clock budget is exhausted between HiGHS's
own solve finishing and the post-solve gate that decides whether to run the
root LP). Before the fix, ``solve_milp_std`` still returned
``status="optimal"``/``"infeasible"`` with ``gap_certified=True`` in that case,
even though the check meant to catch a wrong tree bound at the root (#1295)
never ran -- a certificate claimed but not verified (CLAUDE.md #1).

A ``time_limit`` far smaller than the time this call itself takes (imports,
matrix marshaling, HiGHS setup) reliably reproduces the race deterministically:
HiGHS solves the trivial MILP well inside that "budget" in wall-clock terms,
but by the time ``remaining()`` is checked after the solve, the elapsed time
already exceeds the requested limit.
"""

import numpy as np
from discopt.solvers.lp_milp_highs import StdForm, solve_milp_std


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
    out = solve_milp_std(sf, time_limit=0.001, gap_tolerance=1e-4, max_nodes=1000)
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
