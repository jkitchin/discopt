"""An *expired* MILP budget must not be wire-identical to *no* budget (#928).

Both MILP entry points in ``lp_bindings.rs`` used to take a bare ``f64`` and map it
with ``if time_limit_s > 0.0 { Some(t) } else { None }``, while the Python side spelled
"no limit" as ``0.0``. The two meanings therefore collapsed onto one wire value, and
the collapse fell the *unsafe* way: a caller that had already spent a shared budget
across earlier attempts — ``MilpRelaxationModel.solve`` under
``DISCOPT_LP_WARM_DEADLINE=1``, whose ``_remaining()`` returns exactly ``0.0`` once the
warm attempts drain it — asked for "stop now" and launched an **unbounded** B&B.

Measured cost of the collapse: AMP's ``solve(solver="amp", time_limit=3.0)`` went from
3.43 s with the flag off to >350 s with it on, and it is the mechanism behind the
"sporadic severe modes" (contvar 500.6 s, bchoco08 80.9 s against a 20 s budget)
previously attributed to diffuse budget accounting.

This is the sentinel-collapse class already documented for ``INF = 1e20`` in the LP
layer, one level up: a value that is *in range* for the parameter is also its own
"absent" marker. The contract is now the one ``parse_deadline`` already applied to the
LP entries — ``None``/``+inf`` mean no limit, negative/NaN are rejected loudly, and
``Some(0.0)`` is an already-elapsed deadline.
"""

import time

import numpy as np
import pytest
from discopt.solvers.milp_simplex import solve_milp

# A knapsack-ish 0/1 MILP sized so the unlimited solve needs a real search (53 nodes)
# while still finishing in milliseconds, which is what lets these live in the `smoke`
# suite and run on every PR. The separation between "stopped at the deadline" and "ran
# the tree" is therefore read off the NODE COUNT (1 vs 53), not the clock.
_N = 34


def _instance():
    rng = np.random.default_rng(9280)
    c = -rng.integers(10, 60, size=_N).astype(np.float64)  # maximize value (min -value)
    w = rng.integers(10, 60, size=_N).astype(np.float64)
    A = np.vstack([w, w[::-1].copy()])
    b = np.array([0.5 * w.sum(), 0.5 * w.sum()], dtype=np.float64)
    bounds = [(0.0, 1.0)] * _N
    integrality = np.ones(_N, dtype=np.int64)
    return c, A, b, bounds, integrality


def _solve(time_limit):
    c, A, b, bounds, integrality = _instance()
    t0 = time.perf_counter()
    res = solve_milp(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integrality,
        time_limit=time_limit,
    )
    return res, time.perf_counter() - t0


@pytest.mark.smoke
def test_expired_budget_returns_immediately():
    """``time_limit=0.0`` means "my budget is spent", not "run forever".

    Before the fix, 0.0 was mapped back to "no deadline" and this call ran the whole
    tree — verified by rebuilding the extension at the parent commit: this test fails
    there with 53 nodes and ``status=OPTIMAL``.
    """
    res, wall = _solve(0.0)

    # The DISCRIMINATOR is the node count, not the clock. This instance solves in
    # milliseconds either way, so a wall-time threshold would pass even with the bug
    # present and the test would be decorative. An expired budget must stop at the
    # first deadline poll (1 node); the unlimited solve takes 53. See
    # ``test_expired_and_unlimited_are_distinguishable`` for the paired assertion.
    assert res.node_count <= 2, (
        f"solve_milp(time_limit=0.0) explored {res.node_count} nodes: an exhausted "
        "budget was treated as 'no limit' (the #928 sentinel collapse), so the B&B "
        "launched unbounded"
    )
    # Whatever it reports must not be a certified answer — it did no search.
    assert res.status.name != "OPTIMAL", (
        f"solve_milp(time_limit=0.0) claimed {res.status} without spending any budget"
    )
    # Operational guard: on a larger instance the same bug is a hang, not a node count.
    assert wall < 2.0, f"solve_milp(time_limit=0.0) ran {wall:.2f}s"


@pytest.mark.smoke
def test_no_limit_still_means_no_limit():
    """The other half of the contract: ``None`` must still run to optimality.

    Without this, "return immediately" could be satisfied by breaking every budget,
    which would be the opposite defect and would silently uncertify the solver.
    """
    res, _ = _solve(None)

    assert res.status.name == "OPTIMAL", (
        f"solve_milp(time_limit=None) returned {res.status}; None must mean no limit"
    )
    assert res.objective is not None and np.isfinite(res.objective)


@pytest.mark.smoke
def test_expired_and_unlimited_are_distinguishable():
    """The two must not be the same call. This is the collapse itself, asserted.

    Stated as a *comparison* rather than two absolute thresholds so it keeps its
    meaning on any machine: the unlimited solve does strictly more work than the
    expired one, and the executed-comparison count is asserted so the test cannot
    silently degrade to checking nothing.

    Work is counted in NODES, deliberately. Wall time does not separate these two at
    this size — measured 0.0031 s expired vs 0.0016 s unlimited, i.e. the *expired*
    call is the slower one, because both pay the same fixed setup and neither runs
    long enough for the search to dominate. An earlier draft asserted
    ``wall_unlimited > wall_expired`` and passed only by scheduling luck.
    """
    comparisons = 0

    expired, _ = _solve(0.0)
    unlimited, _ = _solve(None)

    assert unlimited.node_count > expired.node_count, (
        f"expired budget explored {expired.node_count} nodes vs {unlimited.node_count} "
        "unlimited — the two budgets are still the same wire value"
    )
    comparisons += 1
    assert unlimited.status.name == "OPTIMAL" and expired.status.name != "OPTIMAL", (
        f"expired={expired.status}, unlimited={unlimited.status} — the two budgets did "
        "not lead to different outcomes"
    )
    comparisons += 1

    assert comparisons == 2, "PROBE NEVER FIRED: expected 2 executed comparisons"


@pytest.mark.smoke
@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_invalid_budget_is_rejected_at_the_binding(bad):
    """Negative/NaN are refused at the wire, not silently reinterpreted.

    This is ``parse_deadline``'s rule, now applied to the MILP entries too. It matters
    most for a *negative* budget: under the old ``> 0.0`` test it became "no limit" —
    the single most dangerous possible reading of "you are already past your deadline".

    Asserted against the binding rather than :func:`solve_milp` because the Python
    wrapper clamps with ``max(0.0, ...)``, which sends both of these in as an *expired*
    budget. That clamp fails closed (stop now) and is left alone; this test pins the
    layer that must never again fail open.
    """
    from discopt._rust import solve_milp_py

    # min -x  s.t.  x + s = 1, x binary. One structural column, one slack.
    with pytest.raises(ValueError, match="non-negative and not NaN"):
        solve_milp_py(
            np.array([-1.0, 0.0]),
            np.array([[1.0, 1.0]]),
            np.array([1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0], dtype=np.int64),
            1,
            time_limit_s=bad,
        )


@pytest.mark.smoke
def test_negative_budget_through_solve_milp_fails_closed():
    """The wrapper's clamp must resolve a negative budget to *expired*, never unlimited."""
    res, _ = _solve(-5.0)
    assert res.node_count <= 2 and res.status.name != "OPTIMAL", (
        f"solve_milp(time_limit=-5.0) explored {res.node_count} nodes -> {res.status}: "
        "a budget already blown by 5 s was read as 'no limit'"
    )


@pytest.mark.smoke
def test_infinite_budget_means_no_limit():
    """``+inf`` is the natural spelling of an uncapped budget and must not raise.

    Before the fix this did not merely mis-route — it **panicked** the driver.
    ``inf > 0.0`` sent ``Some(inf)`` through to ``Duration::from_secs_f64`` in
    ``milp_driver.rs``, which aborts with "cannot convert float seconds to Duration:
    value is either too big or NaN" and surfaces in Python as ``PanicException``. Found
    by running this file against the pre-fix extension; it is a second, independent
    defect that the shared ``parse_budget_secs`` validation closes.
    """
    res, _ = _solve(float("inf"))
    assert res.status.name == "OPTIMAL"
