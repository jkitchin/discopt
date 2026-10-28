"""#2013: the HiGHS lazy master's in-tree check-in reports the RUNNING bound.

``solve_milp_with_lazy_cuts`` rebuilds the HiGHS tree after every lazy cut. Its
restart check-in and its final result already carried the running max of the
trees' dual bounds, but the in-tree (``kCallbackMipInterrupt``) check-in passed the
CURRENT tree's bound, which restarts at the rebuilt tree's root. The #1066 route
guard judged that sawtooth and abandoned converging routes: slay07m (feasible at
30 s -> optimal 19.8 s) and p_ball_10b_5p_2d_m (feasible -> optimal 19.7 s).
Rows are only ever added to the master, so a finished tree's bound stays valid.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("highspy", reason="the HiGHS master is a core dependency")

from discopt.solvers.milp_highs import solve_milp_with_lazy_cuts  # noqa: E402


def _tree_that_lasts():
    """A MILP whose tree really runs, plus a separator that restarts it three times
    (the #1066 fixture: 45 binaries, seeded, ~0.15 s)."""
    rng = np.random.default_rng(7)
    n = 45
    c = -rng.integers(5, 60, size=n).astype(float)
    a_ub = rng.integers(1, 40, size=(6, n)).astype(float)
    b_ub = a_ub.sum(axis=1) * 0.32
    state = {"k": 0}

    def _sep(x):
        if state["k"] < 3 and x.sum() > 12 - state["k"]:
            state["k"] += 1
            return [(np.ones(n), 12.0 - state["k"])]
        return None

    return {
        "c": c,
        "A_ub": a_ub,
        "b_ub": b_ub,
        "bounds": [(0.0, 1.0)] * n,
        "integrality": np.ones(n, dtype=int),
        "lazy_callback": _sep,
    }


@pytest.mark.smoke
def test_the_interrupt_snapshot_carries_the_running_bound_not_the_fresh_trees():
    """After a restart, an in-tree poll must never report a bound
    below the bound a finished tree already proved."""
    snaps: list[dict] = []

    def _record(snap):
        snaps.append(dict(snap))
        return False

    res = solve_milp_with_lazy_cuts(
        **_tree_that_lasts(), terminate_callback=_record, terminate_poll_s=0.0
    )
    assert res.callback_stats["restarts"] >= 1
    running = None
    checked = 0
    for snap in snaps:
        if snap["context"] == "restart" and snap["dual_bound"] is not None:
            running = (
                float(snap["dual_bound"]) if running is None else max(running, snap["dual_bound"])
            )
        elif snap["context"] == "interrupt" and running is not None:
            assert snap["dual_bound"] is not None
            assert float(snap["dual_bound"]) >= running - 1e-9 * max(1.0, abs(running)), (
                f"in-tree poll reported {snap['dual_bound']} below the running bound {running}"
            )
            checked += 1
    assert checked > 0, "no in-tree poll fired after a restart, so this asserts nothing"
