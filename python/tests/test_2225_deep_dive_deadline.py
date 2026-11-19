"""#2225: a deep dive does not carry the MILP search past its deadline.

``data/iis_2225_enlight4_probe.npz`` is one MIPcc27 ``enlight4`` IIS probe (the model
with the probe's rows, bounds and integrality; HiGHS proves it infeasible). Several
integer columns are unbounded and the objective is zero, so the ported HiGHS search
dives very deep without deciding it. When the deadline stops the dive,
``openNodesToQueue`` hands every node on the stack to the queue, and each hand-off built
its reduced domain-change stack by a pass over the WHOLE change stack: O(depth x stack)
after the deadline. Measured on this probe, presolve off: a 0.8 s limit returned at
3.45 s, a 3 s limit at 21 s, a 30 s limit not at all (> 150 s). In the IIS entry the
probe's 30 s budget never returned, the portfolio member was abandoned, and the run
ended with no subsystem.

The reduced stack is now built from the per-column last positions and a per-entry
chain flag kept as changes are pushed, so a hand-off costs O(min(stack, 2 ncols)).
The output is identical (the Rust test ``reduced_stack_matches_the_chain_walk_on_long_
stacks`` compares it with HiGHS's chain walk), so the search is unchanged: a node cap
of 60,000 stops at the same node either way (1.76 s vs 7.45 s before).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

DATA = Path(__file__).parent / "data" / "iis_2225_enlight4_probe.npz"


def _load():
    z = np.load(DATA)
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return A, z["row_lo"], z["row_hi"], z["col_lb"], z["col_ub"], z["is_int"]


def test_deep_dive_stops_near_its_deadline():
    """A 4 s limit returns well within 4 + 8 s, stopped by the clock. Before the fix the
    post-deadline queue hand-off alone took tens of seconds (21 s at a 3 s limit)."""
    from discopt.solvers.milp_handle import MilpHandle

    A, rlo, rhi, lb, ub, integ = _load()
    h = MilpHandle(np.zeros(A.shape[1]), A, rlo, rhi, lb, ub, integ)
    limit = 4.0
    t = time.perf_counter()
    r = h.solve(time_limit=limit, presolve=False)
    wall = time.perf_counter() - t
    # The probe is infeasible, so no feasible verdict may come back; the dive must
    # have stopped on the clock, having explored a deep tree (the defect's premise).
    assert r.status == "node_limit" and r.stop_reason == "time", r
    assert r.nodes > 10_000, r
    assert wall < limit + 8.0, f"returned {wall:.1f} s after a {limit} s limit"


@pytest.mark.requires_highs
def test_highs_proves_the_probe_infeasible():
    """The oracle: HiGHS proves the same probe infeasible (so the Rust search above is
    undecided, not wrong)."""
    pytest.importorskip("highspy")
    from discopt._iis_backend import INFEASIBLE, make_backend

    A, rlo, rhi, lb, ub, integ = _load()
    b = make_backend("highs", A, rlo, rhi, lb, ub, integ)
    b.configure(rlo, rhi, lb, ub, integ)
    assert b.run("choose", time.perf_counter() + 60.0) == INFEASIBLE
