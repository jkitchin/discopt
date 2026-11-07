"""The MILP tree honours ``time_limit`` per node, not per batch (highs-parity D0).

``_solve_milp_bb`` checked the deadline once per loop iteration, and an iteration solves a
whole batch of node LPs. On the #1667 unit-commitment model the Rust route defers to this
tree, whose node LPs take ~1 s each, and the solve returned at 141-147 s on a 120 s limit
and at up to 88 s on 60 s (``docs/dev/highs-parity-plan.md`` P0.7). Now the deadline is
checked before every node after the first, and the nodes of the batch not reached are
requeued with their pop-time bound (open, not pruned).

The test slows each node LP by a fixed sleep so the timing does not depend on the host,
and checks both the contract (returns within one node of the limit) and soundness (the
returned bound stays below the true optimum, which the unslowed solve certifies).
"""

from __future__ import annotations

import sys
import time

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np

_SLEEP = 0.4
_LIMIT = 3.0
_BATCH = 16


def _knapsack():
    rng = np.random.default_rng(7)
    n, m = 30, 4
    w = rng.integers(5, 40, size=(m, n)).astype(float)
    v = rng.integers(10, 60, size=n).astype(float)
    cap = 0.35 * w.sum(axis=1)
    mdl = dm.Model("mk")
    x = mdl.binary("x", shape=(n,))
    for i in range(m):
        mdl.subject_to(dm.sum(lambda j: w[i, j] * x[j], over=range(n)) <= float(cap[i]))
    mdl.maximize(dm.sum(lambda j: v[j] * x[j], over=range(n)))
    return mdl


def test_tree_stops_within_one_node_of_the_limit(monkeypatch):
    opt = _knapsack().solve(time_limit=60, milp_backend="native")
    assert opt.status == "optimal" and opt.gap_certified

    real = solver_mod._solve_node_lp_simplex
    starts: list[float] = []
    t0 = [0.0]

    def slow(*a, **k):
        # Slow only the tree's own node solves: the root phase (cut loop, dive)
        # already polls the deadline, and slowing it leaves no time for a batch.
        if k.get("is_root") or sys._getframe(1).f_code.co_name != "_solve_milp_bb":
            return real(*a, **k)
        starts.append(time.perf_counter() - t0[0])
        time.sleep(_SLEEP)
        return real(*a, **k)

    monkeypatch.setattr(solver_mod, "_solve_node_lp_simplex", slow)
    t0[0] = time.perf_counter()
    res = _knapsack().solve(time_limit=_LIMIT, milp_backend="native", batch_size=_BATCH)
    wall = time.perf_counter() - t0[0]
    assert len(starts) >= 3, "the slowed tree never ran a batch: the probe measured nothing"
    assert res.status != "optimal"
    # The contract: no tree node starts after the deadline. Measured before the fix on
    # this model: three node solves started at 3.22, 3.47 and 3.72 s on a 3 s limit
    # (0.25 s per node), wall 4.0 s.
    late = [s for s in starts if s > _LIMIT]
    assert not late, f"tree node solves started after the {_LIMIT} s limit at {late}"
    assert wall < _LIMIT + _SLEEP + 0.5, wall
    # Requeued nodes keep their bound: the reported bound is still a valid one.
    if res.bound is not None:
        assert res.bound >= opt.objective - 1e-6 * max(1.0, abs(opt.objective))
