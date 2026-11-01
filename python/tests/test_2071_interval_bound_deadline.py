"""The per-node interval objective bound respects the solve deadline (#2071).

``_compute_interval_bound`` walks the objective DAG in Python. On QPLIB_5875 (QBN,
15,771 bilinear objective terms) one call takes ~1 s, and the batch bound loop in
``solve_model`` ran it for every node of a batch with no deadline check, so
``time_limit=90`` returned at 106-110 s (#2071).

Past the deadline the loop now stops: a skipped node keeps the bound it already
has (the interval bound is only ever ``max()``-combined with other valid bounds,
so omitting it is sound). The root iteration is exempt (#928 rule 1).

The slow evaluator is simulated: the real bound is computed, then the call sleeps,
but only once the solve is close to its limit, so the tree reaches full batches
first. Before the fix, calls kept starting after the deadline for the rest of the
batch.
"""

import time

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest


def _nonconvex_model(n=60):
    # >= _POUNCE_BATCH_MIN_VARS variables, so the batch path runs.
    m = dm.Model("t2071")
    x = m.continuous("x", shape=(n,), lb=-2.0, ub=2.0)
    m.minimize(sum(x[i] * x[i + 1] for i in range(n - 1)) + sum(dm.sin(3 * x[i]) for i in range(n)))
    return m


def test_interval_bound_not_started_past_deadline(monkeypatch):
    pytest.importorskip("pounce")
    # The batch interval-bound loop lives on the Python spatial tree.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    time_limit = 8.0
    slow_after = time_limit - 1.0
    slow_s = 0.5
    orig = S._compute_interval_bound
    starts = []
    t0 = [None]

    def slow_bound(*a, **k):
        now = time.perf_counter()
        starts.append(now - t0[0])
        bound = orig(*a, **k)
        if now - t0[0] >= slow_after:
            time.sleep(slow_s)
        return bound

    monkeypatch.setattr(S, "_compute_interval_bound", slow_bound)
    m = _nonconvex_model()
    t0[0] = time.perf_counter()
    r = m.solve(time_limit=time_limit)
    wall = time.perf_counter() - t0[0]

    assert r.algorithm_route.startswith("spatial-bb") and "Python tree" in r.algorithm_route, (
        r.algorithm_route
    )
    # The probe fired, and fired in the slow window (CLAUDE.md §6).
    assert len(starts) >= 3, starts
    assert any(s >= slow_after for s in starts), starts
    # No enclosure is started after the limit. 0.05 s covers the gap between the
    # check and the call's first statement.
    late = [s for s in starts if s > time_limit + 0.05]
    assert not late, f"interval bound started past the deadline: {late}"
    # And the solve returns within one slow call of the limit.
    assert wall <= time_limit + slow_s + 1.0, wall
    assert r.status in ("feasible", "time_limit", "optimal")
    assert r.objective is not None and np.isfinite(r.objective)


def test_past_deadline_predicate():
    now = time.perf_counter()
    # Root iteration: never skipped, even on a spent deadline (#928 rule 1).
    assert not S._interval_bound_past_deadline(0, now - 10.0)
    assert S._interval_bound_past_deadline(1, now - 10.0)
    assert not S._interval_bound_past_deadline(5, now + 10.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
