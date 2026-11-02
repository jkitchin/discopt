"""The per-node interval objective bound respects the solve deadline (#2071).

``_compute_interval_bound`` walks the objective DAG in Python. On QPLIB_5875 (QBN,
15,771 bilinear objective terms) one call takes ~1 s, and the batch bound loop in
``solve_model`` ran it for every node of a batch with no deadline check, so
``time_limit=90`` returned at 106-110 s (#2071).

Past the deadline the loop now stops: a skipped node keeps the bound it already
has (the interval bound is only ever ``max()``-combined with other valid bounds,
so omitting it is sound). The root iteration is exempt (#928 rule 1).

The slow evaluator is simulated: one enclosure in a non-root batch sleeps until the
solve's deadline has passed. Before the fix, calls kept starting after the deadline
for the rest of the batch.
"""

import sys
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
    """The deadline falls inside a batch's enclosure loop; the rest of the batch is skipped.

    #2071 follow-up: this test used to sleep 0.5 s in every enclosure started after
    ``time_limit - 1`` and require that one had started in that last second. That
    window is a race against batch granularity: enclosures run once per batch,
    between per-node relaxation solves, so on a slower runner the last batch could
    start its enclosures before the window and spend the rest of the budget on node
    solves. CI saw exactly that (8.1 s solves whose last enclosure started before
    7 s), and the probe failed without the guard ever being tested.

    Now the deadline is put inside the loop on purpose. The first non-root
    enclosure with a later node in its batch that would also get one -- read from
    ``solve_model``'s loop locals -- sleeps until just past the solve's own
    ``_deadline``. The very next step of that loop must then consult the guard on
    a spent deadline and start no further enclosure. The guard is wrapped to prove
    it fired (CLAUDE.md §6).
    """
    pytest.importorskip("pounce")
    # The batch interval-bound loop lives on the Python spatial tree.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    time_limit = 8.0
    orig_bound = S._compute_interval_bound
    orig_guard = S._interval_bound_past_deadline
    starts: list[float] = []
    state = {"deadline": None, "slept": False, "fired_after_sleep": 0, "armed_index": None}

    def guard(iteration, deadline):
        past = orig_guard(iteration, deadline)
        if state["slept"] and past:
            state["fired_after_sleep"] += 1
        return past

    def slow_bound(*a, **k):
        starts.append(time.perf_counter())
        loop = sys._getframe(1).f_locals
        if not state["slept"] and "n_batch" in loop:
            # Fail loudly if the loop's names change, rather than never arming.
            i, n_batch, iteration = loop["i"], loop["n_batch"], loop["iteration"]
            lbs = loop["result_lbs"]
            # A later node of this batch that WOULD get an enclosure: only then does
            # an unguarded loop start one past the deadline.
            pending = any(lbs[j] < S._SENTINEL_THRESHOLD for j in range(i + 1, n_batch))
            if iteration >= 1 and pending:
                state["deadline"] = float(loop["_deadline"])
                state["slept"] = True
                state["armed_index"] = len(starts) - 1
                time.sleep(max(0.0, state["deadline"] - time.perf_counter()) + 0.02)
        return orig_bound(*a, **k)

    monkeypatch.setattr(S, "_compute_interval_bound", slow_bound)
    monkeypatch.setattr(S, "_interval_bound_past_deadline", guard)
    m = _nonconvex_model()
    t0 = time.perf_counter()
    r = m.solve(time_limit=time_limit)
    wall = time.perf_counter() - t0

    assert r.algorithm_route.startswith("spatial-bb") and "Python tree" in r.algorithm_route, (
        r.algorithm_route
    )
    # The probe fired: the deadline was put inside a batch with nodes left, and the
    # guard was then consulted on it and said stop (CLAUDE.md §6).
    assert state["slept"], f"no non-root batch with enclosures left reached ({len(starts)} calls)"
    assert state["fired_after_sleep"] >= 1, "the guard was never consulted past the deadline"
    # No enclosure is started after the deadline. The armed call is the one that
    # carried the clock past it, so ANY enclosure after it started past the
    # deadline -- an exact count, with no timing slack for a fast enclosure to hide
    # in (an unguarded loop's next call starts ~0.02 s after the deadline).
    late = [s - state["deadline"] for s in starts[state["armed_index"] + 1 :]]
    assert not late, f"interval bound started {late} s past the deadline"
    # And the solve returns promptly once the deadline has passed.
    assert wall <= time_limit + 1.5, wall
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
