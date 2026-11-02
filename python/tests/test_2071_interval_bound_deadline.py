"""The per-node interval objective bound respects the solve deadline (#2071).

``_compute_interval_bound`` walks the objective DAG in Python. On QPLIB_5875 (QBN,
15,771 bilinear objective terms) one call takes ~1 s, and the batch bound loop in
``solve_model`` ran it for every node of a batch with no deadline check, so
``time_limit=90`` returned at 106-110 s (#2071).

Past the deadline the loop now stops: a skipped node keeps the bound it already
has (the interval bound is only ever ``max()``-combined with other valid bounds,
so omitting it is sound). The root iteration is exempt (#928 rule 1).

The slow evaluator is simulated, and it is triggered by *work done*, not by the
wall clock (#2103): the first enclosure of a later non-root iteration computes the
real bound and then sleeps until past the solver's own deadline (read from the argument
of ``_interval_bound_past_deadline``) has passed. So the deadline always falls
inside an interval-bound call of a non-root batch whenever the solve reaches that
iteration within the limit (measured ~3.5 s of 8 s on an idle 4-core container,
~5.7 s with every core saturated by competing load), not only when a call happens
to start in a late wall-clock window. An earlier
version slept only once ``elapsed >= time_limit - 1``; on a slow runner no call
started that late and the precondition failed without the property being tested.
Before the fix, calls kept starting after the deadline for the rest of the batch.
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


# The first enclosure of this iteration (or the first later one that computes an
# enclosure) straddles the deadline. Iteration 0 is the root, which is exempt; by
# iteration 2 the batch holds several nodes, so nodes remain after the slow one.
_TRIGGER_ITERATION = 2
_OVERSHOOT_S = 0.25


def test_interval_bound_not_started_past_deadline(monkeypatch):
    pytest.importorskip("pounce")
    # The batch interval-bound loop lives on the Python spatial tree.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    time_limit = 8.0
    orig_bound = S._compute_interval_bound
    orig_past = S._interval_bound_past_deadline
    # Every enclosure call site asks the predicate first, so its last arguments
    # name the iteration and the solver's deadline for the call that follows.
    state = {"iteration": None, "deadline": None, "slowed": None}
    checks = []  # (iteration, returned) for every predicate call
    starts = []  # (perf_counter at call start, iteration)

    def recording_past(iteration, deadline):
        state["iteration"], state["deadline"] = iteration, deadline
        out = orig_past(iteration, deadline)
        checks.append((iteration, out))
        return out

    def slow_bound(*a, **k):
        now = time.perf_counter()
        it = state["iteration"]
        starts.append((now, it))
        bound = orig_bound(*a, **k)
        if state["slowed"] is None and it is not None and it >= _TRIGGER_ITERATION:
            deadline = state["deadline"]
            # Overshoot well past the 0.05 s grace below, so that without the fix
            # the next node's start is unambiguously late.
            time.sleep(max(0.0, deadline + _OVERSHOOT_S - time.perf_counter()))
            state["slowed"] = (now, time.perf_counter(), it, deadline)
        return bound

    monkeypatch.setattr(S, "_interval_bound_past_deadline", recording_past)
    monkeypatch.setattr(S, "_compute_interval_bound", slow_bound)
    m = _nonconvex_model()
    t0 = time.perf_counter()
    r = m.solve(time_limit=time_limit)
    wall = time.perf_counter() - t0

    assert r.algorithm_route.startswith("spatial-bb") and "Python tree" in r.algorithm_route, (
        r.algorithm_route
    )
    # The probe fired (CLAUDE.md §6): a non-root enclosure started before the
    # deadline and returned after it, so the deadline fell inside the loop.
    assert state["slowed"] is not None, (
        f"no enclosure at iteration >= {_TRIGGER_ITERATION} before the deadline",
        sorted({it for it, _ in checks}),
        wall,
    )
    slow_start, slow_end, slow_it, deadline = state["slowed"]
    assert slow_it >= 1 and slow_start < deadline < slow_end, state["slowed"]
    # And the deadline check then fired in that same iteration (nodes remained
    # after the slow one), rather than the batch happening to end there.
    fired = [it for it, out in checks if out]
    assert slow_it in fired, (slow_it, fired, checks[-10:])
    # No enclosure is started after the limit. 0.05 s covers the gap between the
    # check and the call's first statement.
    late = [(s - deadline, it) for s, it in starts if s > deadline + 0.05]
    assert not late, f"interval bound started past the deadline: {late}"
    # And the solve returns promptly once the slow call ends.
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
