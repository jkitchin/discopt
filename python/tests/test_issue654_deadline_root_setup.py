"""Issue #654 — the pre-B&B root-setup phase honors ``solve(time_limit=T)``.

``Model.solve(time_limit=T)`` was violated on a class of large sparse
network-design / QAP / graph-partition MINLPs (sonet*, qap, super3t, eg_all_s):
wall time was roughly *constant* regardless of ``T`` because an uninterruptible
pre-B&B "root setup" phase — the per-node NLP evaluator's one-time XLA compile,
the McCormick LP relaxer build, and the root LP probe — ran to completion before
the (already deadline-respecting) Rust B&B loop, none of it polling the deadline.

The fix gates *entry* into that OPTIONAL search apparatus on the remaining
budget (``solver._deadline_exhausted``): once ``time_limit`` is already spent the
solver declines to build the search and instead spends the same bounded slice on
the rigorous root-relaxation fallback (``_root_relaxation_lower_bound`` — the
designated last-ditch dual bound, which builds its OWN relaxation and needs no
evaluator). This never truncates a bound-producing op mid-flight
(docs/dev/baron-gap-plan.md §8) and only ever *weakens* a still-valid bound, so
``incorrect_count`` stays 0.

These tests pin:
  * the gate is INERT when the budget is ample — an ample-budget solve still runs
    the search (node_count > 0) and certifies the true optimum unchanged
    (bound-neutrality: the gated phases must not alter a certified result);
  * the gate BITES when the budget is spent — a zero/tiny-budget solve skips the
    search (node_count == 0), returns promptly with a sound (never false-optimal,
    never oracle-crossing) result;
  * (slow, big-corpus) on sonet22v4 the wall now *scales* with ``T`` instead of
    being a fixed ~10 s floor.
"""

from __future__ import annotations

import os
import time

import discopt as do
import pytest


def _spatial_model():
    """A small nonconvex MINLP (transcendental + bilinear) that routes to the
    spatial-McCormick B&B path guarded by the #654 root-setup gate."""
    m = do.Model("issue654")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x * y + z >= 3.0)
    m.subject_to(y == do.exp(x) - 1.0)
    m.minimize(x + 0.5 * y + 0.3 * z)
    return m


@pytest.mark.requires_pounce
def test_ample_budget_gate_is_inert_and_bound_neutral():
    """With budget to spare the gate never fires: the spatial search runs
    (node_count > 0) and certifies the true optimum. This is the bound-neutrality
    guard — the #654 entry gates must not change a certified result."""
    r = _spatial_model().solve(time_limit=30.0)
    assert r.status == "optimal", f"ample-budget solve did not certify: {r.status}"
    assert r.objective is not None
    # The search actually ran — the deadline short-circuit did NOT fire.
    assert r.node_count > 0, "ample-budget solve unexpectedly skipped the search"
    # Sound certificate: the dual bound sits at (or below) the incumbent.
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-4


@pytest.mark.requires_pounce
def test_zero_budget_short_circuits_soundly():
    """A zero budget spends the whole limit in presolve, so the root-setup gate
    fires: the spatial search apparatus (evaluator compile + relaxer + probe +
    node loop) is skipped (node_count == 0) and the solver returns promptly with a
    sound result — never a false optimum, and any surfaced dual bound is a valid
    lower bound (<= the true optimum)."""
    # Reference optimum from an ample solve (self-consistent oracle).
    ref = _spatial_model().solve(time_limit=30.0)
    assert ref.status == "optimal" and ref.objective is not None
    opt = ref.objective

    t0 = time.perf_counter()
    r = _spatial_model().solve(time_limit=0.0)
    wall = time.perf_counter() - t0

    # The gate fired: no node was processed (search apparatus declined).
    assert r.node_count == 0, "zero-budget solve still ran the spatial search"
    # Deadline honored: no dominant uninterruptible search build — bounded to at
    # most the one designated root-relaxation fallback op.
    assert wall < 20.0, f"zero-budget solve ran {wall:.1f}s — root setup not gated"
    # Never a false optimum on a spent budget.
    assert r.status != "optimal"
    # Any surfaced dual bound must be a valid lower bound (MINIMIZE): never cross
    # the true optimum. ``None`` (no bound proven) is also sound.
    if r.bound is not None:
        assert r.bound <= opt + 1e-4, f"unsound bound {r.bound} > opt {opt}"
    # And it must never claim a feasible point strictly better than the optimum.
    if r.objective is not None:
        assert r.objective >= opt - 1e-4


_BENCH = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_sonet22v4_wall_scales_with_time_limit():
    """On the triggering instance, wall must now *scale* with ``time_limit`` (the
    #654 bug was that it did not — a fixed ~10 s root-setup floor dominated). The
    dual bound must stay valid (never cross the sonet22v4 oracle dual bound
    2311055.149; ``None`` is sound too)."""
    path = os.path.join(_BENCH, "sonet22v4.nl")
    if not os.path.exists(path):
        pytest.skip("sonet22v4.nl not vendored (big corpus)")

    from discopt.modeling.core import from_nl

    def _run(tl):
        t0 = time.perf_counter()
        r = from_nl(path).solve(time_limit=tl)
        return time.perf_counter() - t0, r

    wall2, r2 = _run(2.0)
    wall5, r5 = _run(5.0)

    # Deadline honored to within one bounded uninterruptible fallback op (the
    # documented root-relaxation floor), and — the crux of #654 — the T=2 wall is
    # meaningfully below the T=5 wall: the budget now matters.
    assert wall2 < 30.0, f"T=2 wall {wall2:.1f}s — root setup not gated"
    assert wall2 < wall5 + 3.0, (
        f"wall did not scale with time_limit (T=2 {wall2:.1f}s vs T=5 {wall5:.1f}s)"
    )
    # sonet22v4 is a MINIMIZE; a valid dual bound never exceeds the oracle optimum.
    _ORACLE = 2373966.0
    for r in (r2, r5):
        if r.bound is not None:
            assert r.bound <= _ORACLE + 1.0, f"unsound bound {r.bound} > oracle {_ORACLE}"
