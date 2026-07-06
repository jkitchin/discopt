"""Tests for the pure-Rust MILP debugger hook and its node-box inspection.

Kept separate from ``test_debug_bnb.py`` so the Rust-fast-path surface can evolve
without churning the core debugger tests. Exercises the ``after_select``
checkpoint that marshals per-node boxes across the PyO3 boundary.
"""

from __future__ import annotations

import discopt as do
import pytest
from discopt import debug
from discopt.debug.context import DebugContext
from discopt.debug.engine import Control
from discopt.debug.session import DebugSession

# ── unit: DebugContext.from_rust batch marshaling ────────────────────────────


@pytest.mark.unit
def test_from_rust_populates_batch_boxes():
    ctx = DebugContext.from_rust(
        {
            "checkpoint": "after_select",
            "iteration": 2,
            "nodes": 5,
            "open_nodes": 2,
            "incumbent": None,
            "bound": 3.0,
            "gap": 0.5,
            "elapsed": 0.4,
            "batch_lb": [[0.0, 0.0], [0.0, 1.0]],
            "batch_ub": [[1.0, 1.0], [1.0, 1.0]],
            "batch_ids": [7, 8],
            "n_vars": 2,
        }
    )
    assert ctx.checkpoint is debug.Checkpoint.AFTER_SELECT
    assert ctx.n_batch == 2
    assert ctx.batch_lb.shape == (2, 2)
    assert int(ctx.batch_ids[1]) == 8
    assert ctx.steer is None  # no tree on the Rust path


@pytest.mark.unit
def test_from_rust_without_batch_is_aggregate_only():
    ctx = DebugContext.from_rust({"checkpoint": "iter_start", "nodes": 3})
    assert ctx.n_batch == 0
    assert ctx.batch_lb is None


# ── unit: print node <i> renders a Rust-path box ─────────────────────────────


@pytest.mark.unit
def test_print_node_on_rust_context():
    ctx = DebugContext.from_rust(
        {
            "checkpoint": "after_select",
            "batch_lb": [[0.0, 2.0]],
            "batch_ub": [[1.0, 5.0]],
            "batch_ids": [42],
            "n_vars": 2,
        }
    )
    sess = DebugSession(_NullFrontend())
    res = sess.engine.execute("print node 0", ctx, sess)
    assert "id=42" in res.output[0]
    assert res.data["lb"] == [0.0, 2.0]
    assert res.data["ub"] == [1.0, 5.0]


class _NullFrontend:
    def interact(self, ctx, session):
        return Control.CONTINUE


# ── integration: real pure-Rust MILP solve ──────────────────────────────────


def _knapsack():
    m = do.Model("dbg_rust_milp")
    xs = [m.integer(f"x{i}", lb=0, ub=1) for i in range(10)]
    vals = [8, 5, 3, 6, 4, 7, 9, 2, 5, 6]
    wts = [5, 3, 2, 4, 3, 5, 6, 1, 2, 4]
    m.subject_to(sum(w * xi for w, xi in zip(wts, xs)) <= 14)
    m.maximize(sum(v * xi for v, xi in zip(vals, xs)))
    return m


@pytest.mark.smoke
def test_rust_milp_after_select_exposes_node_boxes():
    captured: dict = {}

    class Inspect:
        def interact(self, ctx, session):
            if ctx.checkpoint.value == "after_select" and "box" not in captured:
                captured["n_batch"] = ctx.n_batch
                captured["box"] = session.engine.execute("print node 0", ctx, session)
            if ctx.checkpoint.value == "terminated":
                return Control.CONTINUE
            return Control.STEPI

    debug.attach(DebugSession(Inspect()))
    try:
        res = _knapsack().solve(time_limit=15.0, nlp_solver="simplex")
    finally:
        debug.detach()
    assert res.status == "optimal"
    assert captured["n_batch"] >= 1
    assert "id=" in captured["box"].output[0]
    assert captured["box"].data is not None  # real box data crossed PyO3


@pytest.mark.smoke
def test_rust_milp_after_select_bound_neutral():
    """The added after_select fire-site must not perturb the search."""
    base = _knapsack().solve(time_limit=15.0, nlp_solver="simplex")

    class Noop:
        def interact(self, ctx, session):
            return Control.CONTINUE

    debug.attach(DebugSession(Noop()))
    try:
        dbg = _knapsack().solve(time_limit=15.0, nlp_solver="simplex")
    finally:
        debug.detach()
    assert dbg.node_count == base.node_count
    assert dbg.objective == base.objective
