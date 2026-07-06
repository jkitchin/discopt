"""Tests for the interactive branch-and-bound debugger (``discopt.debug``).

Covers three things:
  * pure-logic units (command engine, condition parsing, safe-steer) with a
    fake tree, no solver — marked ``unit``;
  * end-to-end checkpoint firing on a real spatial-McCormick solve;
  * the correctness gate: a no-op debugger is **bound-neutral** (identical
    ``node_count`` and objective vs. no debugger), per CLAUDE.md §5.
"""

from __future__ import annotations

import discopt as do
import numpy as np
import pytest
from discopt import debug
from discopt.debug.context import DebugContext
from discopt.debug.engine import Control, DebugCommandEngine
from discopt.debug.session import DebugSession

# ── fakes ───────────────────────────────────────────────────────────────────


class FakeTree:
    """Minimal stand-in for PyTreeManager used by the unit tests."""

    def __init__(self, inc=None, stats=None):
        self._inc = inc
        self._stats = stats or {
            "total_nodes": 7,
            "open_nodes": 3,
            "global_lower_bound": 4.2,
            "gap": 0.12,
        }
        self.hint = None

    def stats(self):
        return self._stats

    def incumbent(self):
        return self._inc

    def inject_incumbent(self, sol, obj):
        if self._inc is None or obj < self._inc[1]:
            self._inc = (np.asarray(sol, dtype=float), float(obj))
            return True
        return False

    def set_branch_hints(self, ids, vhint):
        self.hint = (list(ids), list(vhint))


class RecordingFrontend:
    """Records every checkpoint it is asked to interact at, then resumes."""

    def __init__(self, control=Control.CONTINUE, walk=False):
        self.hits: list[str] = []
        self._control = control
        self._walk = walk

    def interact(self, ctx, session):
        self.hits.append(ctx.checkpoint.value)
        if self._walk and ctx.checkpoint.value != "terminated":
            return Control.STEPI
        return self._control


def _true_obj(x):
    """The 'true objective' the fake validator evaluates: sum + 2.9."""
    return float(np.sum(x)) + 2.9


def _fake_validator(feasible=True):
    """Validator double matching discopt.debug.steer.Validator: the candidate
    is 'feasible' iff requested, and its true objective is _true_obj(x) — which
    deliberately differs from the relaxation bound in result_lbs."""

    def validate(x):
        xv = np.asarray(x, dtype=np.float64).copy()
        if not feasible:
            return False, xv, float("nan")
        return True, xv, _true_obj(xv)

    return validate


def _batch_ctx(tree, checkpoint, validator=None):
    return DebugContext.build(
        checkpoint,
        tree=tree,
        iteration=0,
        elapsed=0.1,
        batch_lb=np.array([[0.0, 0.0]]),
        batch_ub=np.array([[1.0, 1.0]]),
        batch_ids=np.array([11]),
        result_lbs=np.array([3.5]),
        result_sols=np.array([[0.5, 0.5]]),
        result_feas=np.array([True]),
        validator=validator,
    )


# ── unit: command engine ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_conditional_breakpoint_fires_on_metric():
    eng = DebugCommandEngine()
    tree = FakeTree()
    sess = DebugSession(RecordingFrontend())
    ctx = DebugContext.build(debug.Checkpoint.ITER_START, tree=tree, iteration=0)
    eng.execute("break if gap<0.2", ctx, sess)
    assert eng.hit_reason(ctx) == "condition: gap<0.2"
    # A condition that does not hold does not fire.
    eng2 = DebugCommandEngine()
    eng2.execute("break if gap<0.05", ctx, sess)
    assert eng2.hit_reason(ctx) is None


@pytest.mark.unit
def test_compound_condition_and_rejects_or():
    eng = DebugCommandEngine()
    sess = DebugSession(RecordingFrontend())
    ctx = DebugContext.build(
        debug.Checkpoint.ITER_START, tree=FakeTree(inc=(np.zeros(2), 5.0)), iteration=12
    )
    eng.execute("break if iter>10 && gap<0.2", ctx, sess)
    assert eng.hit_reason(ctx) is not None
    res = eng.execute("break if gap<0.2 || iter>3", ctx, sess)
    assert any("||" in line for line in res.output)


@pytest.mark.unit
def test_iteration_breakpoint_and_tbreak_one_shot():
    eng = DebugCommandEngine()
    sess = DebugSession(RecordingFrontend())
    ctx5 = DebugContext.build(debug.Checkpoint.ITER_START, tree=FakeTree(), iteration=5)
    eng.execute("tbreak 5", ctx5, sess)
    assert eng.hit_reason(ctx5) == "iteration 5"
    # one-shot: does not fire a second time
    assert eng.hit_reason(ctx5) is None


# ── unit: safe steer ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_inject_adopts_validated_true_objective_not_relax_bound():
    """An adopted incumbent carries the validator's TRUE objective, never the
    relaxation bound in result_lbs (which would fabricate an incumbent)."""
    tree = FakeTree(inc=(np.array([1.0, 2.0]), 5.05))
    ctx = _batch_ctx(tree, debug.Checkpoint.BEFORE_IMPORT, validator=_fake_validator())
    adopted, obj, reason = ctx.steer.inject(ctx.result_sols[0])
    assert adopted is True
    assert reason == "adopted"
    # true objective 3.9 = _true_obj([0.5, 0.5]); relax bound is 3.5.
    assert obj == pytest.approx(3.9)
    assert tree.incumbent()[1] == pytest.approx(3.9)
    assert tree.incumbent()[1] != pytest.approx(float(ctx.result_lbs[0]))


@pytest.mark.unit
def test_inject_rejects_non_improving_candidate():
    tree = FakeTree(inc=(np.array([1.0, 2.0]), 1.0))
    ctx = _batch_ctx(tree, debug.Checkpoint.BEFORE_IMPORT, validator=_fake_validator())
    adopted, obj, reason = ctx.steer.inject(ctx.result_sols[0])
    assert adopted is False
    assert obj == pytest.approx(3.9)
    assert "not strictly improving" in reason
    assert tree.incumbent()[1] == pytest.approx(1.0)


@pytest.mark.unit
def test_inject_rejects_infeasible_candidate():
    """A candidate the validator rules infeasible never reaches the tree."""
    tree = FakeTree(inc=(np.array([1.0, 2.0]), 5.05))
    validator = _fake_validator(feasible=False)
    ctx = _batch_ctx(tree, debug.Checkpoint.BEFORE_IMPORT, validator=validator)
    adopted, obj, reason = ctx.steer.inject(ctx.result_sols[0])
    assert adopted is False
    assert obj is None
    assert "infeasible" in reason
    assert tree.incumbent()[1] == pytest.approx(5.05)  # untouched


@pytest.mark.unit
def test_inject_refuses_without_validator():
    """No validator wired => inject refuses loudly rather than trusting the
    caller (certificate safety, CLAUDE.md §1)."""
    tree = FakeTree()
    ctx = _batch_ctx(tree, debug.Checkpoint.BEFORE_IMPORT)  # validator=None
    assert ctx.steer.can_inject is False
    with pytest.raises(RuntimeError, match="no candidate validator"):
        ctx.steer.inject(ctx.result_sols[0])
    # The engine command reports it as unavailable instead of raising.
    eng = DebugCommandEngine()
    sess = DebugSession(RecordingFrontend())
    res = eng.execute("inject 0", ctx, sess)
    assert any("inject unavailable" in line for line in res.output)
    assert tree.incumbent() is None  # nothing was injected


@pytest.mark.unit
def test_inject_refuses_nonfinite():
    tree = FakeTree()
    ctx = _batch_ctx(tree, debug.Checkpoint.BEFORE_IMPORT, validator=_fake_validator())
    with pytest.raises(ValueError):
        ctx.steer.inject(np.array([np.nan, 0.0]))


@pytest.mark.unit
def test_hint_records_reordering():
    tree = FakeTree()
    ctx = _batch_ctx(tree, debug.Checkpoint.BEFORE_IMPORT)
    ctx.steer.hint([11], [1])
    assert tree.hint == ([11], [1])


# ── unit: fire hot-path ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_fire_is_noop_when_detached():
    debug.detach()
    assert debug.fire(debug.Checkpoint.ITER_START, tree=FakeTree()) is False
    assert not debug.is_attached()


@pytest.mark.unit
def test_quit_requests_stop():
    class QuitFrontend:
        def interact(self, ctx, session):
            return Control.QUIT

    sess = DebugSession(QuitFrontend())
    debug.attach(sess)
    try:
        stop = debug.fire(debug.Checkpoint.ITER_START, tree=FakeTree(), iteration=0)
        assert stop is True
        assert sess.stop_requested
    finally:
        debug.detach()


# ── end-to-end on a real spatial-McCormick solve ─────────────────────────────


def _spatial_model():
    """A nonconvex MINLP (transcendental + bilinear) that routes to the
    spatial-McCormick B&B loop instrumented by the debugger."""
    m = do.Model("dbg_e2e")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x * y + z >= 3.0)
    m.subject_to(y == do.exp(x) - 1.0)
    m.minimize(x + 0.5 * y + 0.3 * z)
    return m


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_checkpoints_fire_in_lifecycle_order():
    fe = RecordingFrontend(walk=True)
    debug.attach(DebugSession(fe))
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == "optimal"
    seen = set(fe.hits)
    # every wired checkpoint is reached
    for cp in (
        "iter_start",
        "after_select",
        "before_import",
        "after_process",
        "terminated",
    ):
        assert cp in seen, f"{cp} never fired; saw {seen}"
    # order within an iteration is select -> import -> process
    i_sel = fe.hits.index("after_select")
    assert fe.hits.index("before_import") > i_sel
    assert fe.hits.index("after_process") > fe.hits.index("before_import")


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_no_op_debugger_is_bound_neutral():
    """CLAUDE.md §5: an attached no-op debugger must not perturb the search."""
    base = _spatial_model().solve(time_limit=20.0)

    debug.attach(DebugSession(RecordingFrontend(control=Control.CONTINUE)))
    try:
        dbg = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()

    assert dbg.node_count == base.node_count
    assert dbg.objective == base.objective  # exact, not approx


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_inject_cannot_corrupt_certificate(capsys):
    """Regression for the unvalidated-inject hole: steering ``inject`` at the
    BEFORE_IMPORT checkpoint must never change the certified optimum. The
    candidate (a relaxation solution) is validated against the original
    problem; an infeasible point is rejected outright, and a feasible one is
    injected with its true evaluated objective — so the certificate is
    identical either way. Before the fix, the relaxation *bound* was adopted
    as the incumbent objective and the solve certified a wrong optimum."""
    base = _spatial_model().solve(time_limit=20.0)
    assert base.status == "optimal"

    script = ["stop-at steer", "continue", "inject 0", "continue"]
    debug.attach(debug.make_session(script=script))
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()

    err = capsys.readouterr().err
    assert "inject node[0]" in err, "the inject command never ran"
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_scripted_repl_end_to_end(capsys):
    """Drive the human REPL with a scripted command stream (no TTY)."""
    script = [
        "info",
        "break if nodes>=2",
        "continue",
        "print bound",
        "continue",
    ]
    sess = debug.make_session(script=script)
    debug.attach(sess)
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == "optimal"
    err = capsys.readouterr().err
    assert "paused at" in err
    assert "discopt-dbg" in err


# ── JSON agent protocol ──────────────────────────────────────────────────────


def _drive_json(model, commands):
    """Run ``model`` under the JSON frontend fed ``commands``; return events."""
    import json as _json

    from discopt.debug.jsonproto import JsonFrontend

    it = iter(commands)

    def read():
        try:
            c = next(it)
        except StopIteration:
            return None
        return c if isinstance(c, str) else _json.dumps(c)

    events: list[dict] = []
    fe = JsonFrontend(read_fn=read, write_fn=events.append)
    debug.attach(DebugSession(fe))
    try:
        res = model.solve(time_limit=20.0)
    finally:
        debug.detach()
    return res, events


@pytest.mark.unit
def test_json_command_parsing_forms():
    from discopt.debug.jsonproto import _parse_command

    assert _parse_command("continue") == ("continue", None)
    assert _parse_command('"continue"') == ("continue", None)
    assert _parse_command('{"cmd": "print", "args": ["node", "0"], "id": 7}') == (
        "print node 0",
        7,
    )
    assert _parse_command('{"cmd": "break if gap<0.2", "id": 8}') == (
        "break if gap<0.2",
        8,
    )
    cmd, rid = _parse_command('{"nope": 1, "id": 3}')
    assert cmd is None and rid == 3


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_json_protocol_handshake_and_events():
    import json as _json

    res, events = _drive_json(
        _spatial_model(),
        [
            {"cmd": "info", "id": 1},
            {"cmd": "break", "args": ["if", "nodes>=2"], "id": 2},
            "continue",
            {"cmd": "print", "args": ["bound"], "id": 3},
            "continue",
            "continue",
        ],
    )
    assert res.status == "optimal"

    # hello handshake is first and self-describing.
    hello = events[0]
    assert hello["event"] == "hello"
    assert hello["protocol"] == "discopt-dbg/1"
    assert "iter_start" in hello["checkpoints"]
    assert "new_incumbent" in hello["events"]
    assert hello["capabilities"]["mutate_iterate"] is False
    assert hello["capabilities"]["safe_steer"] is True

    kinds = [e["event"] for e in events]
    assert "pause" in kinds
    assert "terminated" in kinds
    # request_id is echoed on results.
    results = {e["request_id"]: e for e in events if e["event"] == "result"}
    assert results[1]["command"] == "info"
    assert results[3]["output"] == ["bound = 1.04943"]
    # the whole stream is strict-JSON clean (no Infinity/NaN).
    for e in events:
        _json.loads(_json.dumps(e, allow_nan=False))


# ── pure-Rust MILP fast-path (PyO3 checkpoint hook) ──────────────────────────


def _pure_milp_model():
    """A pure binary knapsack that routes to the Rust ``solve_milp`` fast-path
    (``nlp_solver='simplex'``) — the loop that runs entirely in Rust."""
    m = do.Model("dbg_milp")
    xs = [m.integer(f"x{i}", lb=0, ub=1) for i in range(10)]
    vals = [8, 5, 3, 6, 4, 7, 9, 2, 5, 6]
    wts = [5, 3, 2, 4, 3, 5, 6, 1, 2, 4]
    m.subject_to(sum(w * xi for w, xi in zip(wts, xs)) <= 14)
    m.maximize(sum(v * xi for v, xi in zip(vals, xs)))
    return m


@pytest.mark.smoke
def test_rust_milp_hook_fires_across_pyo3():
    fe = RecordingFrontend(walk=True)
    debug.attach(DebugSession(fe))
    try:
        res = _pure_milp_model().solve(time_limit=15.0, nlp_solver="simplex")
    finally:
        debug.detach()
    assert res.status == "optimal"
    seen = set(fe.hits)
    # the Rust hook fires the aggregate checkpoints (no per-node batch on this path)
    assert "iter_start" in seen
    assert "after_process" in seen
    assert "terminated" in seen


@pytest.mark.smoke
def test_rust_milp_hook_is_bound_neutral():
    """A no-op debugger must not perturb the pure-Rust MILP search."""
    base = _pure_milp_model().solve(time_limit=15.0, nlp_solver="simplex")

    debug.attach(DebugSession(RecordingFrontend(control=Control.CONTINUE)))
    try:
        dbg = _pure_milp_model().solve(time_limit=15.0, nlp_solver="simplex")
    finally:
        debug.detach()

    assert dbg.node_count == base.node_count
    assert dbg.objective == base.objective


@pytest.mark.smoke
def test_rust_milp_quit_stops_early():
    class QuitFrontend:
        def interact(self, ctx, session):
            return Control.QUIT

    debug.attach(DebugSession(QuitFrontend()))
    try:
        res = _pure_milp_model().solve(time_limit=15.0, nlp_solver="simplex")
    finally:
        debug.detach()
    # Aborted before proving optimality -> a valid, non-"optimal" partial result.
    assert res.status in ("feasible", "node_limit", "time_limit")
