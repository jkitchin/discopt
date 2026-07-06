"""Adversarial tests for the interactive B&B debugger (``discopt.debug``).

Attacks the debugger the way a hostile (or merely clumsy) user/agent would:

* command-engine fuzzing — garbage commands must never escape ``execute``;
* JSON-protocol fuzzing — malformed frames must never kill the stream;
* steering abuse — bad indices, hostile branch hints, maximize-sense inject;
* lifecycle abuse — ``quit`` before any incumbent exists must not corrupt the
  reported status (the certificate: never a false "infeasible"/"optimal");
* hostile frontends — a hook that raises on the Rust path is contained;
* active-stepping bound-neutrality — pausing at every checkpoint must leave
  the certified search bit-for-bit identical (CLAUDE.md §5).
"""

from __future__ import annotations

import time as _time

import discopt as do
import numpy as np
import pytest
from discopt import debug
from discopt.debug.context import DebugContext
from discopt.debug.engine import Control, DebugCommandEngine
from discopt.debug.session import DebugSession
from discopt.solver import _solve_milp_bb

# ── fakes ───────────────────────────────────────────────────────────────────


class FakeTree:
    def __init__(self, inc=None):
        self._inc = inc
        self.hint = None

    def stats(self):
        return {"total_nodes": 7, "open_nodes": 3, "global_lower_bound": 4.2, "gap": 0.12}

    def incumbent(self):
        return self._inc

    def inject_incumbent(self, sol, obj):
        if self._inc is None or obj < self._inc[1]:
            self._inc = (np.asarray(sol, dtype=float), float(obj))
            return True
        return False

    def set_branch_hints(self, ids, vhint):
        self.hint = (list(ids), list(vhint))


def _ok_validator(x):
    xv = np.asarray(x, dtype=np.float64).copy()
    return True, xv, float(np.sum(xv)) + 2.9


def _batch_ctx(tree, validator=None, n_nodes=2):
    return DebugContext.build(
        debug.Checkpoint.BEFORE_IMPORT,
        tree=tree,
        iteration=0,
        elapsed=0.1,
        batch_lb=np.zeros((n_nodes, 2)),
        batch_ub=np.ones((n_nodes, 2)),
        batch_ids=np.arange(11, 11 + n_nodes),
        result_lbs=np.linspace(3.5, 3.6, n_nodes),
        result_sols=np.full((n_nodes, 2), 0.5),
        result_feas=np.ones(n_nodes, dtype=bool),
        validator=validator,
    )


class QuitFrontend:
    def interact(self, ctx, session):
        return Control.QUIT


class WalkFrontend:
    """Pauses at every checkpoint (stepi) — maximally invasive inspection."""

    def __init__(self):
        self.hits = []

    def interact(self, ctx, session):
        self.hits.append(ctx.checkpoint.value)
        # Exercise read-only commands at every pause, then keep stepping.
        session.engine.execute("info", ctx, session)
        session.engine.execute("print nodes", ctx, session)
        if ctx.checkpoint.value == "terminated":
            return Control.CONTINUE
        return Control.STEPI


# ── command-engine fuzzing (unit) ────────────────────────────────────────────

GARBAGE_COMMANDS = [
    "",
    "   ",
    "bogus",
    "béak 3",
    "\x00\x01\x02",
    "break if",
    "break if garbage",
    "break if gap<<3",
    "break if gap<abc",
    "break if nodes>1e999 && gap<-1e999",
    "break if gap==0.1||nodes>2",
    "break on",
    "break on not_an_event",
    "break del",
    "break del xyz",
    "tbreak",
    "tbreak abc",
    "run",
    "run -5",
    "run notanumber",
    "stop-at nonsense",
    "print",
    "print node",
    "print node 99",
    "print node -99",
    "print relax 99",
    "print wat",
    "inject",
    "inject 99",
    "inject abc",
    "hint",
    "hint 1",
    "hint a b",
    "watch " + "x" * 10_000,
    "b " * 2000,
    "info extra args ignored",
]


@pytest.mark.unit
def test_engine_never_raises_on_garbage():
    """Every garbage command yields a CommandResult; nothing escapes execute."""
    sess = DebugSession(QuitFrontend())
    for validator in (None, _ok_validator):
        eng = DebugCommandEngine()
        ctx = _batch_ctx(FakeTree(), validator=validator)
        for cmd in GARBAGE_COMMANDS:
            res = eng.execute(cmd, ctx, sess)
            assert res is not None, cmd
            assert res.control in (Control.NONE, Control.CONTINUE, Control.QUIT), cmd
    # No-batch context (ITER_START): batch-dependent commands must degrade.
    eng = DebugCommandEngine()
    ctx0 = DebugContext.build(debug.Checkpoint.ITER_START, tree=FakeTree(), iteration=0)
    for cmd in ("print nodes", "print node 0", "print relax 0", "inject 0", "hint 1 1"):
        res = eng.execute(cmd, ctx0, sess)
        assert res.output, cmd


@pytest.mark.unit
def test_inject_out_of_range_is_error_and_tree_untouched():
    tree = FakeTree()
    eng = DebugCommandEngine()
    sess = DebugSession(QuitFrontend())
    ctx = _batch_ctx(tree, validator=_ok_validator, n_nodes=2)
    res = eng.execute("inject 99", ctx, sess)
    assert any("error" in line for line in res.output)
    assert tree.incumbent() is None


@pytest.mark.unit
def test_inject_negative_index_does_not_crash():
    """numpy would resolve -1 to the last node; document that it stays safe
    (validated like any candidate) and does not crash."""
    tree = FakeTree()
    eng = DebugCommandEngine()
    sess = DebugSession(QuitFrontend())
    ctx = _batch_ctx(tree, validator=_ok_validator, n_nodes=2)
    res = eng.execute("inject -1", ctx, sess)
    assert res.output  # some outcome reported, no exception
    inc = tree.incumbent()
    if inc is not None:  # if adopted, it must carry the validated objective
        assert inc[1] == pytest.approx(3.9)


# ── JSON protocol fuzzing (unit) ─────────────────────────────────────────────

MALFORMED_FRAMES = [
    "@@@@",
    "123",
    "[1, 2, 3]",
    '{"cmd": 5}',
    '{"nope": true}',
    '{"cmd": null, "id": 9}',
    "{invalid json",
    '"unterminated',
    "NaN",
    '{"cmd": "print", "args": {"k": "v"}}',
    '{"cmd": "break if gap<0.2 || nodes>1"}',
    " ",
]


@pytest.mark.unit
def test_json_frontend_survives_malformed_stream():
    from discopt.debug.jsonproto import JsonFrontend

    frames = MALFORMED_FRAMES + ["continue"]
    it = iter(frames)

    def read():
        try:
            return next(it)
        except StopIteration:
            return None

    events = []
    fe = JsonFrontend(read_fn=read, write_fn=events.append)
    sess = DebugSession(fe)
    ctx = _batch_ctx(FakeTree())
    control = fe.interact(ctx, sess)
    assert control is Control.CONTINUE
    # Every frame produced exactly one result/malformed event; stream stayed up.
    results = [e for e in events if e["event"] == "result"]
    assert len(results) == len([f for f in frames if f.strip()])
    # And the whole stream stayed strict-JSON serializable.
    import json as _json

    for e in events:
        _json.loads(_json.dumps(e, allow_nan=False))


@pytest.mark.unit
def test_json_string_args_do_not_crash():
    """args as a bare string (agent bug) must not crash the frontend."""
    from discopt.debug.jsonproto import _parse_command

    cmd, _ = _parse_command('{"cmd": "print", "args": "bound"}')
    assert isinstance(cmd, str)  # mangled is tolerable; crashing is not


# ── lifecycle abuse: quit-early status validity (e2e) ────────────────────────


def _spatial_model():
    m = do.Model("adv_dbg")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x * y + z >= 3.0)
    m.subject_to(y == do.exp(x) - 1.0)
    m.minimize(x + 0.5 * y + 0.3 * z)
    return m


def _spatial_model_max():
    """Same feasible set, maximize sense — probes internal sign handling."""
    m = do.Model("adv_dbg_max")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x * y + z >= 3.0)
    m.subject_to(y == do.exp(x) - 1.0)
    m.maximize(-(x + 0.5 * y + 0.3 * z))
    return m


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_quit_before_any_incumbent_is_not_false_infeasible():
    """CERTIFICATE: quitting at the very first checkpoint (no incumbent, tree
    not exhausted, no limits hit) proves nothing about the model. The status
    must not be "infeasible" (worst-class false verdict on a feasible model)
    and must not be "optimal" (nothing was certified)."""
    debug.attach(DebugSession(QuitFrontend()))
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status not in ("infeasible", "optimal"), (
        f"quit-before-incumbent produced a false '{res.status}' verdict"
    )


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_quit_early_keeps_bound_below_incumbent():
    """Quit as soon as an incumbent exists: the partial result must still obey
    bound <= objective (min sense) — the certificate invariant."""

    class QuitOnIncumbent:
        def interact(self, ctx, session):
            if ctx.incumbent_obj is not None:
                return Control.QUIT
            return Control.STEPI

    debug.attach(DebugSession(QuitOnIncumbent()))
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status != "optimal" or res.gap is not None
    if res.objective is not None and res.bound is not None:
        assert res.bound <= res.objective + 1e-6, "UNSOUND: bound crossed incumbent"


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_session_reuse_after_quit_is_inert_but_sound():
    """Reusing a quit session leaves it detached: the second solve never
    pauses, and — critically — is not stopped early by the stale session."""
    sess = DebugSession(QuitFrontend())
    debug.attach(sess)
    try:
        _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()

    baseline = _spatial_model().solve(time_limit=20.0)
    debug.attach(sess)  # stale, already-quit session
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == baseline.status == "optimal"
    assert res.objective == baseline.objective


# ── active stepping must be bound-neutral (e2e) ──────────────────────────────


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_step_through_every_checkpoint_is_bound_neutral():
    """CLAUDE.md §5: pausing at EVERY checkpoint and running read-only
    inspection commands at each pause must leave node_count and the certified
    objective exactly unchanged."""
    base = _spatial_model().solve(time_limit=20.0)

    fe = WalkFrontend()
    debug.attach(DebugSession(fe))
    try:
        dbg = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()

    assert len(fe.hits) > 4, "walk frontend never engaged"
    assert dbg.node_count == base.node_count, "node count drifted under stepping"
    assert dbg.objective == base.objective, "objective drifted under stepping"


# ── steering abuse on a real solve (e2e) ─────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_garbage_hints_cannot_corrupt_certificate():
    """Hostile branch hints (bogus node ids, out-of-range/negative variable
    indices) may at most reorder the search; the certified optimum must be
    unchanged and the commands must not crash the REPL."""
    base = _spatial_model().solve(time_limit=20.0)

    script = [
        "stop-at steer",
        "continue",
        "hint 999999 999999",
        "hint 0 -5",
        "hint -1 0",
        "continue",
    ]
    debug.attach(debug.make_session(script=script))
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_inject_on_maximize_model_keeps_certificate():
    """Sign-flip probe: `inject` on a MAXIMIZE model must validate in the
    internal min sense; the certified optimum must match the baseline."""
    base = _spatial_model_max().solve(time_limit=20.0)
    assert base.status == "optimal"

    script = ["stop-at steer", "continue", "inject 0", "inject 1", "continue"]
    debug.attach(debug.make_session(script=script))
    try:
        res = _spatial_model_max().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_repl_survives_garbage_mid_solve():
    """A REPL script full of garbage must not derail the solve."""
    base = _spatial_model().solve(time_limit=20.0)
    script = list(GARBAGE_COMMANDS[:20]) + ["continue"]
    debug.attach(debug.make_session(script=script))
    try:
        res = _spatial_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


# ── hostile frontends (e2e) ──────────────────────────────────────────────────


def _pure_milp_model():
    m = do.Model("adv_milp")
    xs = [m.integer(f"x{i}", lb=0, ub=1) for i in range(10)]
    vals = [8, 5, 3, 6, 4, 7, 9, 2, 5, 6]
    wts = [5, 3, 2, 4, 3, 5, 6, 1, 2, 4]
    m.subject_to(sum(w * xi for w, xi in zip(wts, xs)) <= 14)
    m.maximize(sum(v * xi for v, xi in zip(vals, xs)))
    return m


@pytest.mark.smoke
def test_raising_hook_on_rust_path_is_contained(capsys):
    """A frontend that raises at every checkpoint on the pure-Rust MILP path
    must be contained (printed + Continue): the solve completes and certifies
    the same optimum as the baseline."""
    base = _pure_milp_model().solve(time_limit=15.0, nlp_solver="simplex")
    assert base.status == "optimal"

    class BombFrontend:
        def interact(self, ctx, session):
            raise RuntimeError("hostile frontend")

    debug.attach(DebugSession(BombFrontend()))
    try:
        res = _pure_milp_model().solve(time_limit=15.0, nlp_solver="simplex")
    finally:
        debug.detach()
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_raising_frontend_on_python_path_detaches_cleanly():
    """On the Python-driven path a raising frontend propagates (loud failure,
    no silent swallowing) — but it must not poison the NEXT solve: the guard
    in Model.solve() detaches, and a follow-up solve certifies normally."""

    class BombFrontend:
        def interact(self, ctx, session):
            raise RuntimeError("hostile frontend")

    with pytest.raises(RuntimeError, match="hostile frontend"):
        _spatial_model().solve(time_limit=20.0, debug=DebugSession(BombFrontend()))
    assert not debug.is_attached(), "solve() failed to detach after the crash"

    res = _spatial_model().solve(time_limit=20.0)
    assert res.status == "optimal"


# ═════════════════════════════════════════════════════════════════════════════
# Sweep 2 — orthogonal axes: trace invariants at every checkpoint, coverage of
# ALL four instrumented loops (spatial / NLP-BB / MILP-BB / MIQP-BB) plus the
# Rust fast-path, stateful chaos fuzzing during live solves, and odd model
# classes (continuous-only, infeasible, maximize).
# ═════════════════════════════════════════════════════════════════════════════


class TraceFrontend:
    """Walks every checkpoint (stepi) recording (cp, bound, incumbent, iter)."""

    def __init__(self):
        self.trace = []

    def interact(self, ctx, session):
        self.trace.append((ctx.checkpoint.value, ctx.best_bound, ctx.incumbent_obj, ctx.iteration))
        if ctx.checkpoint.value == "terminated":
            return Control.CONTINUE
        return Control.STEPI


def _assert_trace_invariants(trace):
    """Certificate + lifecycle invariants that must hold at EVERY checkpoint,
    not just at the end of the solve (internal min sense throughout):
    bound never crosses the incumbent, the dual bound never regresses, the
    incumbent never worsens, iterations are monotone, TERMINATED fires exactly
    once and last, INCUMBENT_FOUND always carries an incumbent."""
    assert trace, "no checkpoints fired"
    cps = [t[0] for t in trace]
    assert cps[-1] == "terminated", f"trace did not end at terminated: {cps[-5:]}"
    assert cps.count("terminated") == 1
    prev_bound = -np.inf
    prev_inc = np.inf
    prev_iter = 0
    for cp, bound, inc, it in trace:
        if inc is not None and np.isfinite(bound):
            assert bound <= inc + 1e-6 * max(1.0, abs(inc)), (
                f"UNSOUND at {cp}: bound {bound} crossed incumbent {inc}"
            )
        if np.isfinite(bound):
            assert bound >= prev_bound - 1e-9, f"dual bound regressed at {cp}"
            prev_bound = max(prev_bound, bound)
        if inc is not None:
            assert inc <= prev_inc + 1e-9, f"incumbent worsened at {cp}"
            prev_inc = min(prev_inc, inc)
        if cp == "incumbent_found":
            assert inc is not None, "incumbent_found fired without an incumbent"
        assert it >= prev_iter or cp == "terminated"
        prev_iter = max(prev_iter, it)


def _traced_solve(model, **solve_kwargs):
    fe = TraceFrontend()
    debug.attach(DebugSession(fe))
    try:
        res = model.solve(time_limit=20.0, **solve_kwargs)
    finally:
        debug.detach()
    return res, fe.trace


# ── models routing to each instrumented loop ─────────────────────────────────


def _convex_minlp():
    """Convex MINLP for the NLP-BB loop (loop 2, via nlp_bb=True). The
    (z - 1.5)^2 term makes the root NLP relaxation fractional in z, so the
    loop must branch (a root-integral model would fathom immediately and
    never reach the BEFORE_IMPORT steer point)."""
    m = do.Model("adv2_nlpbb")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x + y + z <= 4.0)
    m.minimize((x - 1.5) ** 2 + (y - 2.5) ** 2 + (z - 1.5) ** 2)
    return m


def _nested_heuristic_minlp():
    """Convex MINLP whose NLP-BB run launches a nested solve_model heuristic
    (restricted sub-MINLP): the regression probe for outermost-solve
    suppression. Root-integral in z, so the outer loop is short and the
    nested solve dominates the trace if suppression is broken."""
    m = do.Model("adv2_nested")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x + y + z <= 4.0)
    m.minimize((x - 1.5) ** 2 + (y - 2.5) ** 2 + 0.5 * z)
    return m


def _miqp_model():
    """Convex MIQP routing to the MIQP-BB loop (loop 4)."""
    m = do.Model("adv2_miqp")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    z = m.integer("z", lb=0, ub=4)
    m.subject_to(x + y + z >= 2.0)
    m.minimize(x * x + y * y + 0.3 * z)
    return m


def _run_milp_bb(model):
    """Drive the fallback-only Python MILP-BB loop (loop 3) directly."""
    return _solve_milp_bb(model, 20.0, 1e-6, 8, "best_first", 100_000, _time.perf_counter())


def _milp_model():
    """Small MILP for driving the Python MILP-BB loop (loop 3) directly."""
    m = do.Model("adv2_milp")
    xs = [m.integer(f"x{i}", lb=0, ub=1) for i in range(8)]
    vals = [7, 5, 4, 6, 3, 5, 8, 2]
    wts = [4, 3, 3, 4, 2, 3, 5, 1]
    m.subject_to(sum(w * xi for w, xi in zip(wts, xs)) <= 11)
    m.maximize(sum(v * xi for v, xi in zip(vals, xs)))
    return m


def _infeasible_bilinear():
    """Infeasible only after branching (FBBT alone cannot prove it): needs
    x*y >= 0.5 with x + y <= 1 on [0,1]^2 (max of x*y is 0.25)."""
    m = do.Model("adv2_infeas")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x * y >= 0.5)
    m.subject_to(x + y <= 1.0)
    m.minimize(x + y)
    return m


# ── trace invariants across every loop ───────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_trace_invariants_spatial_loop():
    res, trace = _traced_solve(_spatial_model())
    assert res.status == "optimal"
    _assert_trace_invariants(trace)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_trace_invariants_spatial_maximize():
    res, trace = _traced_solve(_spatial_model_max())
    assert res.status == "optimal"
    _assert_trace_invariants(trace)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_trace_invariants_nlp_bb_loop():
    res, trace = _traced_solve(_convex_minlp(), nlp_bb=True)
    assert res.status == "optimal"
    _assert_trace_invariants(trace)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_trace_invariants_miqp_loop():
    res, trace = _traced_solve(_miqp_model())
    assert res.status == "optimal"
    _assert_trace_invariants(trace)


@pytest.mark.smoke
def test_trace_invariants_rust_fast_path():
    res, trace = _traced_solve(_pure_milp_model(), nlp_solver="simplex")
    assert res.status == "optimal"
    _assert_trace_invariants(trace)


@pytest.mark.smoke
def test_trace_invariants_milp_bb_loop_direct():
    """Loop 3 (_solve_milp_bb) is fallback-only, so drive it directly."""
    fe = TraceFrontend()
    debug.attach(DebugSession(fe))
    try:
        res = _run_milp_bb(_milp_model())
    finally:
        debug.detach()
    assert res.status == "optimal"
    _assert_trace_invariants(fe.trace)


# ── per-loop: no-op neutrality, quit statuses, inject wiring ─────────────────


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_nlp_bb_no_op_debugger_is_bound_neutral():
    base = _convex_minlp().solve(time_limit=20.0, nlp_bb=True)
    debug.attach(DebugSession(WalkFrontend()))
    try:
        dbg = _convex_minlp().solve(time_limit=20.0, nlp_bb=True)
    finally:
        debug.detach()
    assert dbg.node_count == base.node_count
    assert dbg.objective == base.objective


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_miqp_no_op_debugger_is_bound_neutral():
    base = _miqp_model().solve(time_limit=20.0)
    debug.attach(DebugSession(WalkFrontend()))
    try:
        dbg = _miqp_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert dbg.node_count == base.node_count
    assert dbg.objective == base.objective


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_quit_early_not_false_infeasible_nlp_bb():
    debug.attach(DebugSession(QuitFrontend()))
    try:
        res = _convex_minlp().solve(time_limit=20.0, nlp_bb=True)
    finally:
        debug.detach()
    assert res.status not in ("infeasible", "optimal"), (
        f"NLP-BB quit produced a false '{res.status}'"
    )


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_quit_early_not_false_infeasible_miqp():
    debug.attach(DebugSession(QuitFrontend()))
    try:
        res = _miqp_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status not in ("infeasible", "optimal"), (
        f"MIQP-BB quit produced a false '{res.status}'"
    )


@pytest.mark.smoke
def test_quit_early_not_false_infeasible_milp_bb_direct():
    debug.attach(DebugSession(QuitFrontend()))
    try:
        res = _run_milp_bb(_milp_model())
    finally:
        debug.detach()
    assert res.status not in ("infeasible", "optimal"), (
        f"MILP-BB quit produced a false '{res.status}'"
    )


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_inject_on_nlp_bb_keeps_certificate(capsys):
    """Loop 2 wires a validator: inject validates and cannot corrupt."""
    base = _convex_minlp().solve(time_limit=20.0, nlp_bb=True)
    script = ["stop-at steer", "continue", "inject 0", "continue"]
    debug.attach(debug.make_session(script=script))
    try:
        res = _convex_minlp().solve(time_limit=20.0, nlp_bb=True)
    finally:
        debug.detach()
    err = capsys.readouterr().err
    assert "inject node[0]" in err
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_inject_refused_on_miqp_loop(capsys):
    """Loop 4 wires NO validator: inject must refuse loudly, not trust."""
    base = _miqp_model().solve(time_limit=20.0)
    script = ["stop-at steer", "continue", "inject 0", "continue"]
    debug.attach(debug.make_session(script=script))
    try:
        res = _miqp_model().solve(time_limit=20.0)
    finally:
        debug.detach()
    err = capsys.readouterr().err
    assert "no candidate validator" in err, "MIQP inject did not refuse"
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


# ── chaos: deterministic command storm during a live solve ──────────────────


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_chaos_command_storm_keeps_certificate():
    """Rotate through the full command surface (inspection, breakpoints,
    watches, steering, garbage) at every checkpoint of a live solve. The
    certificate must survive and every command must return."""
    base = _spatial_model().solve(time_limit=20.0)

    commands = [
        "info",
        "print incumbent",
        "print bound",
        "print gap",
        "print nodes",
        "print node 0",
        "print relax 0",
        "break if gap<0.3",
        "tbreak 2",
        "watch bound",
        "hint 0 0",
        "inject 0",
        "break",
        "print node -1",
        "break del 2",
        "stop-at process",
        "help",
        "notacommand",
        "break if x||y",
    ]

    class ChaosFrontend:
        def __init__(self):
            self.n = 0
            self.executed = 0

        def interact(self, ctx, session):
            for _ in range(3):
                cmd = commands[self.n % len(commands)]
                self.n += 1
                res = session.engine.execute(cmd, ctx, session)
                assert res is not None, cmd
                self.executed += 1
            if ctx.checkpoint.value == "terminated":
                return Control.CONTINUE
            return Control.STEPI

    fe = ChaosFrontend()
    debug.attach(DebugSession(fe))
    try:
        res = _spatial_model().solve(time_limit=30.0)
    finally:
        debug.detach()
    assert fe.executed >= 15, "chaos frontend barely engaged"
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)
    if res.bound is not None and res.objective is not None:
        assert res.bound <= res.objective + 1e-6


# ── odd model classes ────────────────────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_debugger_silent_on_continuous_path():
    """A continuous NLP routes around all instrumented loops: an attached
    debugger must simply never fire (documented gap), not crash the solve."""
    m = do.Model("adv2_cont")
    x = m.continuous("x", lb=0.0, ub=3.0)
    m.minimize((x - 1.0) ** 2)
    fe = TraceFrontend()
    debug.attach(DebugSession(fe))
    try:
        res = m.solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res.status == "optimal"
    assert fe.trace == [], "continuous path unexpectedly fired checkpoints"


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_infeasible_model_traced_and_on_error():
    """Certified infeasibility under full stepping: trace invariants hold,
    the verdict survives the debugger, and on-error pauses with the reason."""
    res, trace = _traced_solve(_infeasible_bilinear())
    assert res.status in ("infeasible", "unknown")
    if trace:
        _assert_trace_invariants(trace)

    fe = TraceFrontend()
    debug.attach(DebugSession(fe, enter_on_error=True))
    try:
        res2 = _infeasible_bilinear().solve(time_limit=20.0)
    finally:
        debug.detach()
    assert res2.status == res.status
    if trace:  # the loop ran, so TERMINATED fired with a non-optimal status
        assert [t[0] for t in fe.trace] == ["terminated"]


@pytest.mark.smoke
@pytest.mark.requires_pounce
def test_nested_heuristic_solves_are_suppressed():
    """The NLP-BB path launches nested solve_model heuristics (restricted
    sub-MINLPs). Only the OUTERMOST solve may fire checkpoints: nested solves
    interleaving into the session would collide breakpoints, break trace
    monotonicity, and let `quit` kill a heuristic instead of the solve.
    Regression: this model produced TWO `terminated` events before the
    outermost-solve guard."""
    res, trace = _traced_solve(_nested_heuristic_minlp(), nlp_bb=True)
    assert res.status == "optimal"
    cps = [t[0] for t in trace]
    assert cps.count("terminated") == 1, f"nested solve leaked checkpoints: {cps}"
    _assert_trace_invariants(trace)
