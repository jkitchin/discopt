"""Conformance tests for the declared tightening schedule (Card 3a).

The point of ``discopt._jax.tightening_schedule`` is that the schedule stops
being "whatever order the blocks happen to sit in inside ``solve_model``" and
becomes a declaration that the code is checked against.  These tests are what
make it a declaration rather than a comment:

* :func:`test_declared_anchors_occur_in_declared_order` parses the AST of each
  host function and asserts the declared anchors appear in the declared order.
  A stage inserted out of order, renamed away, or deleted fails here.
* the recorder tests assert ``record``/``explain`` behave, including the loud
  refusal on an undeclared stage name.

Every test prints its executed-assertion count and the file-level check at the
bottom fails if the total is zero (CLAUDE.md §6 — a probe that traverses nothing
must not read as a pass).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from discopt._jax import tightening_schedule as ts

# Number of order/containment assertions actually executed across this module.
EXECUTED: dict[str, int] = {"order": 0, "anchor": 0, "recorder": 0}


def _host_function_ast(host: str) -> tuple[ast.FunctionDef, str]:
    """Return the ``ast.FunctionDef`` for a ``module:function`` host string."""
    module_name, func_name = host.split(":")
    import importlib

    module = importlib.import_module(module_name)
    path = Path(inspect.getfile(module))
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node, str(path)
    raise AssertionError(f"host function {host} not found (searched {path})")


def _called_names_with_lines(fn: ast.FunctionDef) -> list[tuple[str, int]]:
    """Every call/attribute-access name in ``fn``, with its line, in line order.

    Both plain calls (``foo(...)``) and method calls (``x.foo(...)``) count, and
    so do bare ``Name``/``attribute`` references, because two schedule anchors
    are structural markers (a local variable, and a helper reached indirectly)
    rather than direct calls.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                out.append((func.id, node.lineno))
            elif isinstance(func, ast.Attribute):
                out.append((func.attr, node.lineno))
        elif isinstance(node, ast.Name):
            out.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            out.append((node.attr, node.lineno))
    out.sort(key=lambda p: p[1])
    return out


@pytest.mark.smoke
@pytest.mark.parametrize("schedule", ts.ALL_SCHEDULES, ids=lambda s: s.name)
def test_declared_anchors_occur_in_declared_order(schedule):
    """Each schedule's anchors appear, in order, inside its host function.

    Matching is greedy and monotone: anchor *k* must occur at a line strictly
    after the line matched for anchor *k-1*.  That is what lets ``obbt_tighten_root``
    serve as both the root-OBBT anchor and the per-node-OBBT anchor in the same
    host function without the two schedules aliasing each other.
    """
    fn, path = _host_function_ast(schedule.host)
    names = _called_names_with_lines(fn)

    cursor = -1
    matched: list[tuple[str, int]] = []
    for stage in schedule:
        hit = next((ln for nm, ln in names if nm == stage.anchor and ln > cursor), None)
        if hit is None:
            present_anywhere = any(nm == stage.anchor for nm, _ in names)
            raise AssertionError(
                f"{schedule.name}: stage {stage.name!r} anchor {stage.anchor!r} "
                f"not found after line {cursor} in {schedule.host} ({path}). "
                f"anchor present elsewhere in the function: {present_anywhere}. "
                "Either the stage moved (fix the declaration in "
                "discopt/_jax/tightening_schedule.py) or the order regressed."
            )
        matched.append((stage.name, hit))
        cursor = hit
        EXECUTED["anchor"] += 1

    lines = [ln for _, ln in matched]
    assert lines == sorted(lines), f"{schedule.name}: matched lines not monotone: {matched}"
    EXECUTED["order"] += 1
    print(f"[schedule-order] {schedule.name}: {len(matched)} anchors matched in order {matched}")


@pytest.mark.smoke
def test_stage_names_are_unique_across_all_schedules():
    seen: set[str] = set()
    for sched in ts.ALL_SCHEDULES:
        for stage in sched:
            assert stage.name not in seen, f"duplicate stage name {stage.name!r}"
            seen.add(stage.name)
            EXECUTED["anchor"] += 1
    assert len(seen) == len(ts._KNOWN_STAGE_NAMES)


@pytest.mark.smoke
def test_every_stage_documents_a_gate_and_a_soundness_note():
    """A stage with no stated gate or no soundness note is a stage nobody can audit."""
    n = 0
    for sched in ts.ALL_SCHEDULES:
        for stage in sched:
            assert stage.gate.strip(), f"{stage.name}: empty gate"
            assert len(stage.soundness.strip()) > 30, f"{stage.name}: no soundness note"
            n += 1
    EXECUTED["anchor"] += n
    assert n >= 12, f"only {n} stages declared — the schedule lost coverage"


@pytest.mark.smoke
def test_record_rejects_an_undeclared_stage_loudly():
    ts.reset_run()
    with pytest.raises(KeyError, match="unknown tightening stage"):
        ts.record("no_such_stage")
    EXECUTED["recorder"] += 1


@pytest.mark.smoke
def test_record_accumulates_and_explain_reports_it():
    ts.reset_run()
    ts.record("root_fbbt", n_tightened=4, wall_s=0.25)
    ts.record("root_fbbt", n_tightened=3, wall_s=0.25)
    ts.declined("root_obbt", "known convex")

    runs = ts.current_runs()
    assert runs["root_fbbt"].calls == 2
    assert runs["root_fbbt"].n_tightened == 7
    assert runs["root_fbbt"].wall_s == pytest.approx(0.5)
    assert runs["root_fbbt"].ran is True
    assert runs["root_obbt"].ran is False
    assert "skipped: known convex" in runs["root_obbt"].gate_verdict
    EXECUTED["recorder"] += 6

    text = ts.ROOT.explain()
    assert "root_fbbt" in text
    assert "tightened=7" in text
    assert "skipped: known convex" in text
    # A stage nothing recorded must say so rather than invent a verdict.
    assert "not reached" in text
    EXECUTED["recorder"] += 4
    ts.reset_run()


@pytest.mark.smoke
def test_reset_run_is_per_thread():
    import threading

    ts.reset_run()
    ts.record("root_fbbt", n_tightened=1)
    seen: list[int] = []

    def worker():
        # A fresh thread must start with an empty record, not inherit ours.
        seen.append(len(ts.current_runs()))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == [0], f"thread-local leak: {seen}"
    assert len(ts.current_runs()) == 1
    EXECUTED["recorder"] += 2
    ts.reset_run()


@pytest.mark.smoke
def test_explain_all_covers_every_schedule():
    ts.reset_run()
    text = ts.explain("all")
    for sched in ts.ALL_SCHEDULES:
        assert sched.name in text
        EXECUTED["recorder"] += 1
    with pytest.raises(KeyError):
        ts.explain("nope")
    EXECUTED["recorder"] += 1


@pytest.mark.smoke
def test_probe_actually_executed_assertions():
    """CLAUDE.md §6: zero executed comparisons is a failure, not a pass.

    Ordered last in the file so the counters above have run; the assertion is on
    the anchor/order counters, which only a real AST traversal can raise.
    """
    total = sum(EXECUTED.values())
    print(f"[schedule-conformance] executed assertions: {EXECUTED} total={total}")
    assert EXECUTED["anchor"] > 0, "no anchor assertions executed — the AST walk found nothing"
    assert EXECUTED["order"] >= len(ts.ALL_SCHEDULES), (
        f"only {EXECUTED['order']} of {len(ts.ALL_SCHEDULES)} schedules order-checked"
    )
    assert total >= 30, f"suspiciously few executed assertions: {total}"


@pytest.mark.smoke
def test_a_real_solve_populates_the_schedule_record():
    """The recorder must fire on a real solve, not only in unit tests.

    CLAUDE.md §6: an instrument that is wired but never fires reads as a pass
    while measuring nothing. This solves a small nonconvex MINLP end-to-end and
    asserts that specific declared root stages carry a record afterwards, and
    that ``explain()`` renders them.
    """
    import discopt

    m = discopt.Model("schedule_live")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    m.subject_to(x * x + y <= 9.0)
    m.subject_to(x + y >= 1.5)
    m.minimize(-(x + 2 * y))

    res = m.solve(time_limit=20.0)
    assert res.status in ("optimal", "feasible"), res.status

    runs = ts.current_runs()
    assert runs, "no tightening stage recorded for a solve that ran a spatial tree"

    # Pre-dispatch stages: these run before any route is chosen, so they are
    # unconditional whichever loop the model ends up on.
    for required in ("declared_box_tightening", "rust_root_presolve"):
        assert required in runs, f"{required} never recorded; recorded: {sorted(runs)}"
        assert runs[required].ran, f"{required} recorded as not-run"
        EXECUTED["recorder"] += 2

    # At least one ROOT FBBT stage and at least one NODE stage must have fired,
    # whichever of the two B&B loops this model routed to. Asserting on the
    # union rather than on one route is what keeps this test a wiring check
    # instead of a routing check.
    root_fbbt_stages = {"root_fbbt", "nlp_bb_root_fbbt", "pre_factorable_fbbt"}
    node_stages = {s.name for s in ts.SPATIAL_NODE} | {s.name for s in ts.NLP_BB_NODE}
    assert root_fbbt_stages & set(runs), f"no root FBBT stage recorded: {sorted(runs)}"
    assert node_stages & set(runs), f"no node stage recorded: {sorted(runs)}"
    EXECUTED["recorder"] += 2

    text = ts.explain("all")
    assert "RAN" in text
    print("[schedule-live] route stages recorded: " + ", ".join(sorted(runs)))
    print(text)
    EXECUTED["recorder"] += 1
