"""Conformance tests for the declared solve-path routing table (Card 4a).

The point of :mod:`discopt.routing` is that ``solve_model``'s dispatch tree
stops being "whatever order the ``if`` statements happen to sit in" and becomes
a declaration the code is checked against.  These tests are what make it a
declaration rather than a comment:

* :func:`test_route_markers_occur_in_declared_order` parses ``solve_model``'s
  AST and asserts the ``_rt.entered`` / ``_rt.fell_through`` markers appear
  exactly once each, in exactly the declared order.  A gate inserted, deleted,
  renamed or moved fails here.
* :func:`test_declared_handler_is_called_in_its_route_region` asserts each
  route's declared handler is really invoked inside that route's span, so the
  declaration cannot drift into fiction.
* :func:`test_callback_fallthrough_guards_are_still_present` is the soundness
  test: the three #740/#748 fall-throughs exist because the specialized engine
  cannot honour ``lazy_constraints`` / ``incumbent_callback``, and turning one
  back into an early ``return`` would delete its guard.  That is a false-optimal
  waiting to happen, so it is pinned at source level.

Every test contributes to an executed-assertion count and the file-level check
at the bottom fails if the total is zero (CLAUDE.md §6 — a probe that traverses
nothing must not read as a pass).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from discopt import routing as rt

EXECUTED: dict[str, int] = {"order": 0, "marker": 0, "handler": 0, "guard": 0, "recorder": 0}

# The routes whose fall-through exists purely because the specialized engine
# cannot honour a user callback.  Losing one is a soundness regression (#740 /
# #748), so the set itself is pinned here rather than derived.
CALLBACK_FALLTHROUGHS = ("class_milp", "class_miqp", "nlp_bb_auto")


def _solve_model_ast() -> tuple[ast.FunctionDef, str, str]:
    """``solve_model``'s AST node, the file it came from, and that file's text."""
    import discopt.solver as solver_mod

    path = Path(inspect.getfile(solver_mod))
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve_model":
            return node, str(path), text
    raise AssertionError(f"solve_model not found in {path}")


def _markers(fn: ast.FunctionDef) -> list[tuple[int, str, str]]:
    """Every routing marker call in ``fn`` as ``(lineno, route_name, kind)``.

    A marker is a call to ``entered`` / ``fell_through`` (however the module is
    aliased at the call site) whose first argument is a string literal.  Both
    ``_rt.entered("x")`` and a bare ``entered("x")`` are recognised, so an import
    style change cannot make this probe silently traverse nothing.
    """
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in ("entered", "fell_through"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        value = node.args[0].value
        if isinstance(value, str):
            out.append((node.lineno, value, name))
    out.sort(key=lambda t: t[0])
    return out


def _handler_call_lines(fn: ast.FunctionDef, handler: str) -> list[int]:
    lines = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            nm = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if nm == handler:
                lines.append(node.lineno)
    return sorted(lines)


def _first_marker_lines(fn: ast.FunctionDef) -> dict[str, int]:
    """Route name -> line of its FIRST marker (a route may also record a decline)."""
    out: dict[str, int] = {}
    for ln, name, _kind in _markers(fn):
        out.setdefault(name, ln)
    return out


@pytest.mark.smoke
def test_route_markers_occur_in_declared_order():
    """The markers inside ``solve_model`` are exactly ROUTE_TABLE, in order.

    This is the whole point of the declaration: the physical dispatch order and
    the declared order are the same object, and a divergence is a test failure
    rather than a comment that quietly went stale.  A route may carry more than
    one marker (the branch-level fall-throughs record their decline as well as
    their entry), so ordering is asserted on each route's *first* marker.
    """
    fn, path, _ = _solve_model_ast()
    marks = _markers(fn)
    seen: list[str] = []
    for _, name, _kind in marks:
        if name not in seen:
            seen.append(name)
    found = seen
    assert found == list(rt.ROUTE_NAMES), (
        "routing markers in solve_model do not match discopt.routing.ROUTE_TABLE.\n"
        f"  in {path}\n"
        f"  source order:   {found}\n"
        f"  declared order: {list(rt.ROUTE_NAMES)}\n"
        "Either a gate moved/was added/was removed (update ROUTE_TABLE) or the "
        "dispatch order regressed."
    )
    EXECUTED["order"] += 1
    EXECUTED["marker"] += len(found)
    print(f"[routing-order] {len(found)} markers matched in declared order in {path}")


@pytest.mark.smoke
def test_every_declared_route_is_marked_and_every_marker_is_declared():
    fn, _, _ = _solve_model_ast()
    marks = _markers(fn)
    counts: dict[str, int] = {}
    for _, name, _kind in marks:
        counts[name] = counts.get(name, 0) + 1
    for route in rt.ROUTE_TABLE:
        assert counts.get(route.name, 0) >= 1, (
            f"route {route.name!r} has no marker in solve_model; the declaration "
            "would report it as 'not reached' forever"
        )
        EXECUTED["marker"] += 1
    for name in counts:
        assert name in rt.ROUTE_NAMES, f"undeclared route marker {name!r} in solve_model"
        EXECUTED["marker"] += 1
    # Exactly one *entry* (``entered``) marker per route: two would make the walk
    # ambiguous about which gate actually dispatched.
    entry_counts: dict[str, int] = {}
    for _, name, kind in marks:
        if kind == "entered":
            entry_counts[name] = entry_counts.get(name, 0) + 1
    for route in rt.ROUTE_TABLE:
        assert entry_counts.get(route.name, 0) == 1, (
            f"route {route.name!r} has {entry_counts.get(route.name, 0)} entered() "
            "markers; exactly one is required"
        )
        EXECUTED["marker"] += 1


@pytest.mark.smoke
def test_marker_kind_matches_the_declaration():
    """A ``fell_through`` marker may only appear on a route declared fallthrough."""
    fn, _, _ = _solve_model_ast()
    for _, name, kind in _markers(fn):
        route = next(r for r in rt.ROUTE_TABLE if r.name == name)
        if kind == "fell_through":
            assert route.fallthrough, (
                f"{name}: source calls fell_through() but the declaration says fallthrough=False"
            )
        EXECUTED["marker"] += 1


@pytest.mark.smoke
def test_branch_level_callback_fallthroughs_record_their_decline():
    """The two branch-level #748 fall-throughs must record a runtime decline.

    ``class_milp`` and ``class_miqp`` reach their engine's dispatch point and
    then decline; without a ``fell_through`` marker the walk would show them as
    "gate false", which is a lie about why the spatial loop ran.  (``nlp_bb_auto``
    declines inside its gate expression, so it has nothing to mark at runtime —
    its guard is pinned by the source-level test below.)
    """
    fn, _, _ = _solve_model_ast()
    kinds: dict[str, set[str]] = {}
    for _, name, kind in _markers(fn):
        kinds.setdefault(name, set()).add(kind)
    for name in ("class_milp", "class_miqp"):
        assert "fell_through" in kinds.get(name, set()), (
            f"route {name!r} has no fell_through() marker — either the #748 "
            "fall-through was turned back into a return (a soundness regression) "
            "or the marker was dropped"
        )
        EXECUTED["guard"] += 1


@pytest.mark.smoke
def test_declared_handler_is_called_in_its_route_region():
    """Each route's declared handler is really invoked inside that route's span.

    A route's region runs from its own marker to the next route's marker (or,
    for ``handler_precedes_marker`` routes, from the previous marker to its
    own).  A declaration naming a handler the code never calls there is a
    declaration that has drifted into fiction.
    """
    fn, path, _ = _solve_model_ast()
    by_name = _first_marker_lines(fn)
    order = list(rt.ROUTE_NAMES)

    checked = 0
    for i, name in enumerate(order):
        route = next(r for r in rt.ROUTE_TABLE if r.name == name)
        if not route.handler:
            continue
        here = by_name[name]
        if route.handler_precedes_marker:
            lo = by_name[order[i - 1]] if i > 0 else fn.lineno
            hi = here
        else:
            lo = here
            hi = by_name[order[i + 1]] if i + 1 < len(order) else fn.end_lineno
        hits = [ln for ln in _handler_call_lines(fn, route.handler) if lo <= ln <= hi]
        assert hits, (
            f"route {name!r} declares handler {route.handler!r} but no call to it "
            f"appears in lines {lo}-{hi} of {path}. Either the route now dispatches "
            "somewhere else (fix ROUTE_TABLE) or the handler was removed."
        )
        checked += 1
        EXECUTED["handler"] += 1
    assert checked >= 15, f"only {checked} handler checks executed; the probe is degrading"
    print(f"[routing-handler] {checked} declared handlers located in their own region")


@pytest.mark.smoke
def test_callback_fallthrough_guards_are_still_present():
    """The #740/#748 soundness fall-throughs still exist in the source.

    ``_solve_milp_bb`` / ``_solve_miqp_bb`` / ``_solve_nlp_bb`` neither receive
    nor consult ``lazy_constraints`` / ``incumbent_callback``.  For a lazy
    constraint the callback *defines the feasible set*, so an engine that never
    consults it would accept a point outside the feasible set and certify it —
    a false optimal.  Each of these three routes therefore matches its gate and
    declines.  This test fails the moment one of them is turned back into an
    early ``return``.
    """
    fn, path, text = _solve_model_ast()
    by_name = _first_marker_lines(fn)

    # Parent map so a marker can be resolved to the ``if`` that encloses it.
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _enclosing_if_source(lineno: int) -> tuple[str, int, int]:
        target = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and node.lineno == lineno:
                target = node
                break
        assert target is not None, f"no marker call found at line {lineno}"
        cur: ast.AST | None = target
        while cur is not None:
            cur = parents.get(cur)
            if isinstance(cur, ast.If):
                seg = ast.get_source_segment(text, cur)
                assert seg, "could not recover the enclosing if source"
                return seg, cur.lineno, cur.end_lineno or cur.lineno
        raise AssertionError(f"marker at line {lineno} has no enclosing if statement")

    for name in CALLBACK_FALLTHROUGHS:
        route = next(r for r in rt.ROUTE_TABLE if r.name == name)
        assert route.fallthrough, f"{name} must be declared fallthrough=True"
        assert route.fallthrough_guard, f"{name} must declare its guard source text"
        region, lo, hi = _enclosing_if_source(by_name[name])
        assert route.fallthrough_guard in region, (
            f"the #740/#748 callback guard for route {name!r} is GONE from "
            f"lines {lo}-{hi} of {path}.\n"
            f"  expected source text: {route.fallthrough_guard!r}\n"
            f"  why it matters: {route.fallthrough_reason}\n"
            "This is a soundness guard, not a style choice — do not delete it to "
            "make this test pass."
        )
        EXECUTED["guard"] += 1

    # And the declaration must not quietly shrink the set of guarded routes.
    guarded = {r.name for r in rt.ROUTE_TABLE if r.fallthrough_guard}
    assert set(CALLBACK_FALLTHROUGHS) <= guarded, (
        f"a callback fall-through lost its declared guard: {set(CALLBACK_FALLTHROUGHS) - guarded}"
    )
    EXECUTED["guard"] += 1
    print(f"[routing-guard] {len(CALLBACK_FALLTHROUGHS)} callback fall-through guards present")


@pytest.mark.smoke
def test_every_route_documents_a_gate_and_a_reason():
    n = 0
    for route in rt.ROUTE_TABLE:
        assert route.gate.strip(), f"{route.name}: empty gate"
        assert len(route.reason.strip()) > 40, f"{route.name}: reason too thin to audit"
        if route.fallthrough:
            assert route.fallthrough_reason.strip(), (
                f"{route.name}: declared fallthrough with no stated reason"
            )
        n += 1
    assert n == len(rt.ROUTE_TABLE)
    EXECUTED["marker"] += n


@pytest.mark.smoke
def test_exactly_one_terminal_route_and_it_is_last():
    terminals = [r for r in rt.ROUTE_TABLE if r.terminal]
    assert len(terminals) == 1
    assert terminals[0] is rt.ROUTE_TABLE[-1]
    assert terminals[0].name == "spatial_branch_and_bound"
    EXECUTED["marker"] += 3


# --------------------------------------------------------------------------
# Recorder behaviour
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_recorder_rejects_an_undeclared_route():
    rt.reset_run()
    with pytest.raises(KeyError):
        rt.entered("no_such_route")
    EXECUTED["recorder"] += 1


@pytest.mark.smoke
def test_recorder_and_explain_round_trip():
    rt.reset_run()
    assert rt.current_runs() == {}
    rt.fell_through("class_milp", "lazy_constraints present")
    rt.entered("spatial_branch_and_bound")
    runs = rt.current_runs()
    assert runs["class_milp"].verdict == "FELL THROUGH"
    assert runs["class_milp"].detail == "lazy_constraints present"
    assert runs["spatial_branch_and_bound"].verdict == "ENTERED"
    text = rt.explain(with_schedule=False)
    assert "class_milp" in text
    assert "FELL THROUGH" in text
    assert "-> " in text  # the dispatched route is marked
    EXECUTED["recorder"] += 5
    rt.reset_run()


@pytest.mark.smoke
def test_explain_on_an_unrun_table_says_so_rather_than_inventing_a_walk():
    rt.reset_run()
    text = rt.explain(with_schedule=False)
    assert "never run" in text
    assert text.count("gate false") == len(rt.ROUTE_TABLE)
    EXECUTED["recorder"] += 2


@pytest.mark.smoke
def test_nesting_does_not_clobber_the_outer_walk():
    """The substitution-presolve route re-enters ``solve_model``; the walk must survive."""
    rt.reset_run()
    rt.enter_solve()
    rt.entered("substitution_presolve")
    rt.enter_solve()  # nested solve
    rt.entered("class_lp")
    rt.exit_solve()
    rt.exit_solve()
    runs = rt.current_runs()
    assert "substitution_presolve" in runs and "class_lp" in runs
    EXECUTED["recorder"] += 2
    rt.reset_run()


@pytest.mark.smoke
def test_explain_composes_the_tightening_schedule():
    """Card 4a's explain_routing must compose Card 3a's schedule.explain()."""
    rt.reset_run()
    text = rt.explain()
    assert "routing walk" in text
    assert "root  [discopt.solver:solve_model]" in text, (
        "explain() no longer composes discopt._jax.tightening_schedule.explain()"
    )
    EXECUTED["recorder"] += 2


@pytest.mark.smoke
def test_executed_assertion_count_is_nonzero():
    """CLAUDE.md §6: a probe that traversed nothing must not read as a pass."""
    total = sum(EXECUTED.values())
    print(f"[routing] executed assertions: {EXECUTED} total={total}")
    assert EXECUTED["order"] > 0, "the order probe never ran"
    assert EXECUTED["handler"] >= 15, "the handler probe traversed nothing"
    assert EXECUTED["guard"] >= 3, "the callback-fall-through guard probe traversed nothing"
    assert total > 40, f"executed-assertion total {total} is implausibly low"
