"""#2049: node lineage (#2043) is visible from the REPL and the JSON protocol.

#2043 put each node's parent, depth, creating branch and status on ``DebugContext``,
but the shared command engine never read them, so ``debug=True`` and
``debug="json"`` could not answer "why does this node exist?". These tests drive
both frontends through real solves on the Rust MILP hook and the Python tree and
check the tree they print, plus the engine's rendering on hand-built contexts.
"""

from __future__ import annotations

import io
import json

import discopt.modeling as dm
import pytest
from discopt.debug.context import DebugContext, flat_variable_names
from discopt.debug.engine import Control
from discopt.debug.jsonproto import JsonFrontend, _json_safe
from discopt.debug.repl import ReplFrontend
from discopt.debug.session import DebugSession
from test_2043_debug_node_lineage import CASES

# ── unit: rendering on a hand-built context ─────────────────────────────────


class _NullFrontend:
    def interact(self, ctx, session):
        return Control.CONTINUE


def _lineage(ids, parent, depth, var, side, value, status):
    return {
        "id": ids,
        "parent": parent,
        "depth": depth,
        "branch_var": var,
        "branch_side": side,
        "branch_value": value,
        "status": status,
    }


def _ctx(checkpoint="after_process", model=None):
    ctx = DebugContext.from_rust(
        {
            "checkpoint": checkpoint,
            "batch_lineage": _lineage(
                [0, 4], [-1, 1], [0, 2], [-1, 1], ["", "up"], [float("nan"), 2.5], ["branched"] * 2
            ),
            "created": _lineage(
                [7, 8], [4, 4], [3, 3], [0, 0], ["down", "up"], [0.0, 1.0], ["pending", "pruned"]
            )
            if checkpoint == "after_process"
            else None,
        }
    )
    ctx.model = model
    return ctx


def _run(cmd, ctx):
    sess = DebugSession(_NullFrontend())
    res = sess.engine.execute(cmd, ctx, sess)
    assert res.ok, res.output
    return res


@pytest.mark.unit
def test_print_node_reports_lineage_without_a_box():
    res = _run("print node 1", _ctx())
    assert res.output[1].strip() == "parent=1 depth=2 branch=x[1] >= 2.5 status=branched"
    assert res.data == {
        "id": 4,
        "parent": 1,
        "depth": 2,
        "branch": "x[1] >= 2.5",
        "branch_var": 1,
        "branch_side": "up",
        "branch_value": 2.5,
        "status": "branched",
    }
    root = _run("print node 0", _ctx()).data
    assert root["branch"] == "root" and root["parent"] == -1


@pytest.mark.unit
def test_print_node_out_of_range_fails_cleanly():
    sess = DebugSession(_NullFrontend())
    for i in ("2", "-1"):
        res = sess.engine.execute(f"print node {i}", _ctx(), sess)
        assert not res.ok and "out of range" in res.output[0]


@pytest.mark.unit
def test_print_nodes_and_created_render_lineage():
    res = _run("print nodes", _ctx())
    assert res.output[2] == "  [1] id=4 parent=1 depth=2 branch=x[1] >= 2.5 status=branched"
    assert [r["id"] for r in res.data["nodes"]] == res.data["node_ids"] == [0, 4]

    res = _run("print created", _ctx())
    assert res.output[1:] == [
        "  id=7 parent=4 depth=3 branch=x[0] <= 0 status=pending",
        "  id=8 parent=4 depth=3 branch=x[0] >= 1 status=pruned",
    ]
    assert [r["status"] for r in res.data["created"]] == ["pending", "pruned"]

    res = _run("print created", _ctx("after_select"))
    assert res.output == ["no children created at this checkpoint"] and res.data is None


@pytest.mark.unit
def test_branch_text_uses_model_names():
    m = dm.Model("names")
    m.continuous("a", lb=0, ub=1)
    m.continuous("b", shape=(2,), lb=0, ub=1)
    res = _run("print created", _ctx(model=m))
    assert "branch=a <= 0" in res.output[1]
    ctx = _ctx(model=m)
    assert ctx.batch_lineage.branch_text(1, flat_variable_names(m)) == "b[0] >= 2.5"


@pytest.mark.unit
def test_flat_variable_names_refuses_a_size_mismatch():
    m = dm.Model("names")
    m.continuous("a", lb=0, ub=1)
    m.binary("Y", shape=(2, 2))
    assert flat_variable_names(m) == ["a", "Y[0,0]", "Y[0,1]", "Y[1,0]", "Y[1,1]"]
    assert flat_variable_names(m, 5) is not None
    # The tree's box is not this model's variable space: name nothing.
    assert flat_variable_names(m, 7) is None
    assert flat_variable_names(None) is None


@pytest.mark.unit
def test_json_pause_event_carries_batch_and_created():
    fe = JsonFrontend(read_fn=lambda: None, write_fn=lambda e: None)
    caps = fe._hello()["capabilities"]
    assert caps["pause_batch_lineage"] is True and caps["pause_created"] is True

    ev = fe._pause_event(_ctx())
    assert ev["n_batch"] == 2
    assert ev["batch"]["ids"] == [0, 4]
    assert ev["batch"]["branch_side"] == ["", "up"]
    assert ev["created"]["status"] == ["pending", "pruned"]
    # NaN (the root's branch value) must not reach the wire.
    safe = json.dumps(_json_safe(ev), allow_nan=False)
    assert json.loads(safe)["batch"]["branch_value"] == [None, 2.5]

    ev = fe._pause_event(_ctx("after_select"))
    assert "created" not in ev and "batch" in ev
    ev = fe._pause_event(DebugContext.from_rust({"checkpoint": "iter_start"}))
    assert "batch" not in ev and "created" not in ev


# ── integration: both frontends over real solves ────────────────────────────

# Each pause runs the two lineage prints, then resumes to the next checkpoint.
_N_PAUSES = 60
_SCRIPT = ["print nodes", "print created", "stepi"] * _N_PAUSES


def _drive_repl(make):
    out = io.StringIO()
    r = make().solve(debug=DebugSession(ReplFrontend(script=_SCRIPT, out=out)))
    return r, out.getvalue()


def _drive_json(make):
    cmds = iter(_SCRIPT)
    events: list[dict] = []
    fe = JsonFrontend(read_fn=lambda: next(cmds, None), write_fn=events.append)
    r = make().solve(debug=DebugSession(fe))
    return r, events


@pytest.mark.parametrize("make,env,route", CASES, ids=["rust-milp", "python-tree"])
def test_frontends_report_the_tree(make, env, route, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    r, events = _drive_json(make)
    assert r.status == "optimal"
    assert route in (r.algorithm_route or ""), r.algorithm_route
    for e in events:  # strict JSON, including the root's NaN branch value
        json.loads(json.dumps(e, allow_nan=False))

    selected: dict[int, int] = {}  # id -> depth, for every node selected so far
    created: set[int] = set()
    created_status: dict[int, str] = {}
    later_selected: set[int] = set()
    n_children_checked = 0
    n_pauses_with_batch = 0
    pause = None
    for e in events:
        if e["event"] in ("pause", "terminated"):
            pause = e
            continue
        if e["event"] != "result" or e["command"] not in ("print nodes", "print created"):
            continue
        if e["command"] == "print nodes" and pause["checkpoint"] == "after_select":
            n_pauses_with_batch += 1
            batch = pause["batch"]
            assert batch["ids"] == e["data"]["node_ids"]
            assert [n["id"] for n in e["data"]["nodes"]] == batch["ids"]
            for n in e["data"]["nodes"]:
                later_selected.add(n["id"])
                if n["parent"] != -1:
                    assert n["parent"] in selected, f"parent {n['parent']} never selected"
                    assert n["depth"] == selected[n["parent"]] + 1
                    assert n["id"] in created
                    n_children_checked += 1
                selected[n["id"]] = n["depth"]
        if e["command"] == "print created" and pause["checkpoint"] == "after_process":
            rows = e["data"]["created"]
            assert [n["id"] for n in rows] == pause["created"]["ids"]
            for n in rows:
                assert n["branch"] != "root" and n["depth"] >= 1
                created.add(n["id"])
                created_status[n["id"]] = n["status"]
    assert n_pauses_with_batch > 0 and n_children_checked > 0, (
        n_pauses_with_batch,
        n_children_checked,
    )
    # A created child is selected later, was pruned at creation, or was still
    # pending when the scripted pauses ran out.
    for nid in created:
        assert nid in later_selected or created_status[nid] in ("pruned", "pending"), (
            nid,
            created_status[nid],
        )

    # The REPL runs the same engine: the same solve prints the same lines.
    r2, text = _drive_repl(make)
    assert r2.node_count == r.node_count
    json_lines = [
        ln
        for e in events
        if e["event"] == "result" and e["command"] in ("print nodes", "print created")
        for ln in e["output"]
    ]
    assert sum("parent=" in ln for ln in json_lines) > 0
    repl_lines = set(text.splitlines())
    for ln in json_lines:
        assert ln in repl_lines, ln
    # The text and the JSON data agree row by row (``data`` has every row; the
    # text stops at a "... N more" line).
    n_rows = 0
    for e in events:
        if e["event"] == "result" and e["command"] == "print created" and e["data"]:
            shown = [ln for ln in e["output"][1:] if not ln.startswith("  ...")]
            for row, ln in zip(e["data"]["created"], shown):
                n_rows += 1
                assert ln == (
                    f"  id={row['id']} parent={row['parent']} depth={row['depth']} "
                    f"branch={row['branch']} status={row['status']}"
                )
    assert n_rows > 0


def test_python_tree_branch_text_uses_model_names():
    _, events = _drive_json(CASES[1][0])
    branches = {
        n["branch"]
        for e in events
        if e["event"] == "result" and e["command"] == "print created" and e["data"]
        for n in e["data"]["created"]
    }
    assert branches and all(b.split()[0] in ("x", "y") for b in branches), branches
    assert all(("<=" in b) or (">=" in b) for b in branches)
