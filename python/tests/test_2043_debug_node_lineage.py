"""#2043: the debugger reports each node's parent, depth and creating branch.

Before this, ``DebugContext`` carried a batch's ids and boxes but not how the nodes
relate, so a frontend that draws the search tree had to rebuild it from box
containment -- O(N^2), blind to children pruned before selection, and wrong
whenever bound propagation also tightened a sibling's other coordinates.

Each test records every checkpoint of a real solve and checks the tree the
debugger reports against the boxes it reports, on the Python tree and on the
pure-Rust MILP route (whose hook marshals the same fields).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.debug.engine import Control
from discopt.debug.session import DebugSession

TOL = 1e-9


class _Recorder:
    """Records lineage and boxes at each checkpoint; never pauses the solve."""

    def __init__(self):
        self.selected: list[tuple[int, int, int, str, float, np.ndarray, np.ndarray]] = []
        self.created: set[int] = set()
        self.closed: dict[int, str] = {}
        self.n_after_select = 0
        self.n_after_process = 0

    def interact(self, ctx, session):
        cp = str(ctx.checkpoint)
        if cp == "after_select":
            self.n_after_select += 1
            lin = ctx.batch_lineage
            assert lin is not None, "after_select carried no lineage"
            assert list(lin.ids) == list(ctx.batch_ids)
            for k, nid in enumerate(lin.ids):
                self.selected.append(
                    (
                        int(nid),
                        int(lin.parent[k]),
                        int(lin.depth[k]),
                        lin.branch_side[k],
                        float(lin.branch_value[k]),
                        # Copies: the solve loop tightens its batch arrays in place
                        # after this checkpoint, and the children inherit the box
                        # the tree held at selection.
                        np.array(ctx.batch_lb[k], dtype=float),
                        np.array(ctx.batch_ub[k], dtype=float),
                    )
                )
                assert int(ctx.batch_branch_var[k]) == int(lin.branch_var[k])
        elif cp == "after_process":
            self.n_after_process += 1
            assert ctx.created is not None, "after_process carried no created list"
            for k, nid in enumerate(ctx.created.ids):
                assert ctx.created.depth[k] >= 1
                self.created.add(int(nid))
            lin = ctx.batch_lineage
            assert lin is not None, "after_process carried no batch lineage"
            for k, nid in enumerate(lin.ids):
                self.closed[int(nid)] = lin.status[k]
        return Control.STEPI


def _check_tree(rec: _Recorder, branch_vars: dict[int, int]) -> int:
    """Assert the reported tree is consistent with the reported boxes. Returns the
    number of non-root nodes checked, so a caller can prove the check ran."""
    assert rec.n_after_select > 0 and rec.n_after_process > 0
    seen: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    checked = 0
    for nid, parent, depth, side, value, lb, ub in rec.selected:
        if parent == -1:
            assert depth == 0 and side == "" and np.isnan(value)
        else:
            assert parent in seen, f"node {nid}: parent {parent} was never selected earlier"
            p_depth, p_lb, p_ub = seen[parent]
            assert depth == p_depth + 1
            assert np.all(lb >= p_lb - TOL) and np.all(ub <= p_ub + TOL), (
                f"node {nid}: box not inside parent {parent}'s"
            )
            assert nid in rec.created, f"node {nid} was selected but never reported created"
            var = branch_vars[nid]
            assert side in ("down", "up")
            if side == "down":
                assert ub[var] <= value + TOL
            else:
                assert lb[var] >= value - TOL
            checked += 1
        seen[nid] = (depth, lb, ub)
    # Every selected node was closed somehow by the time the batch was processed.
    for nid, *_ in rec.selected:
        assert rec.closed.get(nid) in ("branched", "pruned", "fathomed"), (nid, rec.closed.get(nid))
    return checked


class _VarRecorder(_Recorder):
    """Also keeps each selected node's branching variable for ``_check_tree``."""

    def __init__(self):
        super().__init__()
        self.branch_var: dict[int, int] = {}

    def interact(self, ctx, session):
        if str(ctx.checkpoint) == "after_select":
            for k, nid in enumerate(ctx.batch_lineage.ids):
                self.branch_var[int(nid)] = int(ctx.batch_lineage.branch_var[k])
        return super().interact(ctx, session)


def _market_split():
    """A 2x10 market-split instance (Cornuejols & Dawande): cuts do not close it, so
    the Rust MILP engine branches (239 nodes, measured)."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 100, size=(2, 10))
    b = a.sum(axis=1) // 2
    m = dm.Model("market_split")
    x = [m.binary(f"x{i}") for i in range(10)]
    sp = [m.continuous(f"sp{j}", lb=0, ub=1000) for j in range(2)]
    sm = [m.continuous(f"sm{j}", lb=0, ub=1000) for j in range(2)]
    for j in range(2):
        m.subject_to(sum(int(a[j, i]) * x[i] for i in range(10)) + sp[j] - sm[j] == int(b[j]))
    m.minimize(sum(sp) + sum(sm))
    return m


def _ncnlp():
    m = dm.Model("ncnlp")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize(x**4 - 3 * x**2 + y**2 + x * y)
    return m


# The pure MILP is pinned to the Rust engine, whose hook marshals the lineage; the
# continuous model takes the Python tree because a debugger is attached (#2042).
CASES = [
    (_market_split, {"DISCOPT_LP_MILP_BACKEND": "rust"}, "native-milp"),
    (_ncnlp, {}, "Python tree"),
]


@pytest.mark.parametrize("make,env,route", CASES, ids=["rust-milp", "python-tree"])
def test_reported_tree_matches_reported_boxes(make, env, route, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    rec = _VarRecorder()
    r = make().solve(debug=DebugSession(rec))
    assert r.status == "optimal"
    assert route in (r.algorithm_route or ""), r.algorithm_route
    checked = _check_tree(rec, rec.branch_var)
    assert checked > 0, "no non-root node was selected, so nothing was checked"
    # ``created`` covers every selected child (checked per node above) and also the
    # children a route prunes before selecting them, so it can only be larger.
    n_selected_children = sum(1 for s in rec.selected if s[1] != -1)
    assert len(rec.created) >= n_selected_children


@pytest.mark.parametrize("make,env,route", CASES, ids=["rust-milp", "python-tree"])
def test_lineage_is_bound_neutral(make, env, route, monkeypatch):
    """Reading lineage is bookkeeping: an attached recorder changes nothing.

    Compared against a no-op session (same route, so the only difference is the
    lineage reads), not against a detached solve, which may take another route.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    class _Silent:
        def interact(self, ctx, session):
            return Control.CONTINUE

    base = make().solve(debug=DebugSession(_Silent()))
    rec = make().solve(debug=DebugSession(_VarRecorder()))
    assert rec.node_count == base.node_count
    assert rec.objective == pytest.approx(base.objective, abs=1e-9)


def test_tree_manager_lineage_rejects_unknown_ids():
    from discopt._rust import PyTreeManager

    tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
    tm.initialize()
    d = tm.node_lineage(np.array([0], dtype=np.int64))
    assert int(d["parent"][0]) == -1 and int(d["depth"][0]) == 0
    assert list(tm.last_created()) == [0]
    with pytest.raises(ValueError):
        tm.node_lineage(np.array([5], dtype=np.int64))
