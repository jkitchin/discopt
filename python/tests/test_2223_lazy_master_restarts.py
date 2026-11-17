"""Issue #2223 cause A: bounded tree restarts for the in-house lazy master.

The Rust lazy master builds its root, root cuts and tree before any OA row exists
and never restarts; the HiGHS master restarts after each lazy veto. Measured on
``cvxnonsep_psig40r`` (60 s): the master spent 2,347 nodes / 30.4 s without
certifying and the route fell back (35.3 s total); with
``DISCOPT_MILP_LAZY_RESTARTS=3`` one master call certified in 60 nodes / 1.6 s
(1.8 s total, no fallback). These tests pin the restart contract of
``milp_simplex.solve_milp_with_lazy_cuts`` (see ``_lazy_restarts``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from discopt.solvers import SolveStatus

ms = pytest.importorskip("discopt.solvers.milp_simplex")


def _instance(seed: int = 2223):
    """max value over x in {0..2}^8 with one static knapsack row and four hidden
    knapsack rows the separator enforces lazily (most violated first)."""
    rng = np.random.default_rng(seed)
    n = 8
    c = -rng.integers(5, 40, n).astype(float)
    A = rng.integers(1, 20, (1, n)).astype(float)
    b = np.array([A.sum() * 0.9])
    H = rng.integers(1, 20, (4, n)).astype(float)
    h = H.sum(axis=1) * 0.45
    return c, A, b, H, h, [(0, 2)] * n, np.ones(n)


def _reference(c, A, b, H, h, bounds, integ):
    ref = ms.solve_milp(
        c,
        A_ub=np.vstack([A, H]),
        b_ub=np.concatenate([b, h]),
        bounds=bounds,
        integrality=integ,
        gap_tolerance=1e-9,
    )
    assert ref.status == SolveStatus.OPTIMAL
    return float(ref.objective)


class _Trees:
    """Counts the trees a call runs (wraps the one-tree worker)."""

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []
        orig = ms._solve_lazy_tree

        def wrapped(*args, **kwargs):
            self.calls.append(
                {
                    "node_callback": kwargs.get("node_callback") is not None,
                    "node_hook_cut_cap": kwargs.get("node_hook_cut_cap"),
                    "rows": 0 if kwargs.get("b_ub") is None else int(np.size(kwargs["b_ub"])),
                }
            )
            return orig(*args, **kwargs)

        monkeypatch.setattr(ms, "_solve_lazy_tree", wrapped)


def _separator(H, h, trees, log):
    returned: list[tuple[int, int]] = []  # (tree index, hidden row) of every returned row

    def lazy(x):
        tree = len(trees.calls) - 1
        viol = H @ x - h
        # The carried-row contract: a row returned in an EARLIER tree is part of
        # this tree's matrix, so no point violating it may reach the separator.
        for t, r in returned:
            if t < tree:
                log["checked"] = log.get("checked", 0) + 1
                if viol[r] > 1e-6:
                    log["reoffered"] += 1
        log["calls"] += 1
        r = int(np.argmax(viol))
        if viol[r] > 1e-9:
            returned.append((tree, r))
            return [(H[r], float(h[r]))]
        return None

    return lazy


def test_restart_flag_parsing(monkeypatch):
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "3")
    assert ms._lazy_restarts() == 3
    monkeypatch.delenv("DISCOPT_MILP_LAZY_RESTARTS")
    assert ms._lazy_restarts() == 3  # graduated default
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "0")
    assert ms._lazy_restarts() == 0
    for bad in ("-1", "x"):
        monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", bad)
        with pytest.raises(ValueError, match="DISCOPT_MILP_LAZY_RESTARTS"):
            ms._lazy_restarts()


def test_restart_carries_rows_and_keeps_the_optimum(monkeypatch):
    c, A, b, H, h, bounds, integ = _instance()
    opt = _reference(c, A, b, H, h, bounds, integ)
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "3")
    trees = _Trees(monkeypatch)
    log = {"calls": 0, "reoffered": 0}
    res = ms.solve_milp_with_lazy_cuts(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integ,
        gap_tolerance=1e-9,
        lazy_callback=_separator(H, h, trees, log),
    )
    stats = res.callback_stats
    # The mechanism fired (CLAUDE.md section 6): more than one tree, rows carried.
    assert stats["restarts"] >= 1 and len(trees.calls) == stats["restarts"] + 1
    assert stats["restart_rows_lazy"] >= 1
    assert trees.calls[-1]["rows"] == 1 + stats["restart_rows_lazy"]
    assert log["calls"] > 0 and stats["mipsol_calls"] == log["calls"]
    assert log.get("checked", 0) > 0, "no point reached a restarted tree's separator"
    assert log["reoffered"] == 0, "a point cut off by a carried row reached the separator"
    # A restart stop is not a caller stop, and the final tree certifies.
    assert stats["terminated"] is False and stats["terminate_calls"] == 0
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(opt, abs=1e-6)
    assert res.bound <= opt + 1e-6
    assert np.all(H @ res.x <= h + 1e-6)


def test_restarts_opt_out_is_one_tree(monkeypatch):
    c, A, b, H, h, bounds, integ = _instance()
    opt = _reference(c, A, b, H, h, bounds, integ)
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "0")
    trees = _Trees(monkeypatch)
    log = {"calls": 0, "reoffered": 0}
    res = ms.solve_milp_with_lazy_cuts(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integ,
        gap_tolerance=1e-9,
        lazy_callback=_separator(H, h, trees, log),
    )
    assert len(trees.calls) == 1 and "restarts" not in res.callback_stats
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(opt, abs=1e-6)


def test_caller_stop_wins_over_a_pending_restart(monkeypatch):
    c, A, b, H, h, bounds, integ = _instance()
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "3")
    trees = _Trees(monkeypatch)
    log = {"calls": 0, "reoffered": 0}
    lazy = _separator(H, h, trees, log)
    vetoes = [0]

    def counting_lazy(x):
        rows = lazy(x)
        vetoes[0] += bool(rows)
        return rows

    res = ms.solve_milp_with_lazy_cuts(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integ,
        gap_tolerance=1e-9,
        lazy_callback=counting_lazy,
        terminate_callback=lambda _s: vetoes[0] > 0,
    )
    stats = res.callback_stats
    assert vetoes[0] >= 1, "no veto: the restart was never requested"
    assert stats["terminate_calls"] >= 1 and stats["terminated"] is True
    assert len(trees.calls) == 1 and stats["restarts"] == 0
    assert res.status not in (SolveStatus.OPTIMAL, SolveStatus.INFEASIBLE)


def test_node_cut_cap_spans_restarts(monkeypatch):
    c, A, b, H, h, bounds, integ = _instance()
    opt = _reference(c, A, b, H, h, bounds, integ)
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "3")
    trees = _Trees(monkeypatch)
    log = {"calls": 0, "reoffered": 0}
    k = [0]

    def node(x):
        # Globally valid, distinct, loose rows: x_j <= 2 + small offset.
        k[0] += 1
        j = k[0] % len(x)
        row = np.zeros(len(x))
        row[j] = 1.0
        return [(row, 2.0 + 1e-3 * k[0])]

    cap = 3
    res = ms.solve_milp_with_lazy_cuts(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        integrality=integ,
        gap_tolerance=1e-9,
        lazy_callback=_separator(H, h, trees, log),
        node_callback=node,
        node_hook_rounds=2,
        node_hook_cut_cap=cap,
    )
    stats = res.callback_stats
    assert stats["restarts"] >= 1 and k[0] > 0
    assert stats["restart_rows_node"] <= cap
    # Every tree gets only what is left of the cap, and no node hook once it is spent.
    assert trees.calls[0]["node_callback"] and trees.calls[0]["node_hook_cut_cap"] == cap
    caps = [t["node_hook_cut_cap"] for t in trees.calls if t["node_callback"]]
    assert caps == sorted(caps, reverse=True)
    last = trees.calls[-1]
    if last["node_callback"]:
        assert last["node_hook_cut_cap"] == cap - stats["restart_rows_node"], trees.calls
    else:
        assert stats["restart_rows_node"] == cap, trees.calls
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(opt, abs=1e-6)


def _ellipsoid(seed: int = 2223):
    """max c.x over x in {0..3}^6 inside the convex ellipsoid ``w.x^2 <= R``, which
    lives only in the separators (tangent cuts): unlike the four hidden knapsack
    rows above, it keeps producing rows after a restart, so fractional nodes of a
    restarted tree still separate. The optimum is by enumeration (4^6 points)."""
    rng = np.random.default_rng(seed)
    n = 6
    w = rng.uniform(0.5, 2.0, n)
    c = -rng.uniform(1.0, 5.0, n)
    R = 0.35 * float(w @ np.full(n, 9.0))

    def cut(x):
        x = np.asarray(x, dtype=float)
        g = float(w @ x**2 - R)
        if g <= 1e-9:
            return []
        grad = 2.0 * w * x
        return [(grad, float(grad @ x - g))]

    pts = np.array(list(itertools.product(range(4), repeat=n)), dtype=float)
    feas = pts[(pts**2) @ w <= R + 1e-9]
    opt = float(np.min(feas @ c))
    kw = dict(
        A_ub=np.ones((1, n)),
        b_ub=np.array([100.0]),
        bounds=[(0, 3)] * n,
        integrality=np.ones(n),
        gap_tolerance=1e-9,
        time_limit=60.0,
    )
    return c, w, R, cut, opt, kw


def test_node_hook_fires_in_restarted_trees_under_the_default(monkeypatch):
    """The node-hook path under the graduated default K (test_1060/test_1141 pin
    ``=0`` for their single-tree semantics, so this is its coverage at K=3)."""
    c, w, R, cut, opt, kw = _ellipsoid()
    monkeypatch.delenv("DISCOPT_MILP_LAZY_RESTARTS", raising=False)
    assert ms._lazy_restarts() == 3
    trees = _Trees(monkeypatch)
    per_tree: dict[int, int] = {}

    def node(x):
        rows = cut(x)
        if rows:
            t = len(trees.calls) - 1
            per_tree[t] = per_tree.get(t, 0) + 1
        return rows

    res = ms.solve_milp_with_lazy_cuts(
        c,
        **kw,
        lazy_callback=cut,
        node_callback=node,
        node_hook_rounds=2,
        node_hook_cut_cap=200,
    )
    stats = res.callback_stats
    assert stats["restarts"] >= 1, "no restart: this is the single-tree path again"
    assert stats["mipnode_calls"] > 0 and stats["driver_node_cuts"] > 0
    assert any(t >= 1 for t in per_tree), f"node hook never fired after a restart: {per_tree}"
    assert res.status == SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(opt, abs=1e-6)
    assert res.bound <= opt + 1e-6
    assert float(w @ res.x**2) <= R + 1e-6


def test_an_accepted_point_survives_a_tree_that_loses_its_seed(monkeypatch):
    """#2228 review: a restart hands the accepted point on only as ``mip_start``. If
    a later tree drops it and reports INFEASIBLE (or no point), the call must still
    return the accepted point, uncertified -- never INFEASIBLE after an acceptance.

    The trees are scripted, so the scenario is exact: tree 1 offers a feasible point
    (accepted) and an infeasible one (vetoed), then stops at its check-in for the
    restart without publishing a point; tree 2 loses the seed and reports
    INFEASIBLE."""
    c, w, R, cut, opt, kw = _ellipsoid()
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "3")
    good = np.zeros(6)
    bad = np.full(6, 3.0)
    assert not cut(good) and cut(bad)
    trees: list[dict] = []

    def scripted(*args, **kwargs):
        trees.append(kwargs)
        if len(trees) == 1:
            assert not kwargs["lazy_callback"](good)
            assert kwargs["lazy_callback"](bad)
            assert kwargs["terminate_callback"]({"dual_bound": -np.inf}) is True
            return ms.MILPResult(
                status=SolveStatus.ITERATION_LIMIT, node_count=3, callback_stats={}
            )
        return ms.MILPResult(status=SolveStatus.INFEASIBLE, node_count=1, callback_stats={})

    monkeypatch.setattr(ms, "_solve_lazy_tree", scripted)
    res = ms.solve_milp_with_lazy_cuts(c, **kw, lazy_callback=cut)
    assert len(trees) == 2, "the restart did not happen"
    assert np.allclose(trees[1]["mip_start"], good), "the accepted point was not the seed"
    assert res.status == SolveStatus.ITERATION_LIMIT
    assert res.x is not None and np.allclose(res.x, good)
    assert res.objective == pytest.approx(float(c @ good))
    assert res.callback_stats["restarts"] == 1


@pytest.mark.slow
@pytest.mark.correctness
def test_psig40r_master_tree_is_small_with_restarts(monkeypatch):
    import os
    from pathlib import Path

    import discopt
    from discopt.modeling import from_nl

    here = Path(__file__).parent
    assert os.path.realpath(discopt.__file__).startswith(os.path.realpath(here.parent / "discopt"))
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "3")
    masters: list = []
    orig = ms.solve_milp_with_lazy_cuts

    def counted(*args, **kwargs):
        out = orig(*args, **kwargs)
        masters.append(out)
        return out

    monkeypatch.setattr(ms, "solve_milp_with_lazy_cuts", counted)
    # 60 s backstop over the ~2 s measured (35 s without restarts).
    res = from_nl(str(here / "data" / "minlplib_nl" / "cvxnonsep_psig40r.nl")).solve(time_limit=60)
    assert masters, "the Rust lazy master never ran"
    assert sum(int(m.callback_stats.get("restarts", 0)) for m in masters) >= 1
    assert sum(int(m.node_count) for m in masters) < 500
    route = str(res.algorithm_route)
    assert "fell back" not in route, route
    assert res.gap_certified and res.status == "optimal"
    assert res.objective == pytest.approx(86.5452, rel=1e-4)
