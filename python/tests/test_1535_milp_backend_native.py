"""#1535: ``milp_backend=`` selects the engine for a pure LP / MILP.

``"native"`` forces discopt's own branch and bound so the search can be watched
(teaching, inspection); ``"highs"`` is the #1229 HiGHS route; ``None`` keeps the env
var / default. These tests pin the routing (which driver ran, with which options),
the options the native MILP tree honours (``strategy``, ``node_callback``,
``milp_cuts``), the route reported on the result, and every combination that would
otherwise quietly run a different engine than the one named.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest

pytest.importorskip("highspy")


def _knapsack(n: int = 20, seed: int = 0, maximize: bool = True) -> dm.Model:
    rng = np.random.default_rng(seed)
    w = rng.integers(5, 30, n)
    v = rng.integers(5, 40, n)
    m = dm.Model("knapsack")
    x = m.binary("x", shape=(n,))
    value = sum(int(v[i]) * x[i] for i in range(n))
    if maximize:
        m.maximize(value)
    else:
        m.minimize(-value)
    m.subject_to(sum(int(w[i]) * x[i] for i in range(n)) <= int(w.sum() // 2))
    return m


def _lp() -> dm.Model:
    m = dm.Model("lp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-x - 2 * y)
    m.subject_to(x + y <= 8)
    m.subject_to(x - y >= -4)
    return m


def _spy(monkeypatch, name: str) -> list:
    """Record every call of ``discopt.solver.<name>`` and pass it through."""
    calls: list = []
    real = getattr(S, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(S, name, wrapper)
    return calls


@pytest.mark.smoke
def test_native_knapsack_branches_and_matches_highs():
    """The issue's acceptance: a 20-variable knapsack on the native tree explores
    nodes and certifies the HiGHS optimum."""
    ref = _knapsack().solve()
    res = _knapsack().solve(milp_backend="native")
    assert ref.status == res.status == "optimal"
    assert res.node_count > 0
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.gap_certified
    assert res.algorithm_route.startswith("native-milp")


def test_native_milp_runs_the_engine_with_the_callers_options(monkeypatch):
    """Python-MILP-tree retirement (R1): ``milp_backend="native"`` is the Rust MILP
    engine, which takes the options the Python tree used to own."""
    import discopt._rust as _rust

    highs = _spy(monkeypatch, "_solve_milp_highs")
    engine = _spy(monkeypatch, "_solve_milp_simplex")
    tree = _spy(monkeypatch, "_solve_milp_bb")
    drv: list = []
    real = _rust.solve_milp_csc_py

    def drv_spy(*a, **kw):
        drv.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(_rust, "solve_milp_csc_py", drv_spy)
    res = _knapsack().solve(
        milp_backend="native",
        strategy="depth_first",
        milp_cuts=False,
        branching_rule="most_fractional",
    )
    assert highs == [] and tree == []
    assert len(engine) == 1
    assert engine[0][1]["root_cuts"] is False
    assert engine[0][1]["branching_rule"] == "most_fractional"
    searches = [kw for kw in drv if "abs_gap_tol" in kw]
    assert searches, "the engine never searched"
    for kw in searches:
        assert kw["node_selection"] == "depth_first"
        assert kw["branching_rule"] == "most_fractional"
        assert kw["root_cuts"] == 0
    assert res.algorithm_route.startswith("native-milp: discopt monolithic Rust MILP engine")
    assert "strategy='depth_first'" in res.algorithm_route


def test_default_route_is_unchanged(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    tree = _spy(monkeypatch, "_solve_milp_bb")
    res = _knapsack().solve()
    assert tree == []
    assert res.node_count == 0 or res.algorithm_route.startswith("highs-milp")
    assert (res.solver_stats or {}).get("route/lp_milp_backend") == 1.0


def test_keyword_overrides_env_var_both_ways(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    res = _knapsack().solve(milp_backend="highs")
    assert (res.solver_stats or {}).get("route/lp_milp_backend") == 1.0
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    res = _knapsack().solve(milp_backend="native")
    assert res.algorithm_route.startswith("native-milp")


def test_milp_cuts_false_skips_the_root_cut_loop():
    on = _knapsack().solve(milp_backend="native")
    off = _knapsack().solve(milp_backend="native", milp_cuts=False)
    cut_keys = lambda r: {k for k in (r.solver_stats or {}) if k.startswith("cuts/")}  # noqa: E731
    # The probe must have something to switch off, or the OFF arm proves nothing.
    assert cut_keys(on), "root cut loop added no cut on the knapsack; test instance is stale"
    assert cut_keys(off) == set()
    assert "root cuts off" in off.algorithm_route
    assert off.objective == pytest.approx(on.objective, abs=1e-6)
    assert off.status == "optimal" and off.gap_certified


@pytest.mark.parametrize("maximize", [True, False])
def test_node_callback_fires_with_values_in_the_users_sense(maximize):
    seen = []
    res = _knapsack(maximize=maximize).solve(
        milp_backend="native", milp_cuts=False, node_callback=lambda ctx, m: seen.append(ctx)
    )
    assert res.status == "optimal"
    assert len(seen) > 0
    counts = [c.node_count for c in seen]
    assert counts == sorted(counts) and counts[-1] <= res.node_count
    last = seen[-1]
    assert last.x_relaxation.shape == (20,)
    assert last.incumbent_obj == pytest.approx(res.objective, abs=1e-4)
    assert last.best_bound is not None
    for c in seen:
        if c.best_bound is None:
            continue
        # A dual bound never crosses the optimum: above it for max, below for min.
        if maximize:
            assert c.best_bound >= res.objective - 1e-6
        else:
            assert c.best_bound <= res.objective + 1e-6


@pytest.mark.parametrize("maximize", [True, False])
def test_node_callback_on_the_engine_reports_the_same_payload(monkeypatch, maximize):
    """Python-MILP-tree retirement: on the Rust engine the callback is fed from the
    driver's batch checkpoint, with the tree's payload in the user's sense: the batch's
    best relaxation (``node_bound`` / ``x_relaxation``) and a dual bound that never
    crosses the optimum."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    seen = []
    res = _knapsack(maximize=maximize).solve(node_callback=lambda ctx, m: seen.append(ctx))
    assert res.status == "optimal" and res.algorithm_route.startswith("native-milp")
    assert len(seen) > 0
    counts = [c.node_count for c in seen]
    assert counts == sorted(counts) and counts[-1] <= res.node_count
    with_relax = [c for c in seen if c.x_relaxation is not None]
    assert with_relax, "no batch reported a relaxation"
    assert all(c.x_relaxation.shape == (20,) and c.node_bound is not None for c in with_relax)
    for c in seen:
        if c.best_bound is not None:
            if maximize:
                assert c.best_bound >= res.objective - 1e-6
            else:
                assert c.best_bound <= res.objective + 1e-6
        if c.node_bound is not None:  # a node's relaxation bound also never crosses it
            if maximize:
                assert c.node_bound >= res.objective - 1e-6
            else:
                assert c.node_bound <= res.objective + 1e-6


def test_native_lp_uses_the_discopt_simplex():
    ref = _lp().solve()
    res = _lp().solve(milp_backend="native")
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.algorithm_route.startswith("native-lp")


def test_native_with_lazy_constraints_reports_the_spatial_route():
    """#748 keeps a MILP with lazy constraints on spatial B&B; the result says so."""
    calls = []

    def lazy(ctx, model):
        calls.append(1)
        return []

    res = _knapsack(n=8).solve(milp_backend="native", lazy_constraints=lazy)
    ref = _knapsack(n=8).solve()
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.algorithm_route.startswith("native-spatial")
    assert calls


@pytest.mark.parametrize(
    "model, kwargs, match",
    [
        (_knapsack, {"milp_backend": "gurobi"}, "expected one of"),
        (_knapsack, {"milp_cuts": False}, "milp_backend='native'"),
        (_knapsack, {"milp_backend": "highs", "milp_cuts": False}, "milp_backend='native'"),
        (_knapsack, {"milp_backend": "native", "nlp_solver": "simplex"}, "nlp_solver"),
        (_knapsack, {"milp_backend": "native", "solver": "amp"}, "different engines"),
        (_lp, {"milp_backend": "native", "milp_cuts": False}, "does not run for an LP"),
        (
            _knapsack,
            {"milp_backend": "native", "milp_cuts": True, "lazy_constraints": lambda *a: []},
            "spatial",
        ),
    ],
)
def test_conflicting_or_meaningless_requests_are_refused(model, kwargs, match):
    with pytest.raises(ValueError, match=match):
        model().solve(**kwargs)


def test_refused_for_a_nonlinear_model():
    m = dm.Model("minlp")
    x = m.continuous("x", lb=0, ub=4)
    z = m.binary("z")
    m.minimize((x - 1.5) ** 2 + z)
    m.subject_to(x <= 3 * z + 1)
    with pytest.raises(ValueError, match="LP or MILP"):
        m.solve(milp_backend="native")


# ── branching rules ─────────────────────────────────────────────────────


def _multi_knapsack(n: int = 20, k: int = 5, seed: int = 0) -> dm.Model:
    """Several rows, so a node LP has several fractional columns to choose from (a
    single-row knapsack vertex has at most one, which makes every rule agree)."""
    rng = np.random.default_rng(seed)
    W = rng.integers(5, 40, (k, n))
    v = rng.integers(10, 50, n)
    m = dm.Model("multi_knapsack")
    x = m.binary("x", shape=(n,))
    m.maximize(sum(int(v[i]) * x[i] for i in range(n)))
    for r in range(k):
        m.subject_to(sum(int(W[r, i]) * x[i] for i in range(n)) <= int(W[r].sum() // 2))
    return m


def test_branching_rules_change_the_search_but_not_the_answer():
    """On the Rust engine (retirement step 4): every rule certifies the same optimum;
    the fractionality rules never probe, full strong branching probes more than the
    default reliability rule (which probes only unreliable candidates, within a node
    budget), and the rules do not all explore the same tree."""
    ref = _multi_knapsack().solve()
    nodes, probes = {}, {}
    for rule in S._MILP_BRANCHING_RULES:
        res = _multi_knapsack().solve(milp_backend="native", milp_cuts=False, branching_rule=rule)
        assert res.status == "optimal" and res.gap_certified
        assert res.objective == pytest.approx(ref.objective, abs=1e-6)
        assert f"branching_rule={rule!r}" in res.algorithm_route
        assert res.algorithm_route.startswith("native-milp: discopt monolithic Rust MILP")
        nodes[rule] = res.node_count
        probes[rule] = (res.solver_stats or {})["branching/strong_probe_lps"]
    assert probes["most_fractional"] == probes["least_fractional"] == 0, probes
    assert probes["strong"] > 0 and probes["strong"] >= probes["pseudocost"], probes
    assert len(set(nodes.values())) > 1, nodes


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"branching_rule": "strong"}, "milp_backend='native'"),
        ({"milp_backend": "native", "branching_rule": "random"}, "expected one of"),
        (
            {"milp_backend": "native", "branching_rule": "strong", "cut_callback": lambda c, m: []},
            "spatial",
        ),
    ],
)
def test_branching_rule_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _knapsack().solve(**kwargs)


def test_branching_rule_refused_on_an_lp():
    with pytest.raises(ValueError, match="does not run for an LP"):
        _lp().solve(milp_backend="native", branching_rule="strong")


# ── callbacks on a MILP without milp_backend ────────────────────────────


def test_node_callback_on_a_default_milp_runs_the_engine(monkeypatch):
    """Previously the HiGHS route returned with the callback never called. Since the
    Python-MILP-tree retirement the callback is served by the Rust engine's batch
    checkpoint (``_milp_node_callback_hook``), not by the Python tree."""
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    highs = _spy(monkeypatch, "_solve_milp_highs")
    seen = []
    ref = _knapsack().solve()
    highs.clear()
    res = _knapsack().solve(node_callback=lambda ctx, m: seen.append(ctx))
    assert highs == []
    assert seen and res.node_count > 0
    assert res.algorithm_route.startswith("native-milp"), res.algorithm_route
    assert "route/python_milp_tree_entry" not in (res.solver_stats or {})
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)


def test_node_callback_with_explicit_highs_is_refused():
    with pytest.raises(ValueError, match="no node or cut hook"):
        _knapsack().solve(milp_backend="highs", node_callback=lambda ctx, m: None)


def test_cut_callback_on_a_milp_is_called():
    """Neither MILP engine nor the NLP-BB auto-select has a cut hook; the callback
    used to be silently skipped (0 calls) on every pure-MILP route."""
    calls = []

    def cut(ctx, model):
        calls.append(ctx.node_id)
        return []

    ref = _knapsack(n=10).solve()
    res = _knapsack(n=10).solve(cut_callback=cut)
    assert calls
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)


def test_cut_callback_with_explicit_nlp_bb_is_refused():
    with pytest.raises(ValueError, match="nlp_bb=True"):
        _knapsack(n=10).solve(cut_callback=lambda ctx, m: [], nlp_bb=True)


def test_node_callback_on_an_lp_warns_that_it_is_not_called():
    with pytest.warns(UserWarning, match="node_callback not called"):
        _lp().solve(node_callback=lambda ctx, m: None)


# ── review of #1538 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"milp_backend": "native", "nlp_bb": True},
        {"milp_backend": "native", "nlp_bb": True, "branching_rule": "strong", "milp_cuts": False},
        {"milp_backend": "highs", "nlp_bb": True},
    ],
)
def test_nlp_bb_true_with_an_explicit_backend_is_refused(kwargs):
    """Review 1 (blocking): nlp_bb=True returned from NLP-BB before the LP/MILP
    dispatch, so the named engine and every tree option silently did not run
    (measured: optimal, node_count=1, route None)."""
    with pytest.raises(ValueError, match="nlp_bb=True"):
        _knapsack().solve(**kwargs)


@pytest.mark.parametrize("model", [_knapsack, _lp])
@pytest.mark.parametrize("cb", ["lazy_constraints", "incumbent_callback"])
def test_highs_with_a_feasibility_callback_is_refused(model, cb):
    """Review 2: these callbacks route to spatial B&B; HiGHS never runs."""
    with pytest.raises(ValueError, match="milp_backend='highs' cannot honour"):
        model().solve(milp_backend="highs", **{cb: lambda *a: None})


def test_native_lp_route_names_the_engine_that_answered():
    """Review 3: the label used to say "simplex" whatever answered."""
    res = _lp().solve(milp_backend="native")
    stats = res.solver_stats or {}
    assert ("lp/engine_simplex" in stats) != ("lp/engine_pounce" in stats)
    engine = "Rust simplex" if "lp/engine_simplex" in stats else "POUNCE"
    assert engine in res.algorithm_route


def test_tree_options_on_a_model_dispatched_as_an_lp_warn(monkeypatch):
    """Review 4: the options are validated against the raw model's class. Should
    the dispatch see an LP (every integer gone), they would do nothing, so warn.

    No in-tree pass changes a variable's type today, so the dispatch-time class is
    forced here: the validation's call (the first) reports MILP, later calls the
    model's true class (LP)."""
    import discopt._relax.problem_classifier as pc

    real = pc.classify_problem
    calls = []

    def stub(model):
        calls.append(1)
        return pc.ProblemClass.MILP if len(calls) == 1 else real(model)

    monkeypatch.setattr(pc, "classify_problem", stub)
    with pytest.warns(UserWarning, match="branching_rule unused"):
        res = _lp().solve(milp_backend="native", branching_rule="strong")
    assert len(calls) > 1
    assert res.status == "optimal"


def test_node_callback_reports_no_bound_for_a_batch_without_a_relaxation():
    """Review 5: an all-sentinel batch used to report node_bound ~ +-1e30 and the
    box midpoint as x_relaxation."""

    class _Tree:
        def stats(self):
            return {"total_nodes": 3, "global_lower_bound": 1.0, "gap": 0.5}

        def incumbent(self):
            return None

    seen = []
    m = _knapsack(n=4)
    lbs = np.full(2, S._INFEASIBILITY_SENTINEL)
    sols = np.full((2, 6), 0.5)
    S._fire_milp_node_callback(lambda c, mm: seen.append(c), m, _Tree(), lbs, sols, 4, True, 0.0)
    assert len(seen) == 1
    assert seen[0].node_bound is None and seen[0].x_relaxation is None
    # A batch with one real bound still reports it (and its point), in user sense.
    lbs[1] = -7.0
    sols[1, :4] = [1.0, 0.0, 1.0, 0.0]
    S._fire_milp_node_callback(lambda c, mm: seen.append(c), m, _Tree(), lbs, sols, 4, True, 0.0)
    assert seen[1].node_bound == pytest.approx(7.0)  # maximize: internal -obj
    assert seen[1].x_relaxation.tolist() == [1.0, 0.0, 1.0, 0.0]
