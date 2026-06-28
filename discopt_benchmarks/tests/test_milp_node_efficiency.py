"""Smoke tests for the MILP node-efficiency micro-bench (issue #331, Step 1).

Fast and deterministic: solve the two smallest ``mdk`` instances with the
discopt simplex engine and assert the bench's *attribution machinery* is sound —
the instance generator is reproducible, the engine proves optimality, the root
LP / root-after-cuts / gap-closed metrics are ordered correctly, and every
ablation lever produces a comparable node count. SCIP is exercised only if
``pyscipopt`` is importable. The full table is the nightly run.
"""

from __future__ import annotations

import numpy as np
import pytest

milp_ne = pytest.importorskip("discopt_benchmarks.perf.milp_node_efficiency")

pytestmark = pytest.mark.smoke

# The Rust simplex engine must be present for the engine-side bench to run.
_rust = pytest.importorskip("discopt._rust")
if not hasattr(_rust, "solve_milp_py"):
    pytest.skip("discopt._rust.solve_milp_py unavailable", allow_module_level=True)


def test_gen_mdk_is_deterministic():
    a = milp_ne.gen_mdk(30, 5)
    b = milp_ne.gen_mdk(30, 5)
    assert a.name == b.name == "mdk30x5"
    assert a.weights.shape == (5, 30)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.cap, b.cap)
    # Half-sum capacities and integer weights in [1, 100].
    assert a.weights.min() >= 1 and a.weights.max() <= 100
    np.testing.assert_array_equal(a.cap, np.floor(0.5 * a.weights.sum(axis=1)))


def test_gen_sparse_mdk_is_deterministic_and_sparse():
    a = milp_ne.gen_sparse_mdk(60, 20)
    b = milp_ne.gen_sparse_mdk(60, 20)
    assert a.name == b.name == "smdk60x20"
    np.testing.assert_array_equal(a.weights, b.weights)
    # Each row constrains only ~25% of the items (the rest are exactly zero).
    nnz_frac = (a.weights != 0).mean()
    assert 0.15 <= nnz_frac <= 0.35, nnz_frac


def test_reduced_cost_fixing_is_sound_and_reduces_nodes():
    inst = milp_ne.gen_mdk(90, 12)
    std = milp_ne._std_form(inst)
    off = milp_ne.solve_discopt(
        std,
        milp_ne._cfg(
            root_cuts=16,
            cut_rounds=1,
            max_pool_cuts=128,
            heuristics=True,
            presolve=True,
            strong_branch=True,
            reduced_cost_fixing=False,
        ),
        max_nodes=2_000_000,
    )
    on = milp_ne.solve_discopt(
        std,
        milp_ne._cfg(
            root_cuts=16,
            cut_rounds=1,
            max_pool_cuts=128,
            heuristics=True,
            presolve=True,
            strong_branch=True,
            reduced_cost_fixing=True,
        ),
        max_nodes=2_000_000,
    )
    assert off.status == "optimal" and on.status == "optimal"
    # Sound: same optimum.
    assert abs(off.obj - on.obj) <= 1e-6 * (1 + abs(off.obj))
    # Reduces (or at least never inflates) the node count.
    assert on.nodes <= off.nodes


def test_sparse_cut_profile_is_sound():
    """The sparse cut profile (`gmi_cuts` off + `cut_select` on) only *chooses
    among* valid cuts, so it must reach the same optimum as prod — the levers can
    change the node count but never the proven objective. (Step 7: the profile is
    a difficulty-keyed wall win on the hardest sparse instance only, so it is a
    caller-tunable lever, not a default — here we lock its soundness.)"""
    inst = milp_ne.gen_sparse_mdk(60, 20)
    std = milp_ne._std_form(inst)
    prod = milp_ne.solve_discopt(std, milp_ne.ABLATION["prod"], max_nodes=2_000_000)
    profile_cfg = dict(milp_ne.ABLATION["prod"])
    profile_cfg.update(
        gmi_cuts=False, cut_select=True, cut_rounds=3,
        root_cuts=2 * inst.m, max_pool_cuts=4 * inst.m,
    )
    profile = milp_ne.solve_discopt(std, profile_cfg, max_nodes=2_000_000)
    assert prod.status == "optimal" and profile.status == "optimal"
    assert abs(prod.obj - profile.obj) <= 1e-6 * (1 + abs(prod.obj))


def test_node_cuts_sound_gated_and_reduce_sparse_nodes():
    """Node cuts (Step 10): density-gated cover separation at fractional nodes.

    * Sound everywhere — same proven optimum with cuts on vs off.
    * Density-gated: on a *dense*-row model the gate suppresses node cuts, so the
      tree (node count) is identical to off — they cannot bloat dense LPs.
    * Effective: on a *sparse*-row model they cut the node count.
    """
    base = {
        "root_cuts": 16, "cut_rounds": 1, "max_pool_cuts": 128, "heuristics": True,
        "presolve": True, "strong_branch": True, "reduced_cost_fixing": True,
    }

    # Dense rows → density gate suppresses node cuts → identical tree.
    dense = milp_ne._std_form(milp_ne.gen_mdk(60, 8))
    d_off = milp_ne.solve_discopt(dense, milp_ne._cfg(**base, node_cuts=False), max_nodes=2_000_000)
    d_on = milp_ne.solve_discopt(dense, milp_ne._cfg(**base, node_cuts=True), max_nodes=2_000_000)
    assert d_off.status == "optimal" and d_on.status == "optimal"
    assert abs(d_off.obj - d_on.obj) <= 1e-6 * (1 + abs(d_off.obj))
    assert d_on.nodes == d_off.nodes, "density gate should make node cuts inert on dense rows"

    # Sparse rows → node cuts are sound and reduce the node count.
    sparse = milp_ne._std_form(milp_ne.gen_sparse_mdk(80, 25))
    cfg_off = milp_ne._cfg(**base, node_cuts=False)
    cfg_on = milp_ne._cfg(**base, node_cuts=True)
    s_off = milp_ne.solve_discopt(sparse, cfg_off, max_nodes=2_000_000)
    s_on = milp_ne.solve_discopt(sparse, cfg_on, max_nodes=2_000_000)
    assert s_off.status == "optimal" and s_on.status == "optimal"
    assert abs(s_off.obj - s_on.obj) <= 1e-6 * (1 + abs(s_off.obj))
    assert s_on.nodes < s_off.nodes, "node cuts should reduce sparse-row node count"


def test_engine_proves_optimum_and_root_bounds_are_ordered():
    inst = milp_ne.gen_mdk(30, 5)
    std = milp_ne._std_form(inst)

    prod = milp_ne.solve_discopt(std, milp_ne.ABLATION["prod"], max_nodes=2_000_000)
    assert prod.status == "optimal"
    z_opt = prod.bound

    z_lp = milp_ne.lp_relaxation_bound(std)
    root = milp_ne.root_bound_after(std, milp_ne.ABLATION["prod"])

    # Minimize sense: LP relaxation ≤ root-after-cuts ≤ optimum. Cuts/presolve
    # may not move the bound, hence ≤ (never a crossing — that would be unsound).
    assert np.isfinite(z_lp) and np.isfinite(root) and np.isfinite(z_opt)
    assert z_lp <= root + 1e-6
    assert root <= z_opt + 1e-6

    closed = milp_ne._gap_closed(z_lp, root, z_opt)
    assert closed is not None and -1e-6 <= closed <= 1.0 + 1e-6


def test_ablation_levers_all_run_and_full_helps():
    inst = milp_ne.gen_mdk(40, 5)
    std = milp_ne._std_form(inst)
    nodes = {}
    objs = {}
    for name, cfg in milp_ne.ABLATION.items():
        run = milp_ne.solve_discopt(std, cfg, max_nodes=2_000_000)
        assert run.status == "optimal", f"{name} did not prove optimal"
        nodes[name] = run.nodes
        objs[name] = run.obj
    # Every config must agree on the optimum (soundness: levers never change it).
    ref = objs["baseline"]
    for name, o in objs.items():
        assert abs(o - ref) <= 1e-6 * (1 + abs(ref)), f"{name} objective drifted"
    # The everything-on config should not explore more nodes than the all-off
    # baseline on this instance (the whole point of the levers).
    assert nodes["full"] <= nodes["baseline"]


@pytest.mark.skipif(not milp_ne.scip_available(), reason="pyscipopt not installed")
def test_scip_objective_matches_discopt():
    inst = milp_ne.gen_mdk(30, 5)
    res = milp_ne.run_instance(inst, max_nodes=2_000_000, time_limit_s=30.0, do_scip=True)
    assert res.objectives_match is True
