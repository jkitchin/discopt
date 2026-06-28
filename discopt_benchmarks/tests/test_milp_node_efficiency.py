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
