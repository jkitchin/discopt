"""Regression tests for issue #280 — the graph-partition / one-hot MIQP primal gap.

On set-partition / assignment-structured MIQPs (``sum_k x[i,k] == 1``: each item in
exactly one slot) a single bit flip always breaks a one-hot row, so the generic
constraint-violation search, RINS, and local branching make little headway and the
solver returns a *sound but suboptimal* incumbent while the dual bound is already
tight. ``one_hot_swap_search`` adds the feasibility-preserving *swap* move (exchange
two items' slots), which stays on the feasible manifold and closes the primal gap.

These tests pin the heuristic directly (deterministic, fast): the move detection,
the gap-closing improvement, feasibility of the result, and the no-op behaviour on
models without one-hot structure (generality — the move is gated on detected
structure, never on a problem name).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt._jax.primal_heuristics import _detect_one_hot_groups, one_hot_swap_search


def _partition_model(N, K, per, edges):
    """min within-partition edge weight; each node in one partition, balanced."""
    m = dm.Model("gp")
    nodes = m.set("nodes", list(range(N)))
    parts = m.set("parts", list(range(K)))
    x = m.binary("x", over=nodes * parts)
    # fast=False keeps the rows in ``model._constraints`` (where the .nl reader puts
    # them for the real graphpart family, and where the swap detector + solver
    # evaluator read them). The fast-API builder path is a separate concern (#681).
    m.constraint(nodes, lambda i: dm.sum(x[i, k] for k in range(K)) == 1, name="assign", fast=False)
    m.constraint(parts, lambda k: dm.sum(x[i, k] for i in range(N)) == per, name="bal", fast=False)
    m.minimize(dm.sum(w * dm.sum(x[i, k] * x[j, k] for k in range(K)) for (i, j, w) in edges))
    return m, x


def _flat_assignment(N, K, assign):
    """Flat x for the model's single (N, K) binary variable from a per-node slot list."""
    xf = np.zeros(N * K, dtype=np.float64)
    for i, k in enumerate(assign):
        xf[i * K + k] = 1.0
    return xf


# A weighted K4 where the balanced split {0,1}|{2,3} (or {0,3}|{1,2}) costs 2 but the
# split {0,2}|{1,3} costs 20 — one swap of nodes 1 and 2 turns the bad split optimal.
_EDGES = [(0, 2, 10.0), (1, 3, 10.0), (0, 1, 1.0), (2, 3, 1.0), (0, 3, 1.0), (1, 2, 1.0)]


@pytest.mark.smoke
def test_detect_one_hot_groups():
    m, _ = _partition_model(4, 2, 2, _EDGES)
    int_mask = np.ones(m._variables[0].size, dtype=bool)  # single all-binary variable
    groups = _detect_one_hot_groups(m, int_mask, int_mask.size)
    # One group per node (the assignment rows); the balance rows (== 2) are excluded.
    assert len(groups) == 4
    assert all(len(g) == 2 for g in groups)
    # Groups partition the 8 binary slots.
    assert sorted(i for g in groups for i in g) == list(range(8))


@pytest.mark.smoke
def test_swap_closes_gap_from_bad_incumbent():
    m, _ = _partition_model(4, 2, 2, _EDGES)
    ev = cached_evaluator(m)
    bad = _flat_assignment(4, 2, [0, 1, 0, 1])  # {0,2}|{1,3}: within weight 20
    assert abs(float(ev.evaluate_objective(bad)) - 20.0) < 1e-9

    res = one_hot_swap_search(m, bad, evaluator=ev, time_budget=1.0, seed=0)
    assert res is not None, "swap search should improve the bad incumbent"
    x_out, obj = res
    assert obj < 20.0 - 1e-9
    assert abs(obj - 2.0) < 1e-6, f"swap should reach the optimum 2.0, got {obj}"
    # Feasible: each node in one partition, each partition balanced.
    xg = np.asarray(x_out[: 4 * 2]).reshape(4, 2)
    assert np.allclose(xg.sum(axis=1), 1)
    assert np.allclose(xg.sum(axis=0), 2)


@pytest.mark.smoke
def test_swap_noop_when_incumbent_already_optimal():
    m, _ = _partition_model(4, 2, 2, _EDGES)
    ev = cached_evaluator(m)
    good = _flat_assignment(4, 2, [0, 0, 1, 1])  # {0,1}|{2,3}: within weight 2 (optimal)
    assert one_hot_swap_search(m, good, evaluator=ev, time_budget=1.0) is None


@pytest.mark.smoke
def test_swap_noop_without_one_hot_structure():
    """A model with no one-hot rows must self-gate to None (generality / safety)."""
    m = dm.Model("nostruct")
    x = m.binary("x", shape=(3,))
    m.subject_to(dm.sum(x[i] for i in range(3)) <= 2)  # packing, not one-hot ==1
    m.minimize(dm.sum(x[i] * x[i] for i in range(3)))
    ev = cached_evaluator(m)
    assert _detect_one_hot_groups(m, np.ones(3, dtype=bool), 3) == []
    assert one_hot_swap_search(m, np.array([1.0, 1.0, 0.0]), evaluator=ev) is None


@pytest.mark.smoke
def test_swap_end_to_end_reaches_optimum():
    """Full solve: the wired-in swap improver drives the incumbent to the optimum."""
    m, _ = _partition_model(4, 2, 2, _EDGES)
    r = m.solve(time_limit=10, gap_tolerance=1e-4)
    xg = np.asarray(r.x["x"]).reshape(4, 2)
    assert np.allclose(xg.sum(axis=1), 1) and np.allclose(xg.sum(axis=0), 2)
    assert abs(r.objective - 2.0) < 1e-6, f"expected optimum 2.0, got {r.objective}"
