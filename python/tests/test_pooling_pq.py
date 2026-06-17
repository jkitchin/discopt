"""Tests for the pooling pq-formulation (discopt.pooling).

Two layers:
  * fast structural tests — the builder emits the expected proportion / pq (RLT)
    / linearization constraints (run by default);
  * known-optima validation (marked ``correctness``) — the pq-formulation solves
    the canonical Haverly instance (optimum 400) and a small analytic blending
    instance (optimum 700) to their global optima, at/near the root node because
    the pq relaxation is tight.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest

from discopt.pooling import (
    Input,
    Output,
    Pool,
    PoolingProblem,
    build_pq_formulation,
    haverly_hpp1,
)


def _constraint_names(model) -> list[str]:
    return [getattr(c, "name", "") or "" for c in model._constraints]


# ───────────────────────── structural (fast) ─────────────────────────


def test_qualities_property_dedup_and_order():
    prob = haverly_hpp1()
    assert prob.qualities == ["sulfur"]


def test_pq_formulation_structure():
    """Haverly: one proportion eq, one pq cut per pool->output arc, one w def per
    (input, pool->output)."""
    m = build_pq_formulation(haverly_hpp1())
    names = _constraint_names(m)
    assert sum(n.startswith("prop[") for n in names) == 1
    assert sum(n.startswith("pq[") for n in names) == 2  # pool->p0, pool->p1
    assert sum(n.startswith("def_w[") for n in names) == 4  # {s0,s1} x {p0,p1}
    # Quality-max specs on both products.
    assert sum(n.startswith("qmax[") for n in names) == 2


def test_proportion_variables_bounded_unit_interval():
    m = build_pq_formulation(haverly_hpp1())
    qvars = [v for v in m._variables if v.name.startswith("q[")]
    assert len(qvars) == 2
    for v in qvars:
        assert float(v.lb) == 0.0 and float(v.ub) == 1.0


def test_empty_direct_arcs_default():
    prob = PoolingProblem(
        inputs=[Input("a", 1.0, {"k": 1.0})],
        pools=[Pool("L")],
        outputs=[Output("o", 1.0)],
        pool_inputs=[("a", "L")],
        pool_outputs=[("L", "o")],
    )
    assert prob.direct == []
    # Builds without error and has no direct-flow z variables.
    m = build_pq_formulation(prob)
    assert not any(v.name.startswith("z[") for v in m._variables)


# ───────────────────────── known-optima validation ─────────────────────────


@pytest.mark.correctness
def test_haverly_hpp1_global_optimum():
    """Canonical Haverly Pooling Problem 1 — global optimum is profit 400."""
    m = build_pq_formulation(haverly_hpp1())
    res = m.solve()
    assert res.status == "optimal"
    assert abs(float(res.objective) - 400.0) < 1e-3
    # pq relaxation is tight: the optimum is proven in very few nodes.
    assert res.node_count <= 50


@pytest.mark.correctness
def test_blending_tradeoff_global_optimum():
    """Cheap source is high-sulfur; spec forces a 50/50 blend -> profit 700."""
    prob = PoolingProblem(
        inputs=[
            Input("s0", cost=5.0, quality={"s": 1.0}),
            Input("s1", cost=1.0, quality={"s": 3.0}),
        ],
        pools=[Pool("L")],
        outputs=[Output("p", price=10.0, demand=100.0, quality_max={"s": 2.0})],
        pool_inputs=[("s0", "L"), ("s1", "L")],
        pool_outputs=[("L", "p")],
    )
    res = build_pq_formulation(prob).solve()
    assert res.status == "optimal"
    assert abs(float(res.objective) - 700.0) < 1e-3
