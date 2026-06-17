"""Standard pooling problem and the pq (proportion) formulation.

The **pooling problem** routes raw inputs of known quality through blending
*pools* to outputs with quality specifications, maximizing profit. The blended
quality at a pool is a *bilinear* function of inflows, so the problem is a
nonconvex NLP — small instances already defeat a naive (``p``-formulation)
global solve because its McCormick relaxation is weak.

This module builds the **pq-formulation** of Tawarmalani & Sahinidis (2002),
which uses proportion variables ``q[i,l]`` (the fraction of pool ``l``'s inflow
coming from input ``i``) together with the Reformulation–Linearization cuts

    sum_i w[i,l,j] = y[l,j]          (pq cuts, one per pool->output arc),

where ``w[i,l,j] = q[i,l] * y[l,j]``. These redundant-but-tightening equalities
make the pq relaxation provably at least as tight as the p- and q-formulations,
and on classic instances (e.g. Haverly) the root relaxation is already exact —
the global optimum is found at the root node.

The builder emits a standard :class:`discopt.modeling.core.Model`; the global
machinery (McCormick relaxation + spatial B&B) is reused unchanged, so the
formulation only *tightens* the relaxation and never changes the feasible region
or the optimum.

References
----------
Tawarmalani, M. & Sahinidis, N. V. "Convexification and Global Optimization in
Continuous and Mixed-Integer Nonlinear Programming." Kluwer, 2002.
Haverly, C. A. "Studies of the behavior of recursion for the pooling problem."
ACM SIGMAP Bulletin, 1978.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import discopt.modeling.core as dm
from discopt.modeling.core import Expression, Variable

__all__ = [
    "Input",
    "Pool",
    "Output",
    "PoolingProblem",
    "build_pq_formulation",
    "haverly_hpp1",
]


@dataclass
class Input:
    """A raw input (source) with per-unit cost and quality attributes."""

    name: str
    cost: float
    quality: dict[str, float]
    availability: Optional[float] = None  # max supply; None = unbounded


@dataclass
class Pool:
    """A blending pool with optional throughput capacity."""

    name: str
    capacity: Optional[float] = None


@dataclass
class Output:
    """A product with price, demand cap, and quality bounds."""

    name: str
    price: float
    demand: Optional[float] = None  # max demand; None = unbounded
    quality_max: dict[str, float] = field(default_factory=dict)
    quality_min: dict[str, float] = field(default_factory=dict)


@dataclass
class PoolingProblem:
    """A standard pooling problem over inputs, pools, outputs, and arcs.

    Arcs are given as name pairs:

    - ``pool_inputs``  : ``(input, pool)``  raw input feeds a pool
    - ``pool_outputs`` : ``(pool, output)`` pool blend feeds an output
    - ``direct``       : ``(input, output)`` input bypasses pools to an output
    """

    inputs: list[Input]
    pools: list[Pool]
    outputs: list[Output]
    pool_inputs: list[tuple[str, str]]
    pool_outputs: list[tuple[str, str]]
    direct: list[tuple[str, str]] = field(default_factory=list)
    # Fallback upper bound for flows whose bound is not implied by a capacity or
    # demand (keeps the bilinear McCormick envelopes finite).
    big_m: float = 1.0e4

    @property
    def qualities(self) -> list[str]:
        keys: list[str] = []
        for inp in self.inputs:
            for k in inp.quality:
                if k not in keys:
                    keys.append(k)
        return keys


def _output_flow_ub(problem: PoolingProblem, out: Output) -> float:
    """Finite upper bound on the total flow into an output."""
    return out.demand if out.demand is not None else problem.big_m


def build_pq_formulation(problem: PoolingProblem) -> dm.Model:
    """Build the pq-formulation of ``problem`` as a discopt :class:`Model`.

    Variables (all continuous, non-negative):

    - ``q[i,l]`` in [0, 1] : proportion of pool ``l`` inflow from input ``i``
    - ``y[l,j]``            : flow from pool ``l`` to output ``j``
    - ``z[i,j]``            : direct flow from input ``i`` to output ``j``
    - ``w[i,l,j]``          : linearization of ``q[i,l] * y[l,j]``

    The returned model maximizes ``sum_j price_j * outflow_j - sum_i cost_i *
    usage_i`` subject to the proportion, pq, capacity, availability, demand, and
    quality-blending constraints.
    """
    by_input = {inp.name: inp for inp in problem.inputs}
    by_output = {out.name: out for out in problem.outputs}
    by_pool = {p.name: p for p in problem.pools}

    m = dm.Model("pooling_pq")

    # Adjacency helpers.
    inputs_of_pool: dict[str, list[str]] = {p.name: [] for p in problem.pools}
    for i, ell in problem.pool_inputs:
        inputs_of_pool[ell].append(i)
    outputs_of_pool: dict[str, list[str]] = {p.name: [] for p in problem.pools}
    for ell, j in problem.pool_outputs:
        outputs_of_pool[ell].append(j)

    # --- Variables ---
    q: dict[tuple[str, str], Variable] = {}
    for i, ell in problem.pool_inputs:
        q[(i, ell)] = m.continuous(f"q[{i},{ell}]", lb=0.0, ub=1.0)

    y: dict[tuple[str, str], Variable] = {}
    for ell, j in problem.pool_outputs:
        cap = by_pool[ell].capacity
        dem = _output_flow_ub(problem, by_output[j])
        ub = min(c for c in (cap, dem, problem.big_m) if c is not None)
        y[(ell, j)] = m.continuous(f"y[{ell},{j}]", lb=0.0, ub=ub)

    z: dict[tuple[str, str], Variable] = {}
    for i, j in problem.direct:
        ub = _output_flow_ub(problem, by_output[j])
        z[(i, j)] = m.continuous(f"z[{i},{j}]", lb=0.0, ub=ub)

    # w[i,l,j] = q[i,l] * y[l,j]
    w: dict[tuple[str, str, str], Variable] = {}
    for ell, j in problem.pool_outputs:
        ub_y = min(
            c
            for c in (by_pool[ell].capacity, _output_flow_ub(problem, by_output[j]), problem.big_m)
            if c is not None
        )
        for i in inputs_of_pool[ell]:
            w[(i, ell, j)] = m.continuous(f"w[{i},{ell},{j}]", lb=0.0, ub=ub_y)
            m.subject_to(w[(i, ell, j)] == q[(i, ell)] * y[(ell, j)], name=f"def_w[{i},{ell},{j}]")

    # --- Proportion constraints: sum_i q[i,l] = 1 for each non-empty pool ---
    for ell in by_pool:
        ins = inputs_of_pool[ell]
        if ins:
            m.subject_to(dm.sum([q[(i, ell)] for i in ins]) == 1, name=f"prop[{ell}]")

    # --- pq (RLT) cuts: sum_i w[i,l,j] = y[l,j] ---
    for ell, j in problem.pool_outputs:
        m.subject_to(
            dm.sum([w[(i, ell, j)] for i in inputs_of_pool[ell]]) == y[(ell, j)],
            name=f"pq[{ell},{j}]",
        )

    # --- Pool capacity: sum_j y[l,j] <= S_l ---
    for ell, pool in by_pool.items():
        if pool.capacity is not None and outputs_of_pool[ell]:
            m.subject_to(
                dm.sum([y[(ell, j)] for j in outputs_of_pool[ell]]) <= pool.capacity,
                name=f"cap[{ell}]",
            )

    # input usage[i] = sum_{l,j} w[i,l,j] + sum_j z[i,j]
    def input_usage(i: str) -> Optional[Expression]:
        terms = [w[wk] for wk in w if wk[0] == i]
        terms += [z[zk] for zk in z if zk[0] == i]
        return dm.sum(terms) if terms else None

    # output flow[j] = sum_l y[l,j] + sum_i z[i,j]
    def output_flow(j: str) -> Optional[Expression]:
        terms = [y[yk] for yk in y if yk[1] == j]
        terms += [z[zk] for zk in z if zk[1] == j]
        return dm.sum(terms) if terms else None

    # --- Input availability: usage[i] <= A_i ---
    for inp in problem.inputs:
        if inp.availability is not None:
            usage = input_usage(inp.name)
            if usage is not None:
                m.subject_to(usage <= inp.availability, name=f"avail[{inp.name}]")

    # --- Output demand: flow[j] <= D_j ---
    for out in problem.outputs:
        if out.demand is not None:
            flow = output_flow(out.name)
            if flow is not None:
                m.subject_to(flow <= out.demand, name=f"demand[{out.name}]")

    # --- Quality blending bounds ---
    # blended quality numerator for attribute k at output j:
    #   sum_{l,i} lambda[i,k] w[i,l,j] + sum_i lambda[i,k] z[i,j]
    def quality_numerator(j: str, k: str) -> Optional[Expression]:
        terms = []
        for wk in w:  # (i, l, jj)
            if wk[2] == j:
                lam = by_input[wk[0]].quality.get(k, 0.0)
                terms.append(lam * w[wk])
        for zk in z:  # (i, jj)
            if zk[1] == j:
                lam = by_input[zk[0]].quality.get(k, 0.0)
                terms.append(lam * z[zk])
        return dm.sum(terms) if terms else None

    for out in problem.outputs:
        flow = output_flow(out.name)
        if flow is None:
            continue
        for k in problem.qualities:
            num = quality_numerator(out.name, k)
            if num is None:
                continue
            if k in out.quality_max:
                m.subject_to(num <= out.quality_max[k] * flow, name=f"qmax[{out.name},{k}]")
            if k in out.quality_min:
                m.subject_to(num >= out.quality_min[k] * flow, name=f"qmin[{out.name},{k}]")

    # --- Objective: revenue - cost ---
    revenue_terms = []
    for out in problem.outputs:
        flow = output_flow(out.name)
        if flow is not None:
            revenue_terms.append(out.price * flow)
    cost_terms = []
    for inp in problem.inputs:
        usage = input_usage(inp.name)
        if usage is not None:
            cost_terms.append(inp.cost * usage)
    m.maximize(dm.sum(revenue_terms) - dm.sum(cost_terms))

    return m


def haverly_hpp1() -> PoolingProblem:
    """Haverly Pooling Problem 1 — the canonical pooling instance.

    A single pool blends two sulfur-bearing sources; a third source bypasses the
    pool to the second product. The known global optimum is **profit = 400**.
    """
    inputs = [
        Input("s0", cost=6.0, quality={"sulfur": 3.0}),
        Input("s1", cost=16.0, quality={"sulfur": 1.0}),
        Input("s2", cost=10.0, quality={"sulfur": 2.0}),
    ]
    pools = [Pool("pool")]
    outputs = [
        Output("p0", price=9.0, demand=100.0, quality_max={"sulfur": 2.5}),
        Output("p1", price=15.0, demand=200.0, quality_max={"sulfur": 1.5}),
    ]
    return PoolingProblem(
        inputs=inputs,
        pools=pools,
        outputs=outputs,
        pool_inputs=[("s0", "pool"), ("s1", "pool")],
        pool_outputs=[("pool", "p0"), ("pool", "p1")],
        direct=[("s2", "p1")],
    )
