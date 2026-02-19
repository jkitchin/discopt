"""MILP benchmark problems using the discopt modeling API."""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np

from benchmarks.problems.base import TestProblem, register

# Applicable solvers for MILP problems
_SOLVERS = ["ipm", "ripopt", "ipopt"]


# ─────────────────────────────────────────────────────────────
# Helper: 0-1 knapsack via dynamic programming
# ─────────────────────────────────────────────────────────────


def _knapsack_dp(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> int:
    """Solve 0-1 knapsack exactly via DP. Returns optimal value."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v)
            else:
                dp[i][c] = dp[i - 1][c]
    return dp[n][capacity]


# ─────────────────────────────────────────────────────────────
# Helper: build a knapsack Model
# ─────────────────────────────────────────────────────────────


def _build_knapsack(
    name: str,
    weights: list[int],
    values: list[int],
    capacity: int,
) -> dm.Model:
    """Build a 0-1 knapsack as minimize(-value)."""
    n = len(weights)
    m = dm.Model(name)
    x = [m.binary(f"x{i}") for i in range(n)]
    obj = dm.sum([
        -values[i] * x[i] for i in range(n)
    ])
    m.minimize(obj)
    m.subject_to(
        dm.sum([weights[i] * x[i] for i in range(n)]) <= capacity,
        name="capacity",
    )
    return m


# ═════════════════════════════════════════════════════════════
# SMOKE problems (5)
# ═════════════════════════════════════════════════════════════


# 1. milp_knapsack_5 ─────────────────────────────────────────
def _build_knapsack_5() -> dm.Model:
    weights = [2, 3, 4, 5, 9]
    values = [3, 4, 8, 8, 10]
    capacity = 20
    return _build_knapsack("knapsack_5", weights, values, capacity)


register(TestProblem(
    name="milp_knapsack_5",
    category="milp",
    level="smoke",
    build_fn=_build_knapsack_5,
    known_optimum=-29.0,  # items 0,2,3,4: w=20, v=29
    applicable_solvers=_SOLVERS,
    n_vars=5,
    n_constraints=1,
    tags=["knapsack", "binary"],
))


# 2. milp_assignment_3x3 ─────────────────────────────────────
def _build_assignment_3x3() -> dm.Model:
    cost = [[7, 3, 5], [2, 8, 4], [6, 4, 3]]
    m = dm.Model("assignment_3x3")
    x = [
        [m.binary(f"x_{i}_{j}") for j in range(3)]
        for i in range(3)
    ]
    # Minimize total assignment cost
    obj = dm.sum([
        cost[i][j] * x[i][j]
        for i in range(3)
        for j in range(3)
    ])
    m.minimize(obj)
    # Each worker assigned to exactly one job
    for i in range(3):
        m.subject_to(
            dm.sum([x[i][j] for j in range(3)]) == 1,
            name=f"worker_{i}",
        )
    # Each job assigned to exactly one worker
    for j in range(3):
        m.subject_to(
            dm.sum([x[i][j] for i in range(3)]) == 1,
            name=f"job_{j}",
        )
    return m


register(TestProblem(
    name="milp_assignment_3x3",
    category="milp",
    level="smoke",
    build_fn=_build_assignment_3x3,
    known_optimum=8.0,  # 0->1(3), 1->0(2), 2->2(3)
    applicable_solvers=_SOLVERS,
    n_vars=9,
    n_constraints=6,
    tags=["assignment", "binary"],
))


# 3. milp_set_cover_small ────────────────────────────────────
def _build_set_cover_small() -> dm.Model:
    # 4 sets covering 6 elements
    costs = [1, 2, 3, 1]
    # sets[j] = list of elements covered by set j
    sets = [{0, 1, 2}, {2, 3, 4}, {4, 5}, {0, 3, 5}]
    n_elements = 6
    n_sets = 4
    m = dm.Model("set_cover_small")
    y = [m.binary(f"y{j}") for j in range(n_sets)]
    m.minimize(dm.sum([costs[j] * y[j] for j in range(n_sets)]))
    # Each element must be covered by at least one selected set
    for e in range(n_elements):
        covering = [y[j] for j in range(n_sets) if e in sets[j]]
        m.subject_to(dm.sum(covering) >= 1, name=f"cover_{e}")
    return m


register(TestProblem(
    name="milp_set_cover_small",
    category="milp",
    level="smoke",
    build_fn=_build_set_cover_small,
    known_optimum=4.0,  # S0 + S2: cost 1+3=4
    applicable_solvers=_SOLVERS,
    n_vars=4,
    n_constraints=6,
    tags=["set_cover", "binary"],
))


# 4. milp_fixed_charge_2 ─────────────────────────────────────
def _build_fixed_charge_2() -> dm.Model:
    m = dm.Model("fixed_charge_2")
    y1 = m.binary("y1")
    y2 = m.binary("y2")
    x1 = m.continuous("x1", lb=0.0, ub=10.0)
    x2 = m.continuous("x2", lb=0.0, ub=10.0)
    m.minimize(10 * y1 + 15 * y2 + x1 + 2 * x2)
    m.subject_to(x1 + x2 >= 8, name="demand")
    m.subject_to(x1 <= 10 * y1, name="link_1")
    m.subject_to(x2 <= 10 * y2, name="link_2")
    return m


register(TestProblem(
    name="milp_fixed_charge_2",
    category="milp",
    level="smoke",
    build_fn=_build_fixed_charge_2,
    known_optimum=18.0,  # y1=1, x1=8, y2=0, x2=0
    applicable_solvers=_SOLVERS,
    n_vars=4,
    n_constraints=3,
    tags=["fixed_charge", "mixed_binary"],
))


# 5. milp_infeasible ─────────────────────────────────────────
def _build_infeasible() -> dm.Model:
    m = dm.Model("milp_infeasible")
    x = m.binary("x")
    m.minimize(x)
    m.subject_to(x >= 2, name="impossible")
    return m


register(TestProblem(
    name="milp_infeasible",
    category="milp",
    level="smoke",
    build_fn=_build_infeasible,
    known_optimum=float("inf"),
    applicable_solvers=_SOLVERS,
    n_vars=1,
    n_constraints=1,
    tags=["infeasible"],
    expected_status="infeasible",
))


# ═════════════════════════════════════════════════════════════
# FULL problems (+15)
# ═════════════════════════════════════════════════════════════


# 6. milp_knapsack_10 ────────────────────────────────────────
def _build_knapsack_10() -> dm.Model:
    rng = np.random.default_rng(42)
    weights = rng.integers(1, 15, size=10).tolist()
    values = rng.integers(1, 20, size=10).tolist()
    capacity = int(sum(weights) * 0.5)
    return _build_knapsack("knapsack_10", weights, values, capacity)


_kn10_rng = np.random.default_rng(42)
_kn10_w = _kn10_rng.integers(1, 15, size=10).tolist()
_kn10_v = _kn10_rng.integers(1, 20, size=10).tolist()
_kn10_cap = int(sum(_kn10_w) * 0.5)
_kn10_opt = _knapsack_dp(_kn10_w, _kn10_v, _kn10_cap)

register(TestProblem(
    name="milp_knapsack_10",
    category="milp",
    level="full",
    build_fn=_build_knapsack_10,
    known_optimum=-float(_kn10_opt),
    applicable_solvers=_SOLVERS,
    n_vars=10,
    n_constraints=1,
    tags=["knapsack", "binary"],
))


# 7. milp_knapsack_20 ────────────────────────────────────────
def _build_knapsack_20() -> dm.Model:
    rng = np.random.default_rng(123)
    weights = rng.integers(1, 20, size=20).tolist()
    values = rng.integers(1, 30, size=20).tolist()
    capacity = int(sum(weights) * 0.4)
    return _build_knapsack("knapsack_20", weights, values, capacity)


_kn20_rng = np.random.default_rng(123)
_kn20_w = _kn20_rng.integers(1, 20, size=20).tolist()
_kn20_v = _kn20_rng.integers(1, 30, size=20).tolist()
_kn20_cap = int(sum(_kn20_w) * 0.4)
_kn20_opt = _knapsack_dp(_kn20_w, _kn20_v, _kn20_cap)

register(TestProblem(
    name="milp_knapsack_20",
    category="milp",
    level="full",
    build_fn=_build_knapsack_20,
    known_optimum=-float(_kn20_opt),
    applicable_solvers=_SOLVERS,
    n_vars=20,
    n_constraints=1,
    tags=["knapsack", "binary"],
))


# 8. milp_knapsack_50 ────────────────────────────────────────
def _build_knapsack_50() -> dm.Model:
    rng = np.random.default_rng(999)
    weights = rng.integers(1, 25, size=50).tolist()
    values = rng.integers(1, 40, size=50).tolist()
    capacity = int(sum(weights) * 0.35)
    return _build_knapsack("knapsack_50", weights, values, capacity)


_kn50_rng = np.random.default_rng(999)
_kn50_w = _kn50_rng.integers(1, 25, size=50).tolist()
_kn50_v = _kn50_rng.integers(1, 40, size=50).tolist()
_kn50_cap = int(sum(_kn50_w) * 0.35)
_kn50_opt = _knapsack_dp(_kn50_w, _kn50_v, _kn50_cap)

register(TestProblem(
    name="milp_knapsack_50",
    category="milp",
    level="full",
    build_fn=_build_knapsack_50,
    known_optimum=-float(_kn50_opt),
    applicable_solvers=_SOLVERS,
    n_vars=50,
    n_constraints=1,
    tags=["knapsack", "binary"],
))


# 9. milp_facility_3x5 ──────────────────────────────────────
def _build_facility_3x5() -> dm.Model:
    """3 facilities, 5 customers. TU assignment constraints."""
    # Fixed costs for opening facilities
    f_cost = [10, 15, 12]
    # Transport cost[i][j]: facility i -> customer j
    t_cost = [
        [4, 6, 9, 5, 7],
        [5, 3, 8, 6, 4],
        [6, 5, 3, 8, 6],
    ]
    n_fac, n_cust = 3, 5
    m = dm.Model("facility_3x5")
    y = [m.binary(f"open_{i}") for i in range(n_fac)]
    x = [
        [m.binary(f"assign_{i}_{j}") for j in range(n_cust)]
        for i in range(n_fac)
    ]
    # Objective: fixed + transport
    obj = dm.sum([f_cost[i] * y[i] for i in range(n_fac)])
    obj = obj + dm.sum([
        t_cost[i][j] * x[i][j]
        for i in range(n_fac)
        for j in range(n_cust)
    ])
    m.minimize(obj)
    # Each customer served by exactly one facility
    for j in range(n_cust):
        m.subject_to(
            dm.sum([x[i][j] for i in range(n_fac)]) == 1,
            name=f"serve_{j}",
        )
    # Can only assign to open facilities
    for i in range(n_fac):
        for j in range(n_cust):
            m.subject_to(
                x[i][j] <= y[i],
                name=f"link_{i}_{j}",
            )
    return m


# Opt: open fac 2 only (cost 12), all customers -> fac 2:
# t_cost[2] = [6,5,3,8,6], transport=28, total=12+28=40.
# Open 0 only (cost 10): transport=4+6+9+5+7=31, total=41.
# Open 1 only (cost 15): transport=5+3+8+6+4=26, total=41.
# Open 1+2 (cost 27): c0->1(5),c1->1(3),c2->2(3),c3->1(6),
# c4->1(4) = 27+21=48. Worse than single fac 2.
# Opt = 40.
register(TestProblem(
    name="milp_facility_3x5",
    category="milp",
    level="full",
    build_fn=_build_facility_3x5,
    known_optimum=40.0,
    applicable_solvers=_SOLVERS,
    n_vars=18,  # 3 y + 15 x
    n_constraints=20,  # 5 serve + 15 link
    tags=["facility_location", "binary"],
))


# 10. milp_scheduling_5x2 ────────────────────────────────────
def _build_scheduling_5x2() -> dm.Model:
    """5 jobs on 2 machines, minimize makespan.

    Processing times: p[j][k] = time for job j on machine k.
    Binary x[j][k] = 1 if job j assigned to machine k.
    Continuous C = makespan.
    """
    p = [[3, 5], [4, 3], [2, 6], [5, 2], [4, 4]]
    n_jobs, n_mach = 5, 2
    m = dm.Model("scheduling_5x2")
    x = [
        [m.binary(f"x_{j}_{k}") for k in range(n_mach)]
        for j in range(n_jobs)
    ]
    makespan = m.continuous("makespan", lb=0.0, ub=100.0)
    m.minimize(makespan)
    # Each job on exactly one machine
    for j in range(n_jobs):
        m.subject_to(
            dm.sum([x[j][k] for k in range(n_mach)]) == 1,
            name=f"assign_{j}",
        )
    # Makespan >= load on each machine
    for k in range(n_mach):
        load = dm.sum([p[j][k] * x[j][k] for j in range(n_jobs)])
        m.subject_to(load <= makespan, name=f"makespan_{k}")
    return m


# Opt: assign jobs to balance load.
# Machine 0: jobs 0(3),2(2),4(4) = 9
# Machine 1: jobs 1(3),3(2) = 5
# Makespan = 9. Can we do better?
# Machine 0: jobs 0(3),1(4) = 7; Machine 1: jobs 2(6),3(2),4(4) = 12
# Machine 0: jobs 0(3),1(4),3(5) = 12; Machine 1: jobs 2(6),4(4) = 10
# Machine 0: jobs 0(3),2(2),4(4) = 9; Machine 1: jobs 1(3),3(2) = 5 -> 9
# Machine 0: jobs 1(4),2(2) = 6; Machine 1: jobs 0(5),3(2),4(4) = 11
# Machine 0: jobs 0(3),1(4),2(2) = 9; Machine 1: jobs 3(2),4(4) = 6 -> 9
# Machine 0: jobs 0(3),3(5) = 8; Machine 1: jobs 1(3),2(6),4(4) = 13
# Machine 0: jobs 2(2),3(5) = 7; Machine 1: jobs 0(5),1(3),4(4) = 12
# Machine 0: jobs 0(3),1(4),2(2),3(5) = 14; Machine 1: 4(4) = 4
# Best so far: 9 (multiple assignments).
# Machine 0: jobs 3(5),4(4)=9; Machine 1: jobs 0(5),1(3),2(6)=14
# Machine 0: jobs 0(3),2(2),3(5)=10; Machine 1: jobs 1(3),4(4)=7 -> 10
# Machine 0: jobs 1(4),3(5)=9; Machine 1: jobs 0(5),2(6),4(4)=15
# Makespan 9 appears optimal.
register(TestProblem(
    name="milp_scheduling_5x2",
    category="milp",
    level="full",
    build_fn=_build_scheduling_5x2,
    known_optimum=9.0,
    applicable_solvers=_SOLVERS,
    n_vars=11,  # 10 binary + 1 continuous
    n_constraints=7,  # 5 assign + 2 makespan
    tags=["scheduling", "makespan", "mixed_binary"],
))


# 11. milp_bin_packing_8x3 ───────────────────────────────────
def _build_bin_packing_8x3() -> dm.Model:
    """8 items, 3 bins, capacity 15 each. Minimize bins used."""
    sizes = [6, 6, 5, 5, 4, 4, 3, 2]
    n_items, n_bins = 8, 3
    cap = 15
    m = dm.Model("bin_packing_8x3")
    y = [m.binary(f"bin_{k}") for k in range(n_bins)]
    x = [
        [m.binary(f"item_{i}_bin_{k}") for k in range(n_bins)]
        for i in range(n_items)
    ]
    m.minimize(dm.sum([y[k] for k in range(n_bins)]))
    # Each item in exactly one bin
    for i in range(n_items):
        m.subject_to(
            dm.sum([x[i][k] for k in range(n_bins)]) == 1,
            name=f"pack_{i}",
        )
    # Capacity constraints
    for k in range(n_bins):
        m.subject_to(
            dm.sum([
                sizes[i] * x[i][k] for i in range(n_items)
            ]) <= cap * y[k],
            name=f"cap_{k}",
        )
    return m


# Total size=35, 3 bins of cap 15 = 45. Need ceil(35/15)=3 bins.
# Bin 0: 6+6+3=15, Bin 1: 5+5+4=14, Bin 2: 4+2=6 -> 3 bins.
register(TestProblem(
    name="milp_bin_packing_8x3",
    category="milp",
    level="full",
    build_fn=_build_bin_packing_8x3,
    known_optimum=3.0,
    applicable_solvers=_SOLVERS,
    n_vars=27,  # 3 y + 24 x
    n_constraints=11,  # 8 pack + 3 cap
    tags=["bin_packing", "binary"],
))


# 12. milp_lot_sizing_3 ──────────────────────────────────────
def _build_lot_sizing_3() -> dm.Model:
    """Uncapacitated lot sizing, 3 periods.

    demand d=[5,8,3], setup cost f=[10,10,10],
    holding cost h=[1,1,1], production cost c=[2,2,2].
    Binary y_t=1 if we produce in period t.
    Continuous x_t = production, s_t = inventory.
    """
    d = [5, 8, 3]
    f = [10, 10, 10]
    h = [1, 1, 1]
    c = [2, 2, 2]
    n_periods = 3
    max_prod = sum(d)  # max possible production
    m = dm.Model("lot_sizing_3")
    y = [m.binary(f"setup_{t}") for t in range(n_periods)]
    x = [
        m.continuous(f"prod_{t}", lb=0.0, ub=float(max_prod))
        for t in range(n_periods)
    ]
    s = [
        m.continuous(f"inv_{t}", lb=0.0, ub=float(max_prod))
        for t in range(n_periods)
    ]
    obj = dm.sum([
        f[t] * y[t] + c[t] * x[t] + h[t] * s[t]
        for t in range(n_periods)
    ])
    m.minimize(obj)
    # Flow balance: s_{t-1} + x_t = d_t + s_t
    # s_{-1} = 0
    for t in range(n_periods):
        prev_inv = s[t - 1] if t > 0 else 0.0
        m.subject_to(
            prev_inv + x[t] == d[t] + s[t],
            name=f"balance_{t}",
        )
    # Setup linking
    for t in range(n_periods):
        m.subject_to(
            x[t] <= max_prod * y[t],
            name=f"setup_link_{t}",
        )
    return m


# Opt: produce all in period 0: y0=1, x0=16, s0=11, s1=3, s2=0.
# Cost = 10 + 2*16 + 1*11 + 1*3 + 0 = 10+32+11+3 = 56.
# Or produce each period: y=1,1,1, x=5,8,3, s=0,0,0.
# Cost = 30 + 2*16 + 0 = 30+32 = 62.
# Produce periods 0,1: y=1,1,0, x=5,11,0, s=0,3,0.
# Cost = 20+2*16+3 = 55.
# Produce period 0 only: cost = 10+32+11+3 = 56.
# Produce periods 0,1: x0=5,s0=0; x1=11,s1=3; x2=0,s2=0.
# Cost = 10+10 + 2*5+2*11 + 0+3+0 = 20+32+3 = 55.
register(TestProblem(
    name="milp_lot_sizing_3",
    category="milp",
    level="full",
    build_fn=_build_lot_sizing_3,
    known_optimum=55.0,
    applicable_solvers=_SOLVERS,
    n_vars=9,  # 3 y + 3 x + 3 s
    n_constraints=6,  # 3 balance + 3 setup
    tags=["lot_sizing", "mixed_binary"],
))


# 13. milp_capital_budgeting_8 ───────────────────────────────
def _build_capital_budgeting_8() -> dm.Model:
    """Select from 8 projects to maximize NPV under budget.

    NPV and cost are chosen so the LP relaxation is tight
    (optimal LP solution is integral).
    """
    npv = [8, 11, 6, 4, 7, 13, 10, 5]
    cost = [5, 7, 4, 3, 4, 8, 6, 3]
    budget = 20
    n = 8
    m = dm.Model("capital_budgeting_8")
    x = [m.binary(f"proj_{i}") for i in range(n)]
    m.minimize(dm.sum([-npv[i] * x[i] for i in range(n)]))
    m.subject_to(
        dm.sum([cost[i] * x[i] for i in range(n)]) <= budget,
        name="budget",
    )
    return m


# DP for knapsack: weights=cost, values=npv, cap=20.
_cap8_opt = _knapsack_dp(
    [5, 7, 4, 3, 4, 8, 6, 3],
    [8, 11, 6, 4, 7, 13, 10, 5],
    20,
)

register(TestProblem(
    name="milp_capital_budgeting_8",
    category="milp",
    level="full",
    build_fn=_build_capital_budgeting_8,
    known_optimum=-float(_cap8_opt),
    applicable_solvers=_SOLVERS,
    n_vars=8,
    n_constraints=1,
    tags=["capital_budgeting", "binary"],
))


# 14. milp_transportation_2x3 ────────────────────────────────
def _build_transportation_2x3() -> dm.Model:
    """2 suppliers, 3 customers, integer flows.

    Supply=[20,30], demand=[15,10,20].
    Cost=[[2,4,5],[3,1,3]].
    This is a transportation problem (TU constraint matrix),
    so LP relaxation gives integer optimum.
    """
    supply = [20, 30]
    demand = [15, 10, 20]
    cost = [[2, 4, 5], [3, 1, 3]]
    n_sup, n_cust = 2, 3
    m = dm.Model("transportation_2x3")
    x = [
        [
            m.integer(
                f"x_{i}_{j}", lb=0, ub=min(supply[i], demand[j])
            )
            for j in range(n_cust)
        ]
        for i in range(n_sup)
    ]
    obj = dm.sum([
        cost[i][j] * x[i][j]
        for i in range(n_sup)
        for j in range(n_cust)
    ])
    m.minimize(obj)
    # Supply constraints (<=)
    for i in range(n_sup):
        m.subject_to(
            dm.sum([x[i][j] for j in range(n_cust)]) <= supply[i],
            name=f"supply_{i}",
        )
    # Demand constraints (>=)
    for j in range(n_cust):
        m.subject_to(
            dm.sum([x[i][j] for i in range(n_sup)]) >= demand[j],
            name=f"demand_{j}",
        )
    return m


# Opt: x[0][0]=15, x[0][1]=5, x[0][2]=0, x[1][0]=0, x[1][1]=5,
# x[1][2]=20. Cost = 30+20+0+0+5+60 = 115.
# Or: x[0][0]=15, x[0][1]=0, x[0][2]=5, x[1][0]=0, x[1][1]=10,
# x[1][2]=15. Cost = 30+0+25+0+10+45 = 110.
# Or: x[0][0]=15, x[0][1]=0, x[0][2]=0, x[1][0]=0, x[1][1]=10,
# x[1][2]=20. Supply 0 used=15<=20, supply 1 used=30<=30.
# Cost = 30+0+0+0+10+60 = 100.
# Check: x[0][0]=15(cost30), x[1][1]=10(cost10), x[1][2]=20(cost60).
# Supply 0: 15<=20 ok. Supply 1: 30<=30 ok. Total=100.
# Can we do better? x[0][0]=15, x[0][1]=5, x[1][1]=5, x[1][2]=20.
# S0=20, S1=25. Cost=30+20+5+60=115. No.
# x[0][0]=15, x[1][1]=10, x[0][2]=5, x[1][2]=15.
# S0=20, S1=25. Cost=30+10+25+45=110. No.
# 100 looks optimal.
register(TestProblem(
    name="milp_transportation_2x3",
    category="milp",
    level="full",
    build_fn=_build_transportation_2x3,
    known_optimum=100.0,
    applicable_solvers=_SOLVERS,
    n_vars=6,
    n_constraints=5,  # 2 supply + 3 demand
    tags=["transportation", "integer", "network"],
))


# 15. milp_assignment_5x5 ────────────────────────────────────
def _build_assignment_5x5() -> dm.Model:
    """5x5 assignment, TU so LP=IP."""
    cost = [
        [9, 2, 7, 8, 6],
        [6, 4, 3, 7, 5],
        [5, 8, 1, 8, 9],
        [7, 6, 9, 4, 3],
        [8, 3, 8, 9, 7],
    ]
    n = 5
    m = dm.Model("assignment_5x5")
    x = [
        [m.binary(f"x_{i}_{j}") for j in range(n)]
        for i in range(n)
    ]
    obj = dm.sum([
        cost[i][j] * x[i][j]
        for i in range(n)
        for j in range(n)
    ])
    m.minimize(obj)
    for i in range(n):
        m.subject_to(
            dm.sum([x[i][j] for j in range(n)]) == 1,
            name=f"row_{i}",
        )
    for j in range(n):
        m.subject_to(
            dm.sum([x[i][j] for i in range(n)]) == 1,
            name=f"col_{j}",
        )
    return m


# Use Hungarian method mentally:
# 0->1(2), 1->2(3), 2->0(5), 3->4(3), 4->1? No, 1 taken.
# 0->1(2), 1->2(3), 2->2? No.
# Try: 0->1(2), 2->2(1), 3->4(3), 4->1? No.
# 0->1(2), 2->2(1), 3->4(3), need 1,4 assigned to 0,3.
# 1->0(6), 4->3(9) = 6+9=15. Total=2+1+3+15=21. Or
# 1->3(7), 4->0(8) = 15. Total=2+1+3+15=21.
# Try 0->1(2), 2->2(1), 1->0(6), 3->4(3), 4->3(9)=21.
# Try 0->1(2), 1->2(3), 2->0(5), 3->4(3), 4->3(9)? No,
# col 3 used by 4 already for col 3? 3->4=col4, 4->3=col3.
# That's 2+3+5+3+9=22.
# 0->1(2), 2->2(1), 1->0(6), 3->4(3), 4->3(9) = 21.
# 0->1(2), 2->2(1), 1->4(5), 3->3(4), 4->0(8) = 20.
# 0->1(2), 2->2(1), 1->4(5), 4->0(8), 3->3(4) = 20.
# Can we do 19? 0->1(2), 2->2(1), 3->4(3),
# need 1,4 to 0,3: 1->0(6)+4->3(9)=15 => 21.
# 1->3(7)+4->0(8)=15 => 21.
# 0->1(2), 2->2(1), 1->4(5), 3->3(4), 4->0(8) = 20.
# 0->4(6), 2->2(1), 1->1(4), 3->3(4), 4->0(8) = 23. Worse.
# Opt = 20.
register(TestProblem(
    name="milp_assignment_5x5",
    category="milp",
    level="full",
    build_fn=_build_assignment_5x5,
    known_optimum=20.0,
    applicable_solvers=_SOLVERS,
    n_vars=25,
    n_constraints=10,
    tags=["assignment", "binary"],
))


# 16. milp_shortest_path_dag ─────────────────────────────────
def _build_shortest_path_dag() -> dm.Model:
    """Shortest path in a 5-node DAG (0->4) with binary arc vars.

    Arcs (with costs): 0->1(3), 0->2(5), 1->2(1), 1->3(6),
    2->3(2), 2->4(7), 3->4(1).
    """
    arcs = [
        (0, 1, 3), (0, 2, 5), (1, 2, 1), (1, 3, 6),
        (2, 3, 2), (2, 4, 7), (3, 4, 1),
    ]
    n_nodes = 5
    source, sink = 0, 4
    m = dm.Model("shortest_path_dag")
    x = [m.binary(f"arc_{u}_{v}") for u, v, _ in arcs]
    m.minimize(dm.sum([c * x[k] for k, (_, _, c) in enumerate(arcs)]))
    # Flow conservation
    for node in range(n_nodes):
        out_arcs = [
            x[k] for k, (u, _, _) in enumerate(arcs) if u == node
        ]
        in_arcs = [
            x[k] for k, (_, v, _) in enumerate(arcs) if v == node
        ]
        if node == source:
            rhs = 1.0
        elif node == sink:
            rhs = -1.0
        else:
            rhs = 0.0
        out_sum = dm.sum(out_arcs) if out_arcs else 0.0
        in_sum = dm.sum(in_arcs) if in_arcs else 0.0
        m.subject_to(
            out_sum - in_sum == rhs,
            name=f"flow_{node}",
        )
    return m


# Shortest path: 0->1(3)->2(1)->3(2)->4(1) = 7.
# Or: 0->2(5)->3(2)->4(1) = 8. Or 0->1(3)->3(6)->4(1) = 10.
# Opt = 7.
register(TestProblem(
    name="milp_shortest_path_dag",
    category="milp",
    level="full",
    build_fn=_build_shortest_path_dag,
    known_optimum=7.0,
    applicable_solvers=_SOLVERS,
    n_vars=7,
    n_constraints=5,
    tags=["shortest_path", "network", "binary"],
))


# 17. milp_max_flow_4 ────────────────────────────────────────
def _build_max_flow_4() -> dm.Model:
    """Max flow from node 0 to node 3, 4 nodes.

    Arcs (capacities): 0->1(10), 0->2(8), 1->2(5),
    1->3(7), 2->3(10).
    Formulated as min(-flow_value) with integer flows.
    """
    arcs = [
        (0, 1, 10), (0, 2, 8), (1, 2, 5),
        (1, 3, 7), (2, 3, 10),
    ]
    n_nodes = 4
    source, sink = 0, 3
    m = dm.Model("max_flow_4")
    x = [
        m.integer(f"f_{u}_{v}", lb=0, ub=cap)
        for u, v, cap in arcs
    ]
    # Maximize flow = minimize -flow_out_of_source
    out_source = [
        x[k] for k, (u, _, _) in enumerate(arcs) if u == source
    ]
    m.minimize(-1 * dm.sum(out_source))
    # Flow conservation for internal nodes
    for node in range(n_nodes):
        if node in (source, sink):
            continue
        out_arcs = [
            x[k] for k, (u, _, _) in enumerate(arcs) if u == node
        ]
        in_arcs = [
            x[k] for k, (_, v, _) in enumerate(arcs) if v == node
        ]
        m.subject_to(
            dm.sum(out_arcs) == dm.sum(in_arcs),
            name=f"conserve_{node}",
        )
    return m


# Max flow: 0->1(10), 0->2(8). Node 1: in=10, out=1->2+1->3.
# 1->3(7), 1->2(3), node 2: in=8+3=11, out=2->3(10). Excess 1.
# 1->2(5), 1->3(5), node 2: in=8+5=13, out=10. Need 13<=10? No.
# 0->1(10), 0->2(8)=18 total out. Node 1: 1->3(7),1->2(3).
# Node 2: in=8+3=11, out=min(11,10)=10. Sink=7+10=17. Out=18.
# But node 1 out=10, node 2 can only push 10. So flow=17.
# Actually: 0->1(10), 0->2(7). 1->3(7), 1->2(3). 2: 7+3=10, 2->3=10.
# Flow = 17. 0 out = 17. Check: 0->1(10)+0->2(7)=17.
register(TestProblem(
    name="milp_max_flow_4",
    category="milp",
    level="full",
    build_fn=_build_max_flow_4,
    known_optimum=-17.0,
    applicable_solvers=_SOLVERS,
    n_vars=5,
    n_constraints=2,  # 2 internal nodes
    tags=["max_flow", "network", "integer"],
))


# 18. milp_set_cover_8 ──────────────────────────────────────
def _build_set_cover_8() -> dm.Model:
    """8 sets covering 10 elements."""
    costs = [3, 2, 4, 1, 5, 2, 3, 1]
    sets = [
        {0, 1, 2},       # S0
        {2, 3, 4},       # S1
        {4, 5, 6},       # S2
        {6, 7},          # S3
        {0, 3, 5, 8},    # S4
        {1, 7, 8, 9},    # S5
        {5, 6, 9},       # S6
        {0, 9},          # S7
    ]
    n_elements = 10
    n_sets = 8
    m = dm.Model("set_cover_8")
    y = [m.binary(f"y{j}") for j in range(n_sets)]
    m.minimize(dm.sum([costs[j] * y[j] for j in range(n_sets)]))
    for e in range(n_elements):
        covering = [y[j] for j in range(n_sets) if e in sets[j]]
        m.subject_to(dm.sum(covering) >= 1, name=f"cover_{e}")
    return m


# S0={0,1,2}(3), S1={2,3,4}(2), S2={4,5,6}(4), S3={6,7}(1),
# S5={1,7,8,9}(2), S7={0,9}(1).
# Try S1(2)+S3(1)+S5(2)+S7(1) = 6: covers {2,3,4,6,7,1,8,9,0}
# Missing: {5}. Need S2(4) or S4(5) or S6(3). Add S6(3): total=9.
# S0(3)+S1(2)+S3(1)+S5(2)+S6(3) = 11:
# covers {0,1,2,3,4,6,7,8,9,5,6,9}=all. Cost=11.
# S1(2)+S2(4)+S3(1)+S5(2)+S7(1) = 10:
# {2,3,4,5,6,7,1,8,9,0}=all. Cost=10.
# S1(2)+S3(1)+S5(2)+S6(3)+S7(1) = 9:
# {2,3,4,6,7,1,8,9,5,6,9,0}=all. Cost=9.
# S1(2)+S3(1)+S4(5)+S5(2) = 10: {2,3,4,6,7,0,5,8,1,9}=all. 10.
# S1(2)+S3(1)+S5(2)+S6(3)+S7(1)=9 covers all 10 elements.
# Can we do 8? S1(2)+S5(2)+S6(3)+S7(1)=8:
# {2,3,4,1,7,8,9,5,6,0}=all! Cost=8.
# Can we do 7? Need to check... 4 sets at avg cost <2.
# S3(1)+S5(2)+S7(1) = 4: {6,7,1,8,9,0}. Missing {2,3,4,5}.
# Add S1(2)=6: {2,3,4} gained, missing {5}. Need S2/S4/S6.
# S3(1)+S5(2)+S7(1)+S1(2)+S6(3)=9. Worse than 8.
# S1(2)+S5(2)+S6(3)+S7(1)=8 appears optimal.
register(TestProblem(
    name="milp_set_cover_8",
    category="milp",
    level="full",
    build_fn=_build_set_cover_8,
    known_optimum=8.0,
    applicable_solvers=_SOLVERS,
    n_vars=8,
    n_constraints=10,
    tags=["set_cover", "binary"],
))


# 19. milp_fixed_charge_4 ────────────────────────────────────
def _build_fixed_charge_4() -> dm.Model:
    """4 arcs with fixed charges. Satisfy demand of 15."""
    fixed = [8, 12, 6, 10]
    unit = [1, 2, 3, 1]
    caps = [10, 10, 10, 10]
    m = dm.Model("fixed_charge_4")
    y = [m.binary(f"y{i}") for i in range(4)]
    x = [
        m.continuous(f"x{i}", lb=0.0, ub=float(caps[i]))
        for i in range(4)
    ]
    obj = dm.sum([
        fixed[i] * y[i] + unit[i] * x[i] for i in range(4)
    ])
    m.minimize(obj)
    m.subject_to(
        dm.sum([x[i] for i in range(4)]) >= 15,
        name="demand",
    )
    for i in range(4):
        m.subject_to(
            x[i] <= caps[i] * y[i],
            name=f"link_{i}",
        )
    return m


# Open arc 0 (fixed=8,unit=1): x0=10, cost=8+10=18.
# Open arc 3 (fixed=10,unit=1): x3=5, cost=10+5=15.
# Total=33.
# Open arc 2 (fixed=6,unit=3): x2=5, cost=6+15=21.
# Open 0+3: 8+10+10+5=33.
# Open 0+2: 8+6+10+15=39. Worse.
# Open 0 only (cap 10 < 15): need another.
# Open 0+3: x0=10,x3=5. Cost=8+10+10+5=33.
# Open 0+1: x0=10,x1=5. Cost=8+10+12+10=40.
# Open 2+3: x2=10,x3=5. Cost=6+30+10+5=51. Worse.
# Open 0+3 at 33 is best. Can we do 2 arcs cheaper?
# Only open 3 (cap 10<15): nope.
# 33 appears optimal.
register(TestProblem(
    name="milp_fixed_charge_4",
    category="milp",
    level="full",
    build_fn=_build_fixed_charge_4,
    known_optimum=33.0,
    applicable_solvers=_SOLVERS,
    n_vars=8,  # 4 binary + 4 continuous
    n_constraints=5,  # 1 demand + 4 link
    tags=["fixed_charge", "mixed_binary"],
))


# 20. milp_warehouse_2x4 ─────────────────────────────────────
def _build_warehouse_2x4() -> dm.Model:
    """2 warehouses, 4 customers. Open warehouses + assignment.

    Warehouse caps=[15,20], fixed=[20,25].
    Demands=[5,8,6,4]. Transport cost:
    [[2,4,5,3],[3,1,2,6]].
    """
    w_cap = [15, 20]
    w_fixed = [20, 25]
    demand = [5, 8, 6, 4]
    t_cost = [[2, 4, 5, 3], [3, 1, 2, 6]]
    n_w, n_c = 2, 4
    m = dm.Model("warehouse_2x4")
    y = [m.binary(f"open_{i}") for i in range(n_w)]
    x = [
        [
            m.continuous(
                f"ship_{i}_{j}",
                lb=0.0,
                ub=float(demand[j]),
            )
            for j in range(n_c)
        ]
        for i in range(n_w)
    ]
    obj = dm.sum([w_fixed[i] * y[i] for i in range(n_w)])
    obj = obj + dm.sum([
        t_cost[i][j] * x[i][j]
        for i in range(n_w)
        for j in range(n_c)
    ])
    m.minimize(obj)
    # Demand satisfaction
    for j in range(n_c):
        m.subject_to(
            dm.sum([x[i][j] for i in range(n_w)]) >= demand[j],
            name=f"demand_{j}",
        )
    # Capacity + linking
    for i in range(n_w):
        m.subject_to(
            dm.sum([x[i][j] for j in range(n_c)])
            <= w_cap[i] * y[i],
            name=f"cap_{i}",
        )
    return m


# Open both: fixed=45.
# W0: c0=5(10),c3=4(12); W1: c1=8(8),c2=6(12). Trans=42. Tot=87.
# Better: W0: c0=5(10),c2=6(30); W1: c1=8(8),c3=4(24). No worse.
# W0: c0=5(10),c3=4(12)=14<=15. W1: c1=8(8),c2=6(12)=14<=20.
# Trans=10+12+8+12=42. Total=87.
# Open W1 only (cap 20<23=total demand): can't.
# Open W0 only (cap 15<23): can't.
# Must open both. Total=87.
# W0: c0(2),c2(5),c3(3) = 5+6+4=15<=15, trans=10+30+12=52.
# W1: c1(1) = 8<=20, trans=8. Total=45+52+8=105. Worse.
# W0: c0(10)+c3(12)=22, W1: c1(8)+c2(12)=20. Total=87.
# W1 serves cheap ones: c1(1*8=8), c2(2*6=12). W0: c0(2*5=10),
# c3(3*4=12). Total=45+42=87.
# W0:c0(10),c1(32)=42 cap15 used=13; W1:c2(12),c3(24)=36 cap20 u=10.
# Fixed=45, trans=42+36=78. Wait, that's wrong:
# W0:c0=5*2=10, c1=8*4=32? That uses 13 units, cap ok.
# W1:c2=6*2=12, c3=4*6=24, uses 10, cap ok. Trans=10+32+12+24=78.
# Total=45+78=123. Nope, 87 is better.
register(TestProblem(
    name="milp_warehouse_2x4",
    category="milp",
    level="full",
    build_fn=_build_warehouse_2x4,
    known_optimum=87.0,
    applicable_solvers=_SOLVERS,
    n_vars=10,  # 2 binary + 8 continuous
    n_constraints=6,  # 4 demand + 2 cap
    tags=["warehouse", "mixed_binary"],
))
