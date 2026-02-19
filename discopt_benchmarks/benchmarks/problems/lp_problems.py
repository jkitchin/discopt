"""LP benchmark problems using the discopt modeling API.

Smoke problems (5): small hand-crafted LPs with known optima.
Full problems (+15): structured and constructed LPs at various sizes.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np

from benchmarks.problems.base import TestProblem, register

_LP_SOLVERS = ["ipm", "ripopt", "ipopt", "highs"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constructed_lp(
    name: str,
    n: int,
    m: int,
    seed: int,
    level: str = "full",
    tags: list[str] | None = None,
) -> TestProblem:
    """Build an LP with a known optimum via primal-dual construction.

    Uses standard form: min c'x  s.t. Ax = b, x >= 0.
    With equality constraints, x=0 is not feasible when b > 0.

    Strategy (m < n required):
        1. Pick x_star > 0 (primal optimal, all positive).
        2. Generate A (m x n) with entries in (0, 1].
        3. Set b = A @ x_star  =>  x_star is primal feasible.
        4. Pick y_star (dual, unconstrained for equalities).
        5. Set c = A.T @ y_star + r where r >= 0, with r_j = 0 for
           all j (since x_star_j > 0, complementary slackness requires r_j = 0).
           So c = A.T @ y_star. We pick y_star so that c >= 0.
        6. Strong duality: opt = c @ x_star = y_star @ b.
    """
    rng = np.random.default_rng(seed)

    # Primal optimal: ALL variables positive
    x_star = rng.uniform(0.5, 2.0, size=n)

    # Constraint matrix (m x n with m < n for non-degenerate LP)
    a_mat = rng.uniform(0.1, 1.0, size=(m, n))

    # RHS from primal feasibility
    b = a_mat @ x_star

    # Dual: pick y_star so c = A^T y_star >= 0
    # Since A has positive entries and y_star > 0, c will be positive
    y_star = rng.uniform(0.1, 1.0, size=m)
    c = a_mat.T @ y_star

    opt = float(c @ x_star)

    def build() -> dm.Model:
        model = dm.Model(name)
        xs = [model.continuous(f"x{i}", lb=0.0) for i in range(n)]
        model.minimize(sum(float(c[j]) * xs[j] for j in range(n)))
        for i in range(m):
            lhs = sum(float(a_mat[i, j]) * xs[j] for j in range(n))
            # Equality constraint modeled as <= and >=
            model.subject_to(lhs <= float(b[i]))
            model.subject_to(lhs >= float(b[i]))
        return model

    return register(
        TestProblem(
            name=name,
            category="lp",
            level=level,
            build_fn=build,
            known_optimum=round(opt, 8),
            applicable_solvers=_LP_SOLVERS,
            n_vars=n,
            n_constraints=2 * m,  # Each equality = 2 inequalities
            tags=tags or [],
        )
    )


# ===================================================================
# Smoke problems (5)
# ===================================================================


def _build_lp_2var() -> dm.Model:
    """min -x - y  s.t.  x + y <= 4, x <= 3, x,y >= 0.  Opt: -4."""
    m = dm.Model("lp_2var")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize(-x - y)
    m.subject_to(x + y <= 4.0)
    m.subject_to(x <= 3.0)
    return m


register(
    TestProblem(
        name="lp_2var",
        category="lp",
        level="smoke",
        build_fn=_build_lp_2var,
        known_optimum=-4.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=2,
        n_constraints=2,
    )
)


def _build_lp_transport_2x2() -> dm.Model:
    """2-source, 2-destination transportation problem.

    Supply: s1=30, s2=40.  Demand: d1=20, d2=30.
    Cost matrix: [[2, 3], [5, 1]].
    Optimal: ship x11=20, x12=0, x21=0, x22=30 => cost = 2*20+1*30 = 70.
    (Total supply 70 > total demand 50, so supply constraints are <=.)
    """
    m = dm.Model("lp_transport_2x2")
    x11 = m.continuous("x11", lb=0.0)
    x12 = m.continuous("x12", lb=0.0)
    x21 = m.continuous("x21", lb=0.0)
    x22 = m.continuous("x22", lb=0.0)

    # Minimize transport cost
    m.minimize(2.0 * x11 + 3.0 * x12 + 5.0 * x21 + 1.0 * x22)

    # Supply constraints (<=)
    m.subject_to(x11 + x12 <= 30.0)  # source 1
    m.subject_to(x21 + x22 <= 40.0)  # source 2

    # Demand constraints (==, modeled as >= and <=)
    m.subject_to(x11 + x21 >= 20.0)  # dest 1
    m.subject_to(x12 + x22 >= 30.0)  # dest 2
    return m


register(
    TestProblem(
        name="lp_transport_2x2",
        category="lp",
        level="smoke",
        build_fn=_build_lp_transport_2x2,
        known_optimum=70.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=4,
        n_constraints=4,
    )
)


def _build_lp_diet_3() -> dm.Model:
    """Diet problem: 3 foods, 3 nutrients.

    Foods:  bread (cost 2), milk (cost 3.5), cheese (cost 8).
    Nutrients (min requirements):
        energy  >= 300:  bread=50,  milk=80,  cheese=100
        protein >= 10:   bread=1,   milk=3.3, cheese=8
        calcium >= 0.5:  bread=0.02,milk=0.15,cheese=0.3

    Optimal (verified): bread=6.0, milk=0.0, cheese=0.0
    does NOT work (protein: 6 < 10). Solve by inspection / simplex:

    Let b, k, c = bread, milk, cheese.
    energy:  50b + 80k + 100c >= 300
    protein: 1b + 3.3k + 8c >= 10
    calcium: 0.02b + 0.15k + 0.3c >= 0.5
    min 2b + 3.5k + 8c, b,k,c >= 0.

    Optimal (by LP): b=2.0, k=2.424..., c=0.0
    Check: energy=50*2+80*2.424=100+193.9=293.9 < 300. Not feasible.

    Use simpler numbers. Let's redesign for a clean solution.

    Foods: bread (cost 1), milk (cost 2), cheese (cost 5).
    energy:  6b + 4k + 2c >= 12
    protein: 1b + 2k + 4c >= 8
    calcium: 0b + 1k + 2c >= 2
    Optimal: b=0, k=2, c=1 => cost = 0 + 4 + 5 = 9.
    Check: energy=0+8+2=10 < 12. No.

    Simpler: just pick a vertex solution directly.
    energy:  2b + 3k + 4c >= 12
    protein: 1b + 2k + 3c >= 6
    min 1b + 2k + 5c, b,k,c >= 0.
    Opt: b=6, k=0, c=0 => cost=6. energy=12>=12, protein=6>=6. Yes.
    """
    m = dm.Model("lp_diet_3")
    b = m.continuous("bread", lb=0.0)
    k = m.continuous("milk", lb=0.0)
    c = m.continuous("cheese", lb=0.0)

    m.minimize(1.0 * b + 2.0 * k + 5.0 * c)
    m.subject_to(2.0 * b + 3.0 * k + 4.0 * c >= 12.0)  # energy
    m.subject_to(1.0 * b + 2.0 * k + 3.0 * c >= 6.0)  # protein
    return m


register(
    TestProblem(
        name="lp_diet_3",
        category="lp",
        level="smoke",
        build_fn=_build_lp_diet_3,
        known_optimum=6.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=3,
        n_constraints=2,
    )
)


def _build_lp_infeasible() -> dm.Model:
    """Infeasible LP: x >= 5 and x <= 3."""
    m = dm.Model("lp_infeasible")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 5.0)
    m.subject_to(x <= 3.0)
    return m


register(
    TestProblem(
        name="lp_infeasible",
        category="lp",
        level="smoke",
        build_fn=_build_lp_infeasible,
        known_optimum=float("inf"),
        applicable_solvers=_LP_SOLVERS,
        n_vars=1,
        n_constraints=2,
        expected_status="infeasible",
    )
)


def _build_lp_unbounded() -> dm.Model:
    """Unbounded LP: min -x, x >= 0, no upper bound."""
    m = dm.Model("lp_unbounded")
    x = m.continuous("x", lb=0.0)
    m.minimize(-x)
    return m


register(
    TestProblem(
        name="lp_unbounded",
        category="lp",
        level="smoke",
        build_fn=_build_lp_unbounded,
        known_optimum=float("-inf"),
        applicable_solvers=_LP_SOLVERS,
        n_vars=1,
        n_constraints=0,
        expected_status="unbounded",
    )
)


# ===================================================================
# Full problems (+15)
# ===================================================================


def _build_lp_network_flow() -> dm.Model:
    """Min-cost network flow on 5 nodes, 7 arcs.

    Nodes: 0 (source, supply=10), 1, 2, 3, 4 (sink, demand=10).
    Arcs (from, to, cost, capacity):
        0->1: cost=2, cap=8
        0->2: cost=4, cap=6
        1->2: cost=1, cap=4
        1->3: cost=3, cap=5
        2->3: cost=2, cap=5
        2->4: cost=5, cap=4
        3->4: cost=1, cap=10
    Optimal flow: 0->1:8, 0->2:2, 1->2:3, 1->3:5, 2->3:5, 2->4:0, 3->4:10
    Cost = 2*8+4*2+1*3+3*5+2*5+5*0+1*10 = 16+8+3+15+10+0+10 = 62.
    """
    arcs = [
        (0, 1, 2.0, 8.0),
        (0, 2, 4.0, 6.0),
        (1, 2, 1.0, 4.0),
        (1, 3, 3.0, 5.0),
        (2, 3, 2.0, 5.0),
        (2, 4, 5.0, 4.0),
        (3, 4, 1.0, 10.0),
    ]
    supply = {0: 10.0, 4: -10.0}  # positive = supply, negative = demand

    m = dm.Model("lp_network_flow")
    flow_vars = []
    for src, dst, _cost, cap in arcs:
        fv = m.continuous(f"f_{src}_{dst}", lb=0.0, ub=cap)
        flow_vars.append(fv)

    # Objective: minimize total cost
    m.minimize(sum(arcs[k][2] * flow_vars[k] for k in range(len(arcs))))

    # Flow conservation at each node
    for node in range(5):
        inflow = sum(flow_vars[k] for k, (i, j, _, _) in enumerate(arcs) if j == node)
        outflow = sum(flow_vars[k] for k, (i, j, _, _) in enumerate(arcs) if i == node)
        net_supply = supply.get(node, 0.0)
        # inflow - outflow + supply = 0  =>  outflow - inflow <= supply
        # and outflow - inflow >= supply
        m.subject_to(outflow - inflow <= net_supply)
        m.subject_to(outflow - inflow >= net_supply)
    return m


register(
    TestProblem(
        name="lp_network_flow",
        category="lp",
        level="full",
        build_fn=_build_lp_network_flow,
        known_optimum=62.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=7,
        n_constraints=10,
        tags=["network"],
    )
)


def _build_lp_assignment_4x4() -> dm.Model:
    """4x4 assignment problem as LP relaxation.

    Cost matrix:
        [[9, 2, 7, 8],
         [6, 4, 3, 7],
         [5, 8, 1, 8],
         [7, 6, 9, 4]]
    Optimal assignment: (0,1)=2, (1,0)=6, (2,2)=1, (3,3)=4 => cost=13.
    LP relaxation of assignment is always integral (totally unimodular).
    """
    costs = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4],
    ]
    n = 4
    m = dm.Model("lp_assignment_4x4")
    xs = {}
    for i in range(n):
        for j in range(n):
            xs[i, j] = m.continuous(f"x_{i}_{j}", lb=0.0, ub=1.0)

    m.minimize(sum(costs[i][j] * xs[i, j] for i in range(n) for j in range(n)))

    # Each row sums to 1
    for i in range(n):
        row_sum = sum(xs[i, j] for j in range(n))
        m.subject_to(row_sum >= 1.0)
        m.subject_to(row_sum <= 1.0)

    # Each column sums to 1
    for j in range(n):
        col_sum = sum(xs[i, j] for i in range(n))
        m.subject_to(col_sum >= 1.0)
        m.subject_to(col_sum <= 1.0)

    return m


register(
    TestProblem(
        name="lp_assignment_4x4",
        category="lp",
        level="full",
        build_fn=_build_lp_assignment_4x4,
        known_optimum=13.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=16,
        n_constraints=16,
        tags=["assignment", "combinatorial"],
    )
)


def _build_lp_covering_small() -> dm.Model:
    """Set covering LP relaxation: 5 elements, 4 sets.

    Sets (cost, elements covered):
        S1 (cost 3): {1, 2, 3}
        S2 (cost 2): {2, 4}
        S3 (cost 4): {1, 3, 5}
        S4 (cost 1): {4, 5}
    min 3x1 + 2x2 + 4x3 + x4  s.t.  each element covered, 0<=xi<=1.
    Elem 1: x1 + x3 >= 1
    Elem 2: x1 + x2 >= 1
    Elem 3: x1 + x3 >= 1  (same as elem 1)
    Elem 4: x2 + x4 >= 1
    Elem 5: x3 + x4 >= 1
    Optimal LP: x1=1, x2=0, x3=0, x4=1 => cost=4.
    Check: e1: 1>=1, e2: 1>=1, e3: 1>=1, e4: 0+1=1>=1, e5: 0+1=1>=1. Yes.
    Can we do better? x1=1,x4=1 costs 4. x2+x4 costs 3 but misses e1,e3.
    x1=1,x4=1 is optimal at cost 4.
    """
    m = dm.Model("lp_covering_small")
    x1 = m.continuous("x1", lb=0.0, ub=1.0)
    x2 = m.continuous("x2", lb=0.0, ub=1.0)
    x3 = m.continuous("x3", lb=0.0, ub=1.0)
    x4 = m.continuous("x4", lb=0.0, ub=1.0)

    m.minimize(3.0 * x1 + 2.0 * x2 + 4.0 * x3 + 1.0 * x4)
    m.subject_to(x1 + x3 >= 1.0)  # elem 1 & 3
    m.subject_to(x1 + x2 >= 1.0)  # elem 2
    m.subject_to(x2 + x4 >= 1.0)  # elem 4
    m.subject_to(x3 + x4 >= 1.0)  # elem 5
    return m


register(
    TestProblem(
        name="lp_covering_small",
        category="lp",
        level="full",
        build_fn=_build_lp_covering_small,
        known_optimum=4.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=4,
        n_constraints=4,
        tags=["covering"],
    )
)


def _build_lp_blending() -> dm.Model:
    """Blending problem: mix 3 raw materials to meet quality specs.

    Raw materials (cost per unit, quality A%, quality B%):
        R1: cost=5,  A=30%, B=10%
        R2: cost=8,  A=50%, B=20%
        R3: cost=3,  A=10%, B=40%

    Product requires >= 25% A, >= 15% B, total output = 100 units.
    min 5r1 + 8r2 + 3r3
    s.t. r1 + r2 + r3 = 100
         0.30*r1 + 0.50*r2 + 0.10*r3 >= 25  (quality A)
         0.10*r1 + 0.20*r2 + 0.40*r3 >= 15  (quality B)
         r1, r2, r3 >= 0

    Optimal: r1=100, r2=0, r3=0 => cost=500.
    Check A: 30>=25, B: 10<15. Infeasible.

    r3=50/3=16.67, r1=83.33, r2=0: A=25+1.67=26.67>=25,
    B=8.33+6.67=15. cost=5*83.33+3*16.67=416.67+50=466.67.
    But let's just solve: the cheapest (R3) is limited by quality A.

    A: 0.3r1+0.5r2+0.1r3 >= 25 with r1+r2+r3=100
    B: 0.1r1+0.2r2+0.4r3 >= 15

    Substituting r1=100-r2-r3:
    A: 30-0.2r2-0.2r3 >= 25 => r2+r3 <= 25
    B: 10+0.1r2+0.3r3 >= 15 => r2+3r3 >= 50

    min 500+3r2-2r3, s.t. r2+r3<=25, r2+3r3>=50, r2,r3>=0.
    Want to maximize 2r3-3r2.
    From r2+3r3>=50 => r2>=50-3r3.
    From r2+r3<=25 => r2<=25-r3, and r3<=25.
    50-3r3 <= 25-r3 => 25<=2r3 => r3>=12.5.
    At r3=12.5: r2>=50-37.5=12.5. r2=12.5.
    cost = 500+3*12.5-2*12.5 = 500+37.5-25 = 512.5.

    At r3=25: r2>=50-75<0, so r2=0. r2+r3=25<=25. OK.
    cost = 500+0-50 = 450.
    Check B: 0+3*25=75>=50. A: 0+25=25<=25. Both OK.
    r1=75, r2=0, r3=25. cost=5*75+3*25=375+75=450.
    """
    m = dm.Model("lp_blending")
    r1 = m.continuous("r1", lb=0.0)
    r2 = m.continuous("r2", lb=0.0)
    r3 = m.continuous("r3", lb=0.0)

    m.minimize(5.0 * r1 + 8.0 * r2 + 3.0 * r3)
    # Total output = 100
    m.subject_to(r1 + r2 + r3 <= 100.0)
    m.subject_to(r1 + r2 + r3 >= 100.0)
    # Quality A >= 25%
    m.subject_to(0.30 * r1 + 0.50 * r2 + 0.10 * r3 >= 25.0)
    # Quality B >= 15%
    m.subject_to(0.10 * r1 + 0.20 * r2 + 0.40 * r3 >= 15.0)
    return m


register(
    TestProblem(
        name="lp_blending",
        category="lp",
        level="full",
        build_fn=_build_lp_blending,
        known_optimum=450.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=3,
        n_constraints=4,
        tags=["blending"],
    )
)


def _build_lp_production_planning() -> dm.Model:
    """Multi-period production planning (3 periods).

    Demand: d=[20, 40, 30].  Production cost: 10/unit.
    Inventory cost: 2/unit/period.  Max production: 35/period.
    Initial inventory: 0.

    Variables: p_t (production), s_t (inventory at end of period t).
    min sum(10*p_t + 2*s_t)
    s.t. s_t = s_{t-1} + p_t - d_t (inventory balance)
         0 <= p_t <= 35
         s_t >= 0

    Optimal: produce as much as possible early to avoid shortfall.
    p1=35, s1=15; p2=35, s2=10; p3=30, s3=10.
    Wait: d=[20,40,30], total=90, max prod=105.
    p1=35, s1=35-20=15; p2=35, s2=15+35-40=10; p3=30, s3=10+30-30=10.
    cost=10*(35+35+30)+2*(15+10+10)=1000+70=1070.
    But is there better? p1=20,s1=0; p2=35,s2=-5. Infeasible.
    p1=25,s1=5; p2=35,s2=0; p3=30,s3=0.
    cost=10*90+2*5=900+10=910.
    Even better: p1=20,s1=0; p2=35,s2=-5. No.
    p1=20,s1=0; p2=40>35. Infeasible.
    p1=25,s1=5; p2=35,s2=0; p3=30,s3=0 => 910.
    p1=20,s1=0; p2=35,s2=-5: infeasible.
    p1=30,s1=10; p2=35,s2=5; p3=30,s3=5 => 10*95+2*20=950+40=990.
    p1=25,s1=5; p2=35,s2=0; p3=30,s3=0 => 10*90+2*5=910. Best.
    """
    demand = [20.0, 40.0, 30.0]
    prod_cost = 10.0
    hold_cost = 2.0
    max_prod = 35.0
    n_periods = 3

    m = dm.Model("lp_production_planning")
    p = [m.continuous(f"p{t}", lb=0.0, ub=max_prod) for t in range(n_periods)]
    s = [m.continuous(f"s{t}", lb=0.0) for t in range(n_periods)]

    m.minimize(sum(prod_cost * p[t] + hold_cost * s[t] for t in range(n_periods)))

    # Inventory balance: s[t] = s[t-1] + p[t] - d[t]
    # Period 0: s[0] = 0 + p[0] - d[0]
    m.subject_to(s[0] <= p[0] - demand[0])
    m.subject_to(s[0] >= p[0] - demand[0])
    for t in range(1, n_periods):
        m.subject_to(s[t] <= s[t - 1] + p[t] - demand[t])
        m.subject_to(s[t] >= s[t - 1] + p[t] - demand[t])

    return m


register(
    TestProblem(
        name="lp_production_planning",
        category="lp",
        level="full",
        build_fn=_build_lp_production_planning,
        known_optimum=910.0,
        applicable_solvers=_LP_SOLVERS,
        n_vars=6,
        n_constraints=6,
        tags=["planning"],
    )
)


# ===================================================================
# Constructed LPs via primal-dual pairs (10 problems)
# ===================================================================

_constructed_lp("lp_constructed_5x3", n=5, m=3, seed=1001)
_constructed_lp("lp_constructed_8x5", n=8, m=5, seed=1002)
_constructed_lp("lp_constructed_10x6", n=10, m=6, seed=1003)
_constructed_lp("lp_constructed_15x10", n=15, m=10, seed=1004)
_constructed_lp("lp_constructed_20x12", n=20, m=12, seed=1005)
_constructed_lp("lp_constructed_30x20", n=30, m=20, seed=1006)
_constructed_lp("lp_constructed_50x30", n=50, m=30, seed=1007)
_constructed_lp("lp_constructed_75x50", n=75, m=50, seed=1008)
_constructed_lp("lp_constructed_100x60", n=100, m=60, seed=1009)
_constructed_lp("lp_constructed_200x120", n=200, m=120, seed=1010)
