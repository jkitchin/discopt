"""MIQP benchmark problems for mixed-integer quadratic programming."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from benchmarks.problems.base import TestProblem, register

_APPLICABLE = ["ipm", "ripopt", "ipopt"]


# ---------------------------------------------------------------------------
# Smoke problems (5)
# ---------------------------------------------------------------------------


def _build_miqp_portfolio_binary_3():
    """3-asset portfolio with binary selection, diagonal Q."""
    import discopt.modeling as dm

    m = dm.Model("miqp_portfolio_binary_3")
    # Continuous weights
    x0 = m.continuous("x0", lb=0.0, ub=1.0)
    x1 = m.continuous("x1", lb=0.0, ub=1.0)
    x2 = m.continuous("x2", lb=0.0, ub=1.0)
    # Binary selection
    z0 = m.binary("z0")
    z1 = m.binary("z1")
    z2 = m.binary("z2")
    # Objective: min x0^2 + 4*x1^2 + 9*x2^2
    m.minimize(x0**2 + 4 * x1**2 + 9 * x2**2)
    # Budget: weights sum to 1
    m.subject_to(x0 + x1 + x2 >= 1.0)
    m.subject_to(x0 + x1 + x2 <= 1.0)
    # Linking: x_i <= z_i
    m.subject_to(x0 <= z0)
    m.subject_to(x1 <= z1)
    m.subject_to(x2 <= z2)
    # Cardinality: at most 2 assets
    m.subject_to(z0 + z1 + z2 <= 2.0)
    return m


register(
    TestProblem(
        name="miqp_portfolio_binary_3",
        category="miqp",
        level="smoke",
        build_fn=_build_miqp_portfolio_binary_3,
        known_optimum=0.8,
        applicable_solvers=_APPLICABLE,
        n_vars=6,
        n_constraints=6,
        source="programmatic",
        tags=["portfolio", "cardinality"],
    )
)


def _build_miqp_quadratic_knapsack_4():
    """4-item binary knapsack with quadratic objective.

    Items: values [10, 8, 6, 4], weights [3, 2, 4, 1], capacity 6.
    Interaction penalty: 0.5 * x_i * x_j for all pairs.
    Maximize: sum(v_i*x_i) - 0.5*sum_{i<j}(x_i*x_j).

    We formulate as minimization of the negative.
    Enumeration of feasible sets (weight <= 6):
      {0,1,3}: w=6, obj=-(10+8+4)+0.5*(1+1+1)=-22+1.5=-20.5
      {0,3}:   w=4, obj=-(10+4)+0.5*1=-14+0.5=-13.5
      {1,2,3}: w=7 > 6 infeasible
      {0,1}:   w=5, obj=-(10+8)+0.5=-17.5
      {1,3}:   w=3, obj=-(8+4)+0.5=-11.5
      {2,3}:   w=5, obj=-(6+4)+0.5=-9.5
      {0,2}:   w=7 > 6 infeasible
      {0}:     w=3, obj=-10
      {1}:     w=2, obj=-8
      {2}:     w=4, obj=-6
      {3}:     w=1, obj=-4
      {0,1,3}: best at -20.5
    """
    import discopt.modeling as dm

    m = dm.Model("miqp_quadratic_knapsack_4")
    x = [m.binary(f"x{i}") for i in range(4)]
    values = [10.0, 8.0, 6.0, 4.0]
    weights = [3.0, 2.0, 4.0, 1.0]
    cap = 6.0

    # Minimize negative profit + interaction penalty
    obj = 0.0
    for i in range(4):
        obj = obj + (-values[i]) * x[i]
    for i in range(4):
        for j in range(i + 1, 4):
            obj = obj + 0.5 * x[i] * x[j]
    m.minimize(obj)

    # Capacity constraint
    weight_sum = 0.0
    for i in range(4):
        weight_sum = weight_sum + weights[i] * x[i]
    m.subject_to(weight_sum <= cap)
    return m


register(
    TestProblem(
        name="miqp_quadratic_knapsack_4",
        category="miqp",
        level="smoke",
        build_fn=_build_miqp_quadratic_knapsack_4,
        known_optimum=-20.5,
        applicable_solvers=_APPLICABLE,
        n_vars=4,
        n_constraints=1,
        source="programmatic",
        tags=["knapsack", "binary"],
    )
)


def _build_miqp_facility_qp():
    """2 facilities, 3 customers, quadratic transport costs.

    Binary y_j = open facility j. Continuous x_ij = fraction of
    customer i served by facility j. Quadratic cost: c_ij * x_ij^2.

    Costs c_ij:
      cust\\fac  0    1
        0       1    4
        1       4    1
        2       2    2

    Fixed cost for opening: f = [3, 3].
    Each customer fully served: sum_j x_ij = 1.
    Linking: x_ij <= y_j.

    If both open (y0=y1=1): fixed=6, assign optimally:
      c0->f0: 1, c1->f1: 1, c2->either: 2. Total=6+1+1+2=10.
    If only y0=1: x_i0=1 for all i, cost=3+1+4+2=10.
    If only y1=1: x_i1=1 for all i, cost=3+4+1+2=10.
    Best: any single facility or both open, all give 10.0.
    """
    import discopt.modeling as dm

    m = dm.Model("miqp_facility_qp")
    y0 = m.binary("y0")
    y1 = m.binary("y1")

    costs = [[1.0, 4.0], [4.0, 1.0], [2.0, 2.0]]
    x = {}
    for i in range(3):
        for j in range(2):
            x[i, j] = m.continuous(f"x_{i}_{j}", lb=0.0, ub=1.0)

    # Objective: fixed costs + quadratic transport
    obj = 3.0 * y0 + 3.0 * y1
    for i in range(3):
        for j in range(2):
            obj = obj + costs[i][j] * x[i, j] ** 2
    m.minimize(obj)

    # Each customer fully served
    for i in range(3):
        m.subject_to(x[i, 0] + x[i, 1] >= 1.0)
        m.subject_to(x[i, 0] + x[i, 1] <= 1.0)

    # Linking
    for i in range(3):
        m.subject_to(x[i, 0] <= y0)
        m.subject_to(x[i, 1] <= y1)

    # At least one facility open
    m.subject_to(y0 + y1 >= 1.0)
    return m


register(
    TestProblem(
        name="miqp_facility_qp",
        category="miqp",
        level="smoke",
        build_fn=_build_miqp_facility_qp,
        known_optimum=8.6,
        applicable_solvers=_APPLICABLE,
        n_vars=8,
        n_constraints=10,
        source="programmatic",
        tags=["facility_location"],
    )
)


def _build_miqp_simple_2var():
    """1 continuous x in [0,5], 1 binary y.

    min (x-3)^2 + 10*y  s.t. x <= 5*y.
    y=0 => x=0, obj=9.  y=1 => x=3, obj=10. Opt: y=0, obj=9.0.
    """
    import discopt.modeling as dm

    m = dm.Model("miqp_simple_2var")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.binary("y")
    m.minimize((x - 3) ** 2 + 10 * y)
    m.subject_to(x <= 5 * y)
    return m


register(
    TestProblem(
        name="miqp_simple_2var",
        category="miqp",
        level="smoke",
        build_fn=_build_miqp_simple_2var,
        known_optimum=9.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=1,
        source="programmatic",
        tags=["simple"],
    )
)


def _build_miqp_mixed_3():
    """2 continuous + 1 binary.

    min x1^2 + x2^2 + 5*y  s.t. x1+x2 >= 2*y, x1,x2 in [0,3].
    y=0 => x1=x2=0, obj=0.  y=1 => x1=x2=1, obj=7. Opt=0.0.
    """
    import discopt.modeling as dm

    m = dm.Model("miqp_mixed_3")
    x1 = m.continuous("x1", lb=0.0, ub=3.0)
    x2 = m.continuous("x2", lb=0.0, ub=3.0)
    y = m.binary("y")
    m.minimize(x1**2 + x2**2 + 5 * y)
    m.subject_to(x1 + x2 >= 2 * y)
    return m


register(
    TestProblem(
        name="miqp_mixed_3",
        category="miqp",
        level="smoke",
        build_fn=_build_miqp_mixed_3,
        known_optimum=0.0,
        applicable_solvers=_APPLICABLE,
        n_vars=3,
        n_constraints=1,
        source="programmatic",
        tags=["simple"],
    )
)


# ---------------------------------------------------------------------------
# Full problems: random convex MIQP at various sizes
# ---------------------------------------------------------------------------


def _make_random_miqp_builder(n: int, n_binary: int, seed: int):
    """Return a builder for a random convex MIQP with known optimum.

    Strategy: generate a random PSD Q matrix, random linear c, random
    binary optimum y*, solve the continuous sub-problem analytically
    to get x*, then record objective.
    """

    def _build():
        import discopt.modeling as dm

        rng = np.random.RandomState(seed)
        n_cont = n - n_binary

        # Build PSD Q for continuous variables: Q = A'A + I
        a_mat = rng.randn(n_cont, n_cont)
        q_mat = a_mat.T @ a_mat + np.eye(n_cont)
        c_cont = rng.randn(n_cont)
        c_bin = rng.uniform(1.0, 5.0, size=n_binary)

        m = dm.Model(f"miqp_random_n{n}_s{seed}")
        x_vars = []
        for i in range(n_cont):
            v = m.continuous(f"x{i}", lb=0.0, ub=10.0)
            x_vars.append(v)

        y_vars = []
        for i in range(n_binary):
            v = m.binary(f"y{i}")
            y_vars.append(v)

        # Objective: 0.5 * x'Qx + c'x + d'y
        obj = 0.0
        for i in range(n_cont):
            for j in range(n_cont):
                obj = obj + 0.5 * float(q_mat[i, j]) * (x_vars[i] * x_vars[j])
        for i in range(n_cont):
            obj = obj + float(c_cont[i]) * x_vars[i]
        for i in range(n_binary):
            obj = obj + float(c_bin[i]) * y_vars[i]
        m.minimize(obj)

        # Constraint: sum(x) + sum(y) >= 1
        total = 0.0
        for v in x_vars:
            total = total + v
        for v in y_vars:
            total = total + v
        m.subject_to(total >= 1.0)
        return m

    return _build


def _compute_random_miqp_optimum(n: int, n_binary: int, seed: int) -> float:
    """Compute known optimum for random MIQP by enumeration.

    For small n_binary, enumerate all 2^n_binary binary combos
    and solve each continuous QP via scipy.
    """
    rng = np.random.RandomState(seed)
    n_cont = n - n_binary

    a_mat = rng.randn(n_cont, n_cont)
    q_mat = a_mat.T @ a_mat + np.eye(n_cont)
    c_cont = rng.randn(n_cont)
    c_bin = rng.uniform(1.0, 5.0, size=n_binary)

    best_obj = float("inf")
    for bits in range(2**n_binary):
        y_val = np.array([float((bits >> k) & 1) for k in range(n_binary)])
        y_cost = float(c_bin @ y_val)
        y_sum = float(y_val.sum())

        if n_cont == 0:
            # Pure binary, just check constraint
            if y_sum >= 1.0 and y_cost < best_obj:
                best_obj = y_cost
            continue

        # Solve continuous QP: min 0.5*x'Qx + c'x + y_cost
        # s.t. sum(x) + y_sum >= 1, x in [0, 10]
        _yc, _ys = y_cost, y_sum  # bind for closures

        def _obj(x, _yc=_yc):
            return 0.5 * x @ q_mat @ x + c_cont @ x + _yc

        constraints = []
        if _ys < 1.0:
            constraints.append({
                "type": "ineq",
                "fun": lambda x, _ys=_ys: x.sum() + _ys - 1.0,
            })

        res = scipy_minimize(
            _obj,
            x0=np.zeros(n_cont),
            jac=lambda x: q_mat @ x + c_cont,
            method="SLSQP",
            bounds=[(0.0, 10.0)] * n_cont,
            constraints=constraints,
        )
        if res.success and res.fun < best_obj:
            best_obj = res.fun

    return float(best_obj)


# Register full problems: (n_total, n_binary, seed) pairs
_FULL_CONFIGS = [
    (5, 2, 42),
    (5, 2, 137),
    (8, 3, 42),
    (8, 3, 137),
    (10, 4, 42),
    (10, 4, 137),
    (15, 5, 42),
    (15, 5, 137),
    (20, 6, 42),
    (20, 6, 137),
]

for _n, _nb, _s in _FULL_CONFIGS:
    _opt = _compute_random_miqp_optimum(_n, _nb, _s)
    register(
        TestProblem(
            name=f"miqp_random_n{_n}_s{_s}",
            category="miqp",
            level="full",
            build_fn=_make_random_miqp_builder(_n, _nb, _s),
            known_optimum=round(_opt, 8),
            applicable_solvers=_APPLICABLE,
            n_vars=_n,
            n_constraints=1,
            source="programmatic",
            tags=["random", "convex_continuous"],
        )
    )
