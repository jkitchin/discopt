"""Block-structured / two-stage MILP benchmark problems.

These instances have explicit decomposition structure (complicating variables
and/or coupling constraints) and are the canonical targets for the Benders and
Lagrangian solvers (``discopt.solve_benders`` / ``discopt.solve_lagrangian``).
They are registered as ``milp`` problems so the standard benchmark suite also
validates the underlying formulations; the decomposition-specific correctness
checks live in ``python/tests/test_decomposition_benchmarks.py``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np

from benchmarks.problems.base import TestProblem, register

_SOLVERS = ["ipm", "pounce", "ipopt"]


def build_two_stage_facility() -> dm.Model:
    """Open a facility (first stage), then ship to meet demand (recourse).

    min 2y + x1 + x2  s.t.  x1+x2 >= 3,  x1 <= 5y,  x2 <= 5y,  y binary.
    y=0 -> infeasible recourse; y=1 -> cost 2+3 = 5.  Optimum = 5.
    """
    m = dm.Model("dec_two_stage_facility")
    y = m.binary("y")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    m.first_stage(y)
    m.minimize(2 * y + x1 + x2)
    m.subject_to(x1 + x2 >= 3)
    m.subject_to(x1 <= 5 * y)
    m.subject_to(x2 <= 5 * y)
    return m


def build_block_conflict() -> dm.Model:
    """Two selection blocks tied by a conflict coupling. Optimum = 5.

    min 2x0+3x1+2x2+4x3  s.t.  x0+x1>=1 (A),  x2+x3>=1 (B),  x0+x2<=1 (coupling).
    The conflict forbids the two cheapest picks (x0, x2); best is x1, x2 -> 5.
    """
    m = dm.Model("dec_block_conflict")
    x = m.binary("x", shape=(4,))
    m.minimize(2 * x[0] + 3 * x[1] + 2 * x[2] + 4 * x[3])
    m.subject_to(x[0] + x[1] >= 1)
    m.subject_to(x[2] + x[3] >= 1)
    conf = x[0] + x[2] <= 1
    m.subject_to(conf)
    m.mark_coupling(conf)
    return m


def build_capacitated_two_stage() -> dm.Model:
    """Two capacity decisions (first stage) feeding two demands via shared flows.

    min 4a + 4b + (x1 + 2 x2 + 2 x3 + x4)
    s.t.  x1 + x2 = 3                (demand 1)
          x3 + x4 = 3                (demand 2)
          x1 + x3 <= 5 a             (capacity of source 1)
          x2 + x4 <= 5 b             (capacity of source 2)
          a, b binary; x >= 0.
    Opening one source (cost 4) covers both demands (cap 5 >= total 6? no:
    3+3 = 6 > 5), so both must open: 4+4 + cheapest routing. Cheapest routing
    sends x1=3 (cost 3) and x4=3 (cost 3) -> 6. Total = 8 + 6 = 14. Optimum = 14.
    """
    m = dm.Model("dec_capacitated_two_stage")
    a = m.binary("a")
    b = m.binary("b")
    x = m.continuous("x", shape=(4,), lb=0, ub=10)
    m.first_stage(a, b)
    m.minimize(4 * a + 4 * b + (x[0] + 2 * x[1] + 2 * x[2] + x[3]))
    m.subject_to(x[0] + x[1] == 3)
    m.subject_to(x[2] + x[3] == 3)
    m.subject_to(x[0] + x[2] <= 5 * a)
    m.subject_to(x[1] + x[3] <= 5 * b)
    return m


def build_gap_mixed_eq_ineq() -> dm.Model:
    """Generalized-assignment instance with mixed equality + inequality rows.

    This is the exact instance that exposed a MILP soundness bug: the simplex
    equilibration over-scaled a column carrying a ~1e-16 noise (cut) coefficient,
    so a binary came back at -1 and the solver returned 16 instead of the true
    optimum 17. Registered so the correctness gate guards that regression with
    the default solver. min cost.x ; per-agent knapsack `<=` blocks ; each task
    assigned exactly once (`==` coupling).
    """
    rng = np.random.default_rng(0)
    nk, ni = 3, 6
    cost = rng.integers(1, 10, (nk, ni))
    w = rng.integers(3, 9, (nk, ni))
    cap = [int(w[k].sum() * 0.45) for k in range(nk)]
    m = dm.Model("dec_gap_mixed_eq_ineq")
    x = m.binary("x", shape=(nk * ni,))

    def xi(k, i):
        return x[k * ni + i]

    m.minimize(sum(int(cost[k, i]) * xi(k, i) for k in range(nk) for i in range(ni)))
    for k in range(nk):
        m.subject_to(sum(int(w[k, i]) * xi(k, i) for i in range(ni)) <= cap[k])
    for i in range(ni):
        c = sum(xi(k, i) for k in range(nk)) == 1
        m.subject_to(c)
        m.mark_coupling(c)
    return m


register(
    TestProblem(
        name="dec_gap_mixed_eq_ineq",
        category="milp",
        level="smoke",
        build_fn=build_gap_mixed_eq_ineq,
        known_optimum=17.0,
        applicable_solvers=_SOLVERS,
        n_vars=18,
        n_constraints=9,
        tags=["decomposable", "coupling", "regression"],
    )
)

register(
    TestProblem(
        name="dec_two_stage_facility",
        category="milp",
        level="smoke",
        build_fn=build_two_stage_facility,
        known_optimum=5.0,
        applicable_solvers=_SOLVERS,
        n_vars=3,
        n_constraints=3,
        tags=["decomposable", "two_stage", "benders"],
    )
)

register(
    TestProblem(
        name="dec_block_conflict",
        category="milp",
        level="smoke",
        build_fn=build_block_conflict,
        known_optimum=5.0,
        applicable_solvers=_SOLVERS,
        n_vars=4,
        n_constraints=3,
        tags=["decomposable", "coupling", "lagrangian"],
    )
)

register(
    TestProblem(
        name="dec_capacitated_two_stage",
        category="milp",
        level="full",
        build_fn=build_capacitated_two_stage,
        known_optimum=14.0,
        applicable_solvers=_SOLVERS,
        n_vars=6,
        n_constraints=4,
        tags=["decomposable", "two_stage", "benders"],
    )
)
