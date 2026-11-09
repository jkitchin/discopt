"""The retired matrix "pounce" MILP backend (Python-MILP-tree retirement, R11).

``milp_pounce.solve_milp`` used to run the Python MILP tree with POUNCE node LPs.
It is now an alias for :func:`discopt.solvers.milp_simplex.solve_milp` that warns
it is. These tests pin the alias: it warns, it keeps the contract, and selecting
``backend="pounce"`` gets it.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_backend import get_milp_solver
from discopt.solvers.milp_pounce import solve_milp

_KNAPSACK = dict(
    c=np.array([-3.0, -4.0, -5.0]),
    A_ub=np.array([[2.0, 3.0, 4.0]]),
    b_ub=np.array([6.0]),
    bounds=[(0, 1)] * 3,
    integrality=np.array([1, 1, 1]),
)


@pytest.mark.smoke
def test_alias_warns_and_solves_the_knapsack():
    with pytest.warns(FutureWarning, match="retired"):
        r = solve_milp(**_KNAPSACK)
    assert r.status == SolveStatus.OPTIMAL
    assert abs(r.objective - (-8.0)) < 1e-6  # x1 = x3 = 1 (weight 6)
    assert r.bound is not None and r.bound <= r.objective + 1e-9


def test_alias_keeps_the_infeasible_and_unbounded_verdicts():
    with pytest.warns(FutureWarning):
        r = solve_milp(
            c=np.array([1.0]),
            A_ub=np.array([[1.0]]),
            b_ub=np.array([-1.0]),
            bounds=[(0, 5)],
            integrality=np.array([1]),
        )
    assert r.status == SolveStatus.INFEASIBLE
    with pytest.warns(FutureWarning):
        r = solve_milp(c=np.array([-1.0]), bounds=[(0, None)], integrality=np.array([1]))
    assert r.status == SolveStatus.UNBOUNDED


def test_backend_pounce_selects_the_alias():
    solve = get_milp_solver(backend="pounce")
    assert solve is solve_milp
    assert get_milp_solver(prefer_pounce=True).__module__ == "discopt.solvers.milp_simplex"
