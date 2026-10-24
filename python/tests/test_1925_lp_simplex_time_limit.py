"""#1925: ``lp_simplex.solve_lp`` forwards ``time_limit`` to the Rust simplex.

It accepted the argument and dropped it, so OBBT/DBBT's per-LP budget (0.1 s)
did not bound the LP: on a 1000 x 1000 LP a 0.05 s limit took as long (~5 s) as
no limit at all.
"""

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import SolveStatus
from discopt.solvers.lp_simplex import solve_lp


def _lp(n=1000):
    rng = np.random.default_rng(0)
    A = sp.random(n, n, density=0.02, random_state=1, format="csr") + sp.identity(n, format="csr")
    b = A @ np.ones(n) + 1.0
    c = -rng.random(n)
    return c, A, b, [(0, 10)] * n


@pytest.mark.unit
def test_time_limit_stops_the_simplex():
    c, A, b, bounds = _lp()
    t = time.perf_counter()
    r = solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds, time_limit=0.05)
    wall = time.perf_counter() - t
    assert r.status == SolveStatus.TIME_LIMIT
    assert r.objective is None  # no bound from a deadline exit
    assert wall < 2.0, wall


@pytest.mark.unit
def test_spent_budget_returns_no_objective():
    c, A, b, bounds = _lp(200)
    r = solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds, time_limit=0.0)
    assert r.status != SolveStatus.OPTIMAL
    assert r.objective is None


@pytest.mark.unit
def test_ample_budget_matches_unbudgeted_solve():
    c, A, b, bounds = _lp(200)
    r0 = solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds)
    r1 = solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds, time_limit=60.0)
    assert r0.status == r1.status == SolveStatus.OPTIMAL
    assert r1.objective == r0.objective
    np.testing.assert_array_equal(r1.x, r0.x)
