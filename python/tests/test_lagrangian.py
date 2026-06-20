"""Tests for the Lagrangian relaxation solver."""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.lagrangian import solve_lagrangian

try:
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
except ImportError:
    POUNCE_AVAILABLE = False
try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(
    not (POUNCE_AVAILABLE or HAS_HIGHS), reason="no LP/MILP backend available"
)

ABS_TOL = 1e-3


def _knapsack():
    """Maximize value, pick at most 2 of {2,3,2,4} -> 3+4 = 7."""
    m = dm.Model("knap")
    x = m.binary("x", shape=(4,))
    vals = [2, 3, 2, 4]
    m.maximize(sum(vals[i] * x[i] for i in range(4)))
    cpl = sum(x[i] for i in range(4)) <= 2
    m.subject_to(cpl)
    m.mark_coupling(cpl)
    return m


def _two_block_conflict():
    """Two blocks with a conflict coupling. Optimum 5 (x2, x3)."""
    m = dm.Model("blocks")
    x = m.binary("x", shape=(4,))
    m.minimize(2 * x[0] + 3 * x[1] + 2 * x[2] + 4 * x[3])
    m.subject_to(x[0] + x[1] >= 1)  # block A
    m.subject_to(x[2] + x[3] >= 1)  # block B
    conf = x[0] + x[2] <= 1
    m.subject_to(conf)
    m.mark_coupling(conf)
    return m


@pytest.mark.parametrize("method", ["subgradient", "bundle"])
def test_knapsack(method):
    r = solve_lagrangian(_knapsack(), method=method, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(7.0, abs=ABS_TOL)
    # Maximize: reported bound is an upper bound, never below the objective.
    assert r.bound >= r.objective - ABS_TOL


@pytest.mark.parametrize("method", ["subgradient", "bundle"])
def test_two_block_conflict(method):
    r = solve_lagrangian(_two_block_conflict(), method=method, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=ABS_TOL)
    # Minimize: reported bound is a lower bound, never above the objective.
    assert r.bound <= r.objective + ABS_TOL


def test_dispatch_via_model_solve():
    r = _two_block_conflict().solve(decomposition="lagrangian", time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=ABS_TOL)


def test_no_coupling_raises():
    m = dm.Model("nocouple")
    x = m.binary("x", shape=(2,))
    m.minimize(x[0] + x[1])
    m.subject_to(x[0] + x[1] >= 1)
    with pytest.raises(NotImplementedError):
        solve_lagrangian(m, time_limit=10)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        solve_lagrangian(_knapsack(), method="nope", time_limit=10)


def test_multidim_index_rejected_cleanly():
    """A 2-D indexed model raises a clean NotImplementedError, not a stray error."""
    m = dm.Model("twod")
    x = m.binary("x", shape=(2, 3))
    m.minimize(sum(x[k, i] for k in range(2) for i in range(3)))
    c = sum(x[0, i] for i in range(3)) >= 1
    m.subject_to(c)
    m.mark_coupling(c)
    with pytest.raises(NotImplementedError):
        solve_lagrangian(m, time_limit=10)


def test_recovered_primal_is_feasible():
    r = solve_lagrangian(_two_block_conflict(), time_limit=30)
    assert r.x is not None
    xv = r.x["x"]
    assert xv[0] + xv[1] >= 1 - 1e-5
    assert xv[2] + xv[3] >= 1 - 1e-5
    assert xv[0] + xv[2] <= 1 + 1e-5


def _equality_coupling_gap():
    """Assignment-style GAP with an *equality* coupling (each task assigned once),
    split into two <= rows that dualize with sign-restricted multipliers.

    2 agents x 3 tasks; min cost; each agent's knapsack is a block. Brute-forced
    optimum below.
    """
    cost = [[4, 6, 3], [5, 2, 7]]
    w = [[3, 4, 2], [2, 3, 4]]
    cap = [6, 6]
    m = dm.Model("eqgap")
    x = m.binary("x", shape=(6,))

    def xi(k, i):
        return x[k * 3 + i]

    m.minimize(sum(cost[k][i] * xi(k, i) for k in range(2) for i in range(3)))
    for k in range(2):
        m.subject_to(sum(w[k][i] * xi(k, i) for i in range(3)) <= cap[k])
    for i in range(3):
        c = sum(xi(k, i) for k in range(2)) == 1  # equality coupling
        m.subject_to(c)
        m.mark_coupling(c)
    return m, cost, w, cap


@pytest.mark.parametrize("method", ["subgradient", "bundle"])
def test_equality_coupling_sound_and_feasible(method):
    """Equality coupling: the dual bound is sound and the recovered incumbent is
    feasible in *every* constraint (not just the coupling rows)."""
    import itertools

    m, cost, w, cap = _equality_coupling_gap()
    # brute-force optimum
    best = float("inf")
    for bits in itertools.product([0, 1], repeat=6):

        def s(k, i, bits=bits):
            return bits[k * 3 + i]

        if all(sum(w[k][i] * s(k, i) for i in range(3)) <= cap[k] for k in range(2)) and all(
            sum(s(k, i) for k in range(2)) == 1 for i in range(3)
        ):
            best = min(best, sum(cost[k][i] * s(k, i) for k in range(2) for i in range(3)))
    r = solve_lagrangian(m, method=method, time_limit=30)
    if r.bound is not None:
        assert r.bound <= best + 1e-3, f"unsound dual bound {r.bound} > opt {best}"
    if r.x is not None:
        xv = np.asarray(r.x["x"]).ravel().round()
        for k in range(2):
            assert sum(w[k][i] * xv[k * 3 + i] for i in range(3)) <= cap[k] + 1e-5
        for i in range(3):
            assert abs(sum(xv[k * 3 + i] for k in range(2)) - 1) <= 1e-5
        recomputed = sum(cost[k][i] * xv[k * 3 + i] for k in range(2) for i in range(3))
        if r.objective is not None:
            assert recomputed == pytest.approx(r.objective, abs=1e-3)
    if r.status == "optimal":
        assert r.objective == pytest.approx(best, abs=1e-3)
