"""Tests for the Lagrangian relaxation solver."""

import discopt.modeling as dm
import pytest
from discopt.decomposition.lagrangian import solve_lagrangian

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")

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


def test_recovered_primal_is_feasible():
    r = solve_lagrangian(_two_block_conflict(), time_limit=30)
    assert r.x is not None
    xv = r.x["x"]
    assert xv[0] + xv[1] >= 1 - 1e-5
    assert xv[2] + xv[3] >= 1 - 1e-5
    assert xv[0] + xv[2] <= 1 + 1e-5
