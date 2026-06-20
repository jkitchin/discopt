"""Tests for the classical Benders decomposition solver.

Requires highspy for rigorous dual cut generation and the MILP master.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.benders import solve_benders

try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed")

ABS_TOL = 1e-3


def _two_stage_milp():
    """min 2y + x1 + x2 ; y binary first-stage; recourse x1,x2 >= 0.

    y=0 -> recourse infeasible (needs feasibility cut); y=1 -> cost 2+3 = 5.
    """
    m = dm.Model("two_stage")
    y = m.binary("y")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    m.minimize(2 * y + x1 + x2)
    m.subject_to(x1 + x2 >= 3)
    m.subject_to(x1 <= 5 * y)
    m.subject_to(x2 <= 5 * y)
    return m


def test_two_stage_milp_optimum():
    r = solve_benders(_two_stage_milp(), time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=ABS_TOL)
    assert r.x["y"] == pytest.approx(1.0, abs=1e-5)
    # Feasibility + optimality cuts give a tight, certified bound.
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-4


def test_maximize():
    m = dm.Model("max")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=4)
    m.maximize(3 * y - x)
    m.subject_to(x >= 2 * y)  # y=1 -> x>=2 -> 3-2 = 1
    r = solve_benders(m, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=ABS_TOL)


def test_annotated_continuous_lp():
    """Pure-continuous two-stage LP with an explicit first-stage annotation."""
    m = dm.Model("lp2")
    u = m.continuous("u", lb=0, ub=10)
    v = m.continuous("v", lb=0, ub=10)
    m.first_stage(u)
    m.minimize(u + 2 * v)
    m.subject_to(u + v >= 4)  # u=4, v=0 -> 4
    r = solve_benders(m, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(4.0, abs=ABS_TOL)


def test_infeasible():
    m = dm.Model("infeas")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=1)
    m.first_stage(y)
    m.minimize(x + y)
    m.subject_to(x >= 2)  # x in [0,1] but x>=2 -> infeasible recourse for all y
    r = solve_benders(m, time_limit=60)
    assert r.status == "infeasible"
    assert r.bound is None


def test_dispatch_via_model_solve():
    r = _two_stage_milp().solve(decomposition="benders", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=ABS_TOL)


def test_requires_first_stage_variable():
    """No integers and no annotation -> nothing to decompose on the master."""
    m = dm.Model("nomaster")
    a = m.continuous("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=5)
    m.minimize(a + b)
    m.subject_to(a + b >= 2)
    with pytest.raises(NotImplementedError):
        solve_benders(m, time_limit=10)


def test_integer_recourse_rejected():
    """Integer variable forced into recourse must be rejected (LP duals needed)."""
    m = dm.Model("intrecourse")
    u = m.continuous("u", lb=0, ub=5)
    z = m.integer("z", lb=0, ub=5)
    m.first_stage(u)  # u is first-stage, integer z left in recourse
    m.minimize(u + z)
    m.subject_to(u + z >= 3)
    with pytest.raises(NotImplementedError):
        solve_benders(m, time_limit=10)


def test_nonlinear_rejected():
    m = dm.Model("nl")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(x**2 + y)
    m.subject_to(x + y >= 1)
    with pytest.raises(NotImplementedError):
        solve_benders(m, time_limit=10)


def test_multidim_index_rejected_cleanly():
    """A 2-D indexed model raises a clean NotImplementedError, not a stray error."""
    m = dm.Model("twod")
    x = m.binary("x", shape=(2, 3))
    m.first_stage(x)
    m.minimize(sum(x[k, i] for k in range(2) for i in range(3)))
    m.subject_to(sum(x[0, i] for i in range(3)) >= 1)
    with pytest.raises(NotImplementedError):
        solve_benders(m, time_limit=10)


@pytest.mark.slow
def test_matches_monolithic_random():
    """Benders matches brute-force enumeration over the binary master."""
    from discopt.solvers import SolveStatus
    from discopt.solvers.lp_highs import solve_lp

    rng = np.random.default_rng(7)
    n_y, n_x = 2, 3
    for _ in range(15):
        m = dm.Model("rand")
        ys = m.binary("y", shape=(n_y,))
        xs = m.continuous("x", shape=(n_x,), lb=0, ub=5)
        cy = rng.uniform(1, 4, n_y)
        cx = rng.uniform(1, 4, n_x)
        m.minimize(sum(cy[i] * ys[i] for i in range(n_y)) + sum(cx[j] * xs[j] for j in range(n_x)))
        m.subject_to(sum(xs[j] for j in range(n_x)) >= 4)
        for j in range(n_x):
            m.subject_to(xs[j] <= 5 * ys[j % n_y])
        rb = solve_benders(m, time_limit=30)

        best = np.inf
        for yv in itertools.product([0, 1], repeat=n_y):
            rows = [-np.ones(n_x)]
            rhs = [-4.0]
            for j in range(n_x):
                row = np.zeros(n_x)
                row[j] = 1.0
                rows.append(row)
                rhs.append(5 * yv[j % n_y])
            lp = solve_lp(cx, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0, 5)] * n_x)
            if lp.status == SolveStatus.OPTIMAL:
                best = min(best, float(cy @ np.array(yv) + lp.objective))
        assert rb.objective == pytest.approx(best, abs=ABS_TOL)
