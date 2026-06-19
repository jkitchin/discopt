"""Soundness guards for Benders decomposition.

Headline invariant: the reported dual ``bound`` is a *valid lower bound* — it
never exceeds the true optimum, and never exceeds the incumbent objective on an
``optimal`` exit. Every Benders cut is a global underestimator (LP weak
duality), so these must hold for all (mixed-integer) linear instances.
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

pytestmark = [
    pytest.mark.skipif(not HAS_HIGHS, reason="highspy not installed"),
    pytest.mark.correctness,
]


def _brute_force_optimum(cy, cx, demand, n_y, n_x):
    from discopt.solvers import SolveStatus
    from discopt.solvers.lp_highs import solve_lp

    best = np.inf
    for yv in itertools.product([0, 1], repeat=n_y):
        rows = [-np.ones(n_x)]
        rhs = [-float(demand)]
        for j in range(n_x):
            row = np.zeros(n_x)
            row[j] = 1.0
            rows.append(row)
            rhs.append(5.0 * yv[j % n_y])
        lp = solve_lp(cx, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0, 5)] * n_x)
        if lp.status == SolveStatus.OPTIMAL:
            best = min(best, float(cy @ np.array(yv) + lp.objective))
    return best


def test_bound_never_exceeds_true_optimum():
    """Across random instances, bound <= true optimum (the key invariant)."""
    rng = np.random.default_rng(123)
    n_y, n_x, demand = 2, 3, 4.0
    for _ in range(20):
        m = dm.Model("s")
        ys = m.binary("y", shape=(n_y,))
        xs = m.continuous("x", shape=(n_x,), lb=0, ub=5)
        cy = rng.uniform(1, 5, n_y)
        cx = rng.uniform(1, 5, n_x)
        m.minimize(sum(cy[i] * ys[i] for i in range(n_y)) + sum(cx[j] * xs[j] for j in range(n_x)))
        m.subject_to(sum(xs[j] for j in range(n_x)) >= demand)
        for j in range(n_x):
            m.subject_to(xs[j] <= 5 * ys[j % n_y])

        r = solve_benders(m, time_limit=30)
        opt = _brute_force_optimum(cy, cx, demand, n_y, n_x)
        if r.bound is not None:
            assert r.bound <= opt + 1e-3, f"unsound bound {r.bound} > optimum {opt}"
        if r.status == "optimal":
            assert r.objective == pytest.approx(opt, abs=1e-3)


def test_no_false_optimal_certification():
    """On an 'optimal' exit, the bound must not exceed the incumbent objective."""
    rng = np.random.default_rng(321)
    for _ in range(10):
        m = dm.Model("c")
        y = m.binary("y", shape=(3,))
        x = m.continuous("x", shape=(2,), lb=0, ub=6)
        cy = rng.uniform(1, 4, 3)
        m.minimize(sum(cy[i] * y[i] for i in range(3)) + x[0] + x[1])
        m.subject_to(x[0] + x[1] >= 3)
        m.subject_to(x[0] <= 6 * y[0])
        m.subject_to(x[1] <= 6 * y[1])
        r = solve_benders(m, time_limit=30)
        if r.status == "optimal":
            assert r.bound is not None
            assert r.bound <= r.objective + 1e-4
            assert r.gap_certified


def test_bound_is_valid_for_maximize():
    """For maximize, the reported bound is an upper bound on the optimum."""
    m = dm.Model("maxb")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.maximize(4 * y + x)
    m.subject_to(x <= 3 * y)  # y=1 -> x<=3 -> 4+3 = 7
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(7.0, abs=1e-3)
    # Upper bound on a maximize never below the achieved objective.
    assert r.bound >= r.objective - 1e-4
