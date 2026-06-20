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
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
except ImportError:
    POUNCE_AVAILABLE = False
try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = [
    pytest.mark.skipif(not (POUNCE_AVAILABLE or HAS_HIGHS), reason="no LP/MILP backend available"),
    pytest.mark.correctness,
]


def _brute_force_optimum(cy, cx, demand, n_y, n_x):
    # Independent oracle via scipy (no discopt solver, no highspy package).
    from scipy.optimize import linprog

    best = np.inf
    for yv in itertools.product([0, 1], repeat=n_y):
        rows = [-np.ones(n_x)]
        rhs = [-float(demand)]
        for j in range(n_x):
            row = np.zeros(n_x)
            row[j] = 1.0
            rows.append(row)
            rhs.append(5.0 * yv[j % n_y])
        res = linprog(cx, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(0, 5)] * n_x)
        if res.success:
            best = min(best, float(cy @ np.array(yv) + res.fun))
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


@pytest.mark.parametrize("vub", [10.0, 1000.0, 1e6])
def test_bound_active_recourse_is_sound(vub):
    """Regression: recourse optimum set by a variable bound coinciding with a
    coupling row.

    ``min y - v`` s.t. ``v <= vub*y``, ``v in [0, vub]``. At y=1 the row
    ``v <= vub`` and the bound ``v <= vub`` both bind; an interior-point LP
    splits the marginal between them, so the recourse *row* dual is only half
    the true sensitivity. A cut from the row duals alone lands at ``-vub/2``
    instead of the true ``-vub`` and prunes the optimum — the complete-dual cut
    (row duals + variable reduced costs) restores the missing bound term and
    stays sound.
    """
    m = dm.Model("bnd")
    y = m.binary("y")
    v = m.continuous("v", lb=0, ub=vub)
    m.first_stage(y)
    m.minimize(y - v)
    m.subject_to(v <= vub * y)
    r = solve_benders(m, time_limit=30)
    opt = 1.0 - vub  # y=1, v=vub
    assert r.objective == pytest.approx(opt, rel=1e-4)
    assert r.bound is not None
    assert r.bound <= opt + 1e-3 * (1 + abs(opt)), f"unsound bound {r.bound} > optimum {opt}"


def test_equality_coupling_suboptimal_primal_is_sound():
    """Regression: equality coupling + a degenerate recourse where an
    interior-point solve can return a slightly *suboptimal* primal.

    With an equality constraint (split into two ``<=`` rows that both bind) and
    tight capacity coupling, POUNCE occasionally returns a recourse primal a hair
    above the true optimum together with a near-zero coupling-row dual, yielding a
    nearly-flat optimality cut ``eta >= Q_suboptimal``. Anchored at that primal it
    over-cuts (reported lower bound exceeds the incumbent — a false ``optimal``
    certificate). The complete-dual cut (``<= Q_true`` by weak duality) stays
    sound. Instance is the seed-555 t=37 case found by adversarial testing.
    """
    cy = np.array([2.83085168, 1.49786461])
    cx = np.array([1.48384495, 2.21570648, 2.6325476])
    cap = np.array([2.16214416, 2.85835275, 3.08819464])
    dem = 4.746103
    m = dm.Model("eqc")
    y = m.binary("y", shape=(2,))
    x = m.continuous("x", shape=(3,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(
        sum(float(cy[i]) * y[i] for i in range(2)) + sum(float(cx[j]) * x[j] for j in range(3))
    )
    m.subject_to(sum(x[j] for j in range(3)) == dem)
    for j in range(3):
        m.subject_to(x[j] <= float(cap[j]) * y[j % 2])
    r = solve_benders(m, time_limit=60)
    assert r.status == "optimal"
    assert r.bound is not None
    # The reported lower bound must never exceed the incumbent objective.
    assert r.bound <= r.objective + 1e-4, f"unsound bound {r.bound} > incumbent {r.objective}"
    assert r.objective == pytest.approx(12.84146, abs=1e-3)


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
