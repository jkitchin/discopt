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


def test_unbounded_recourse_var_with_feasibility_cuts_is_sound():
    """Regression: a recourse variable with a ``_BIG`` (1e20) open upper bound
    must not corrupt the complete-dual cut constant.

    ``np.isfinite(1e20)`` is ``True``, so a tiny reduced-cost noise at the
    sentinel bound would inject a ``rc * 1e20`` term into the cut. The guard
    skips ``|bound| >= _BIG``. Feasibility cuts fire (y=0 makes ``z`` infeasible),
    exercising the slack columns whose upper bound is the sentinel.
    """
    import warnings

    m = dm.Model("unb")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=1e20)  # open upper sentinel
    z = m.continuous("z", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(y + 0.001 * x + z)
    m.subject_to(z >= 3)
    m.subject_to(z <= 5 * y)  # infeasible when y=0 -> feasibility cut
    m.subject_to(x >= 2 * y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # large-bound advisory
        r = solve_benders(m, time_limit=30)
        mono = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert mono.objective is not None
    assert r.bound is not None
    assert r.bound <= mono.objective + 1e-3, f"unsound bound {r.bound} > optimum {mono.objective}"
    assert r.objective == pytest.approx(mono.objective, rel=1e-3)


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


# ── Phase 0 correctness (C3, C4) ──────────────────────────────


def test_c3_unbounded_recourse_reported():
    """A recourse LP that is unbounded below at a feasible master point means the
    full problem is unbounded — report it, do not stall (C3).

    ``min y - w`` with binary ``y`` first-stage and continuous ``w >= 0`` with no
    upper bound and no cost floor: the recourse ``min -w`` is unbounded below.
    """
    m = dm.Model("unb")
    y = m.binary("y")
    w = m.continuous("w", lb=0.0, ub=None)
    m.first_stage(y)
    m.minimize(y - w)
    m.subject_to(w >= y)
    r = solve_benders(m, time_limit=30)
    assert r.status == "unbounded"


# ── DC-S1 (decomposition-review, confirm-first) ───────────────
#
# DC-S1 was SUSPECTED against the pre-#409 tree: an unbounded-below recourse LP
# reported ``bound = _ETA_FLOOR (-1e12)`` (an invalid populated bound when the
# true optimum is below the floor). PR #409 ("Decomposition module remediation",
# 2026-07-03 — same day as the review) added (1) distinct ``unbounded`` recourse
# detection and (2) the T0.5 eta-floor-withholding guard. These tests pin BOTH
# halves of the finding so the fix cannot silently regress. DC-S1 is
# NOT-REPRODUCED on the current tree; these are the confirming evidence.


def test_dcs1_unbounded_recourse_withholds_bound():
    """The exact DC-S1 review repro (min y - x, x in [0, 1e30], x >= y).

    x's recourse is unbounded below, so the true optimum is -inf. The solver must
    report ``status="unbounded"`` with ``bound=None`` (never the -1e12 floor as a
    valid-looking bound).
    """
    m = dm.Model("dcs1")
    y = m.integer("y", shape=(1,), lb=0, ub=1)  # complicating -> master
    x = m.continuous("x", shape=(1,), lb=0.0, ub=1e30)  # recourse
    m.minimize(y[0] - x[0])
    m.subject_to(x[0] >= y[0], name="couple")
    r = solve_benders(m, max_iterations=50, time_limit=30)
    assert r.status == "unbounded"
    assert r.bound is None
    assert r.gap_certified is False


def test_dcs1_bounded_recourse_below_eta_floor_withholds_bound():
    """The second DC-S1 sub-case: a BOUNDED recourse whose true optimum is below
    the -1e12 eta floor must NOT report the floor as a valid bound.

    ``min y + x`` with ``x in [-5e12, 0]`` (bounded) has optimum -5e12 < -1e12.
    The T0.5 guard detects the eta variable still resting on the floor and
    withholds the bound (``bound=None``); the incumbent objective is still
    correct, and ``gap_certified`` stays ``False`` — no invalid certificate.
    """
    import warnings

    m = dm.Model("dcs1b")
    y = m.integer("y", shape=(1,), lb=0, ub=1)
    x = m.continuous("x", shape=(1,), lb=-5e12, ub=0.0)
    m.minimize(y[0] + x[0])
    m.subject_to(x[0] >= -5e12, name="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_benders(m, max_iterations=50, time_limit=30)
    # The incumbent is the true optimum, but the dual bound is withheld (floor).
    assert r.objective == pytest.approx(-5e12, rel=1e-6)
    assert r.bound is None, f"invalid populated bound {r.bound} (eta floor leaked)"
    assert r.gap_certified is False


def test_c4_progress_guard_no_spin(monkeypatch):
    """If the LP backend returns no duals, cuts cannot separate the master point;
    the solver must bail out quickly instead of spinning to max_iterations (C4)."""
    import discopt.decomposition.benders.solver as bsolver

    m = dm.Model("guard")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(x + y)
    m.subject_to(x + y >= 1)

    # Force the recourse dual generator to strip duals so every cut is degenerate.
    real_get = bsolver.__dict__  # noqa: F841 (kept for clarity)
    from discopt.solvers import lp_backend

    real_lp = lp_backend.get_lp_solver

    def stripped_lp(*a, **k):
        solver = real_lp(*a, **k)

        def wrapped(*aa, **kk):
            res = solver(*aa, **kk)
            res.dual_values = None
            res.reduced_costs = None
            return res

        return wrapped

    monkeypatch.setattr(lp_backend, "get_lp_solver", stripped_lp)
    r = solve_benders(m, time_limit=30, max_iterations=500)
    # Must terminate quickly via the stall guard, not run all 500 iterations.
    assert r.wall_time < 20
