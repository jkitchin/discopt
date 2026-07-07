"""Curated adversarial correctness examples for the decomposition solvers.

Unlike the randomized batteries elsewhere, these are *hand-crafted* instances
with **analytically known optima**, each targeting a specific correctness hazard
that broke (or could break) Benders / Generalized Benders. They double as
permanent regression tests and as a readable catalogue of the edge cases the
cut machinery must handle.

Every test asserts three things:
  1. the reported objective equals the hand-computed optimum,
  2. the reported ``bound`` is *sound* (a valid lower bound for minimize, upper
     bound for maximize — never on the wrong side of the optimum), and
  3. a cross-check against the monolithic ``Model.solve()`` (an independent
     code path).

Hazards covered (see each test's docstring):
  - degenerate recourse where a variable bound coincides with a coupling row
    (the round-1 / round-2 anchoring bugs),
  - recourse optimum pinned at a variable's own bound (reduced-cost term),
  - equality coupling (free-sign / split-row duals),
  - feasibility / no-good cuts when recourse is infeasible for some master
    points,
  - GBD with tight *nonlinear* active constraints (large multipliers),
  - GBD with a linear equality in the recourse and a nonlinear objective,
  - maximize (sense flip), and
  - nonconvex models (the convexity gate must withhold the bound).
"""

import discopt.modeling as dm
import pytest

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
    pytest.mark.skipif(not (POUNCE_AVAILABLE or HAS_HIGHS), reason="no LP/MILP backend"),
    pytest.mark.correctness,
]

ABS = 3e-3


def _assert_min(model_fn, opt, *, certified=True):
    """Solve via Benders decomposition and assert a sound, correct minimization."""
    r = model_fn().solve(decomposition="benders", time_limit=60)
    mono = model_fn().solve(time_limit=60)
    assert r.status == "optimal", f"status {r.status}"
    assert r.objective == pytest.approx(opt, abs=ABS), f"obj {r.objective} != {opt}"
    assert mono.objective == pytest.approx(opt, abs=ABS), f"monolithic disagrees: {mono.objective}"
    if r.bound is not None:
        assert r.bound <= opt + 1e-3 * (1 + abs(opt)), f"unsound lower bound {r.bound} > opt {opt}"
    if certified:
        assert r.gap_certified
    return r


def _assert_max(model_fn, opt):
    """Solve via Benders decomposition and assert a sound, correct maximization."""
    r = model_fn().solve(decomposition="benders", time_limit=60)
    mono = model_fn().solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=ABS)
    assert mono.objective == pytest.approx(opt, abs=ABS)
    if r.bound is not None:
        assert r.bound >= opt - 1e-3 * (1 + abs(opt)), f"unsound upper bound {r.bound} < opt {opt}"
    return r


# ══════════════════════════════════════════════════════════════════════
# Classical Benders (linear recourse)
# ══════════════════════════════════════════════════════════════════════


def test_cb1_bound_active_degenerate():
    """Recourse optimum where a coupling row and a variable bound coincide.

    ``min y - v`` s.t. ``v <= 10y``, ``v in [0,10]``. At y=1 both ``v <= 10`` (row)
    and ``v <= 10`` (bound) bind; an interior-point LP splits the marginal, so a
    row-dual-only cut anchors at -5 instead of the true -10. The complete-dual
    cut restores the bound term. Optimum: y=1, v=10 -> -9.
    """

    def m():
        mo = dm.Model("cb1")
        y = mo.binary("y")
        v = mo.continuous("v", lb=0, ub=10)
        mo.first_stage(y)
        mo.minimize(y - v)
        mo.subject_to(v <= 10 * y)
        return mo

    _assert_min(m, -9.0)


def test_cb2_recourse_pinned_at_own_bound():
    """Recourse optimum pinned at the variable's *own* upper bound (not a row).

    ``min 2y - x`` s.t. ``x <= 10y``, ``x in [0,4]``. At y=1 the binding limit is
    the variable bound x<=4, whose reduced cost (not a row dual) carries the
    sensitivity; the cut's reduced-cost term must capture it. Optimum: y=1, x=4
    -> -2.
    """

    def m():
        mo = dm.Model("cb2")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=4)
        mo.first_stage(y)
        mo.minimize(2 * y - x)
        mo.subject_to(x <= 10 * y)
        return mo

    _assert_min(m, -2.0)


def test_cb3_facility_with_feasibility_cut():
    """Recourse is infeasible for y=0, forcing a feasibility cut.

    ``min 2y + x1 + x2`` s.t. ``x1+x2>=3``, ``x_i <= 5y``. y=0 -> recourse
    infeasible (a feasibility cut excludes it); y=1 -> ship 3 at cost 3, total 5.
    """

    def m():
        mo = dm.Model("cb3")
        y = mo.binary("y")
        x1 = mo.continuous("x1", lb=0, ub=10)
        x2 = mo.continuous("x2", lb=0, ub=10)
        mo.minimize(2 * y + x1 + x2)
        mo.subject_to(x1 + x2 >= 3)
        mo.subject_to(x1 <= 5 * y)
        mo.subject_to(x2 <= 5 * y)
        return mo

    _assert_min(m, 5.0)


def test_cb4_continuous_two_stage():
    """Pure-continuous two-stage LP with an explicit first-stage annotation.

    ``min u + 2v`` s.t. ``u+v>=4``, u first-stage. u=4, v=0 -> 4.
    """

    def m():
        mo = dm.Model("cb4")
        u = mo.continuous("u", lb=0, ub=10)
        v = mo.continuous("v", lb=0, ub=10)
        mo.first_stage(u)
        mo.minimize(u + 2 * v)
        mo.subject_to(u + v >= 4)
        return mo

    _assert_min(m, 4.0)


def test_cb5_covering_with_lp_ip_gap():
    """Set-covering master where the LP relaxation is fractional; needs >=2 open.

    ``min sum y_i + 0.1 sum x_j`` s.t. ``sum x_j >= 3``, ``x_j <= 2 y_j``. One
    facility (cap 2) is infeasible; two (cap 4) cover 3 at cost 2 + 0.3 = 2.3.
    Exercises multiple feasibility cuts before convergence.
    """

    def m():
        mo = dm.Model("cb5")
        y = mo.binary("y", shape=(3,))
        x = mo.continuous("x", shape=(3,), lb=0, ub=5)
        for i in range(3):
            mo.first_stage(y[i])
        mo.minimize(sum(y[i] for i in range(3)) + 0.1 * sum(x[j] for j in range(3)))
        mo.subject_to(sum(x[j] for j in range(3)) >= 3)
        for j in range(3):
            mo.subject_to(x[j] <= 2 * y[j])
        return mo

    _assert_min(m, 2.3)


def test_cb6_maximize_with_binding_coupling():
    """Maximize: the reported bound is an *upper* bound on the optimum.

    ``max 4y + x`` s.t. ``x <= 3y``, ``x in [0,5]``. y=1 -> x=3 -> 7.
    """

    def m():
        mo = dm.Model("cb6")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=5)
        mo.first_stage(y)
        mo.maximize(4 * y + x)
        mo.subject_to(x <= 3 * y)
        return mo

    _assert_max(m, 7.0)


def test_cb7_globally_infeasible():
    """Every master point yields an infeasible recourse -> status infeasible."""

    def m():
        mo = dm.Model("cb7")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=1)
        mo.first_stage(y)
        mo.minimize(x + y)
        mo.subject_to(x >= 2)  # x in [0,1] but x>=2 -> infeasible for all y
        return mo

    r = m().solve(decomposition="benders", time_limit=30)
    assert r.status == "infeasible"
    assert r.bound is None


# ══════════════════════════════════════════════════════════════════════
# Generalized Benders (convex-NLP recourse)
# ══════════════════════════════════════════════════════════════════════


def test_gb1_quadratic_recourse():
    """Baseline convex quadratic recourse. ``min x^2 + y`` s.t. ``x+y>=1``.

    y=0 -> x=1 -> 1 ; y=1 -> x=0 -> 1. Optimum 1.
    """

    def m():
        mo = dm.Model("gb1")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=5)
        mo.first_stage(y)
        mo.minimize(x**2 + y)
        mo.subject_to(x + y >= 1)
        return mo

    _assert_min(m, 1.0)


def test_gb2_recourse_pinned_at_box_bound():
    """GBD recourse optimum pinned where the box and coupling bound coincide.

    ``min (x-10)^2 + y`` s.t. ``x <= 5y``, ``x in [0,5]``. At y=1 the objective is
    decreasing on [0,5], so x=5 where the variable bound and ``x<=5y`` both bind —
    the GBD analogue of the round-1 degeneracy. The Lagrangian-dual cut's box-min
    term must absorb the bound multiplier. y=1 -> 25+1 = 26.
    """

    def m():
        mo = dm.Model("gb2")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=5)
        mo.first_stage(y)
        mo.minimize((x - 10) ** 2 + y)
        mo.subject_to(x <= 5 * y)
        return mo

    _assert_min(m, 26.0)


def test_gb3_tight_nonlinear_active_constraint():
    """A *nonlinear* constraint binds at the recourse optimum (large multiplier).

    ``min 3y - 2x`` s.t. ``x^2 <= 4y``, ``x in [0,5]``. y=1 -> x=2 (x^2=4 active)
    -> 3-4 = -1 ; y=0 -> x=0 -> 0. Exercises the constraint-multiplier path in the
    Lagrangian cut slope. Optimum -1.
    """

    def m():
        mo = dm.Model("gb3")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=5)
        mo.first_stage(y)
        mo.minimize(3 * y - 2 * x)
        mo.subject_to(x * x <= 4 * y)
        return mo

    _assert_min(m, -1.0)


def test_gb4_linear_equality_in_recourse():
    """Linear equality coupling (free-sign multiplier) with a nonlinear objective.

    ``min (x1-2)^2 + (x2-1)^2 + 2y`` s.t. ``x1 + x2 == 2y``. y=0 -> x=0 -> 5 ;
    y=1 -> project (2,1) onto x1+x2=2 -> (1.5,0.5) -> 0.5 + 2 = 2.5. Optimum 2.5.
    """

    def m():
        mo = dm.Model("gb4")
        y = mo.binary("y")
        x = mo.continuous("x", shape=(2,), lb=0, ub=5)
        mo.first_stage(y)
        mo.minimize((x[0] - 2) ** 2 + (x[1] - 1) ** 2 + 2 * y)
        mo.subject_to(x[0] + x[1] == 2 * y)
        return mo

    _assert_min(m, 2.5)


def test_gb5_infeasible_recourse_no_good_cut():
    """GBD recourse infeasible at y=0 -> no-good cut on the 0/1 master.

    ``min 2y + x^2`` s.t. ``x>=2``, ``x<=5y``. y=0 infeasible; y=1 -> x=2 -> 4+2 = 6.
    """

    def m():
        mo = dm.Model("gb5")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=5)
        mo.first_stage(y)
        mo.minimize(2 * y + x * x)
        mo.subject_to(x >= 2)
        mo.subject_to(x <= 5 * y)
        return mo

    _assert_min(m, 6.0)


def test_gb6_multibinary_nonlinear_coupling():
    """Two binaries, separable nonlinear recourse; independent closed-form oracle.

    ``min (x0-2)^2 + (x1-2)^2 + y0 + 2y1`` s.t. ``x_j <= 3 y_j``. For fixed y the
    recourse is x_j = min(2, 3 y_j), so the optimum enumerates to y=(1,1),
    x=(2,2) -> 0 + 1 + 2 = 3.
    """

    def m():
        mo = dm.Model("gb6m")
        y = mo.binary("y", shape=(2,))
        x = mo.continuous("x", shape=(2,), lb=0, ub=5)
        mo.first_stage(y[0])
        mo.first_stage(y[1])
        mo.minimize((x[0] - 2) ** 2 + (x[1] - 2) ** 2 + y[0] + 2 * y[1])
        mo.subject_to(x[0] <= 3 * y[0])
        mo.subject_to(x[1] <= 3 * y[1])
        return mo

    # Independent closed-form oracle over the 4 binary points.
    cy = [1.0, 2.0]
    best = min(
        sum((min(2.0, 3.0 * yv[j]) - 2.0) ** 2 for j in range(2))
        + sum(cy[i] * yv[i] for i in range(2))
        for yv in [(0, 0), (0, 1), (1, 0), (1, 1)]
    )
    assert best == 3.0  # sanity on the oracle itself
    _assert_min(m, best)


def test_gb7_maximize_concave():
    """Concave maximize (convex internally); bound is a valid upper bound.

    ``max 3y - x^2`` s.t. ``x >= 2y``. y=1 -> x=2 -> -1 ; y=0 -> x=0 -> 0. Optimum 0.
    """

    def m():
        mo = dm.Model("gb7x")
        y = mo.binary("y")
        x = mo.continuous("x", lb=0, ub=4)
        mo.first_stage(y)
        mo.maximize(3 * y - x * x)
        mo.subject_to(x >= 2 * y)
        return mo

    _assert_max(m, 0.0)


def test_gb8_nonconvex_withholds_bound():
    """A genuinely nonconvex (cubic) recourse: the gate must withhold the bound.

    ``min x^3 - 4x^2 + 2y`` s.t. ``x+y>=1``, ``x in [0,4]``. The incumbent must
    still match the monolithic optimum, but ``bound`` must be None (no valid
    convex underestimator) so the correctness gate is never threatened.
    """
    mo = dm.Model("gb8nc")
    y = mo.binary("y")
    x = mo.continuous("x", lb=0, ub=4)
    mo.first_stage(y)
    mo.minimize(x**3 - 4 * x * x + 2 * y)
    mo.subject_to(x + y >= 1)
    r = mo.solve(decomposition="benders", time_limit=60)
    mono = dm.Model("gb8m")
    yy = mono.binary("y")
    xx = mono.continuous("x", lb=0, ub=4)
    mono.first_stage(yy)
    mono.minimize(xx**3 - 4 * xx * xx + 2 * yy)
    mono.subject_to(xx + yy >= 1)
    ref = mono.solve(time_limit=60)
    assert ref.objective is not None
    assert r.objective == pytest.approx(ref.objective, abs=1e-2)
    assert r.bound is None
    assert not r.gap_certified


# ══════════════════════════════════════════════════════════════════════
# Randomized soundness attacks on the Phase 0-5 code paths (fixed seeds).
# Each asserts the cardinal invariant: a reported bound is valid, and a
# *certified* answer is never wrong. Oracle = monolithic Model.solve().
# ══════════════════════════════════════════════════════════════════════

import numpy as np  # noqa: E402
from discopt.decomposition.benders import solve_benders  # noqa: E402
from discopt.decomposition.benders.solver import BendersConfig  # noqa: E402
from discopt.decomposition.lagrangian import solve_lagrangian  # noqa: E402


def _rand_two_stage(seed):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(1, 4))
    m = dm.Model(f"ts{seed}")
    b = [m.binary(f"b{k}") for k in range(K)]
    obj = 0
    for k in range(K):
        nk = int(rng.integers(1, 3))
        x = m.continuous(f"x{k}", shape=(nk,), lb=0, ub=10)
        d = float(rng.integers(1, 6))
        m.subject_to(sum(x[i] for i in range(nk)) >= d * b[k])
        for i in range(nk):
            m.subject_to(x[i] <= 10 * b[k])
            obj = obj + float(rng.uniform(0.5, 3.0)) * x[i]
        obj = obj + float(rng.integers(1, 5)) * b[k]
    need = 1 if K < 2 else int(rng.integers(1, K + 1))
    m.subject_to(sum(b) >= need)
    m.minimize(obj)
    return m


@pytest.mark.parametrize("seed", range(8))
def test_rand_multicut_sound_and_deterministic(seed):
    """Multicut/single/threads/in-out all reproduce the monolithic optimum with a
    valid lower bound, and the two backends agree bit-for-bit."""
    opt = float(_rand_two_stage(seed).solve(time_limit=30).objective)
    variants = {
        "mc_seq": dict(multicut=True, backend="sequential"),
        "single": dict(multicut=False, backend="sequential"),
        "mc_thr": dict(multicut=True, backend="threads"),
        "inout": dict(multicut=True, stabilization="inout"),
    }
    res = {}
    for name, kw in variants.items():
        r = solve_benders(_rand_two_stage(seed), config=BendersConfig(time_limit=30, **kw))
        res[name] = r
        if r.bound is not None:
            assert r.bound <= opt + 1e-2, f"{name}: unsound bound {r.bound} > opt {opt}"
        if r.status == "optimal":
            assert r.objective == pytest.approx(opt, abs=ABS), f"{name} wrong certified opt"
    assert res["mc_seq"].objective == res["mc_thr"].objective  # determinism
    assert res["mc_seq"].bound == res["mc_thr"].bound


def _rand_gap(seed):
    rng = np.random.default_rng(seed)
    A, T = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    m = dm.Model(f"gap{seed}")
    x = m.binary("x", shape=(A * T,))
    cost = [[float(rng.integers(1, 9)) for _ in range(T)] for _ in range(A)]
    w = [[float(rng.integers(1, 5)) for _ in range(T)] for _ in range(A)]
    cap = [float(rng.integers(T, 2 * T + 2)) for _ in range(A)]
    m.minimize(sum(cost[a][t] * x[a * T + t] for a in range(A) for t in range(T)))
    for a in range(A):
        m.subject_to(sum(w[a][t] * x[a * T + t] for t in range(T)) <= cap[a])
    for t in range(T):
        c = sum(x[a * T + t] for a in range(A)) == 1  # equality coupling (free multiplier)
        m.subject_to(c)
        m.mark_coupling(c)
    return m


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("method", ["subgradient", "bundle", "kelley"])
def test_rand_lagrangian_dual_is_valid_lower_bound(seed, method):
    """The Lagrangian dual bound never exceeds the optimum (equality coupling
    exercises the free-multiplier path), across all three dual methods, and the
    two backends agree bit-for-bit."""
    mono = _rand_gap(seed).solve(time_limit=30)
    if mono.status != "optimal":
        pytest.skip("degenerate instance")
    opt = float(mono.objective)
    seq = solve_lagrangian(_rand_gap(seed), method=method, backend="sequential", time_limit=15)
    thr = solve_lagrangian(_rand_gap(seed), method=method, backend="threads", time_limit=15)
    for r in (seq, thr):
        if r.bound is not None:
            assert r.bound <= opt + 1e-2, f"{method}: unsound dual bound {r.bound} > opt {opt}"
        if r.status == "optimal":
            assert r.objective == pytest.approx(opt, abs=ABS)
        if r.objective is not None:
            assert r.objective >= opt - ABS, "recovered incumbent below optimum (infeasible?)"
    assert seq.bound == thr.bound and seq.objective == thr.objective  # determinism


@pytest.mark.parametrize("fail_prob", [0.3, 0.6, 0.9])
def test_c1_flaky_nlp_never_certifies_wrong(fail_prob, monkeypatch):
    """A recourse NLP that fails randomly must NEVER yield a wrong *certified*
    answer — the headline C1 invariant, attacked with high failure rates."""
    import discopt.solvers.nlp_pounce as npn

    m = dm.Model("c1adv")
    b = [m.binary(f"b{k}") for k in range(2)]
    obj = 0
    for k in range(2):
        x = m.continuous(f"x{k}", lb=0, ub=5)
        m.subject_to(x * x <= 9 * b[k])
        m.subject_to(x >= 1.5 * b[k])
        obj = obj + x + (k + 1) * b[k]
    m.subject_to(b[0] + b[1] >= 1)
    m.minimize(obj)
    opt = float(m.solve(time_limit=30).objective)

    real = npn.solve_nlp
    rng = np.random.default_rng(int(fail_prob * 100))

    def flaky(evaluator, x0, *a, **k):
        if rng.random() < fail_prob:
            raise RuntimeError("adversarial transient NLP failure")
        return real(evaluator, x0, *a, **k)

    monkeypatch.setattr(npn, "solve_nlp", flaky)
    # Rebuild a fresh model (solve may consume state); reuse m is fine here.
    r = solve_benders(m, time_limit=25)
    if r.gap_certified and r.status == "optimal" and r.objective is not None:
        assert r.objective == pytest.approx(opt, abs=ABS), (
            f"flaky NLP produced a WRONG certified answer {r.objective} (true {opt})"
        )
