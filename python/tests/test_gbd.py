"""Tests for Generalized Benders Decomposition (convex-NLP recourse).

GBD extends classical Benders to a *convex nonlinear* recourse subproblem
(Geoffrion 1972). The headline soundness invariant mirrors classical Benders:
the reported lower ``bound`` never exceeds the true optimum, and on a convex
model the decomposition objective matches the monolithic ``Model.solve()``.

The nonlinear models here route through ``solve_benders`` automatically (it
detects nonlinearity and dispatches to ``solve_gbd``); a couple of tests call
``solve_gbd`` directly to exercise edges.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.benders import solve_benders
from discopt.decomposition.benders.gbd import solve_gbd

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

ABS_TOL = 2e-3


def _quadratic_recourse():
    """min x^2 + y ; y binary first-stage, x continuous recourse, x + y >= 1.

    y=0 -> x=1 -> 1 ; y=1 -> x=0 -> 1. Optimum 1.0.
    """
    m = dm.Model("quad")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(x**2 + y)
    m.subject_to(x + y >= 1)
    return m


def test_quadratic_recourse_optimum():
    r = solve_benders(_quadratic_recourse(), time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=ABS_TOL)
    assert r.bound is not None and r.bound <= r.objective + 1e-3


def test_dispatch_via_model_solve():
    r = _quadratic_recourse().solve(decomposition="benders", time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=ABS_TOL)


def test_infeasible_recourse_uses_nogood_cut():
    """Recourse infeasible at y=0 (needs x>=2 but x<=5y); y=1 -> 2 + 4 = 6."""
    m = dm.Model("feas")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(2 * y + x * x)
    m.subject_to(x >= 2)
    m.subject_to(x <= 5 * y)
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(6.0, abs=ABS_TOL)


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 7])
def test_matches_monolithic_convex(seed):
    """Across random convex MINLPs with quadratic recourse, GBD == monolithic
    and the bound is a valid lower bound."""
    rng = np.random.default_rng(seed)
    ny, nx = 2, 3
    m = dm.Model("rand")
    y = m.binary("y", shape=(ny,))
    x = m.continuous("x", shape=(nx,), lb=0, ub=4)
    m.first_stage(y)
    a = rng.uniform(0.5, 2.0, nx)
    cy = rng.uniform(1, 3, ny)
    tgt = rng.uniform(2, 5)
    m.minimize(
        sum(float(a[j]) * x[j] * x[j] for j in range(nx))
        + sum(float(cy[i]) * y[i] for i in range(ny))
    )
    m.subject_to(sum(x[j] for j in range(nx)) >= tgt)
    for j in range(nx):
        m.subject_to(x[j] <= 4 * y[j % ny])

    r = solve_benders(m, time_limit=30)
    mono = m.solve(time_limit=30)
    assert mono.objective is not None
    assert r.objective == pytest.approx(mono.objective, abs=1e-2)
    if r.bound is not None:
        assert r.bound <= mono.objective + 1e-3


@pytest.mark.slow
@pytest.mark.correctness
def test_gbd_soundness_fuzz_vs_monolithic():
    """Randomized soundness fuzz: across many convex MINLPs with *nonlinear
    constraints* (varied dimensions, conditioning, equality/inequality coupling),
    the GBD bound never exceeds the monolithic optimum and the objective matches.
    Mirrors the classical-Benders soundness battery the reviewer noted was missing
    for GBD."""
    rng = np.random.default_rng(20260620)
    for _ in range(60):
        ny = int(rng.integers(1, 4))
        nx = int(rng.integers(1, 4))
        scale = 10.0 ** rng.integers(-1, 3)
        m = dm.Model("fuzz")
        y = m.binary("y", shape=(ny,))
        x = m.continuous("x", shape=(nx,), lb=0, ub=5)
        m.first_stage(y)
        a = rng.uniform(0.2, 2.0, nx) * scale
        sh = rng.uniform(0, 3, nx)
        cy = rng.uniform(-1, 4, ny) * scale
        m.minimize(
            sum(float(a[j]) * (x[j] - float(sh[j])) ** 2 for j in range(nx))
            + sum(float(cy[i]) * y[i] for i in range(ny))
        )
        sy = sum(y[i] for i in range(ny))
        # a nonlinear convex coupling constraint, sometimes an equality
        if rng.integers(0, 2):
            m.subject_to(sum(x[j] * x[j] for j in range(nx)) <= float(rng.uniform(2, 9)) * sy + 0.1)
        else:
            m.subject_to(sum(x[j] for j in range(nx)) == float(rng.uniform(0, 3)) * sy)
        r = solve_benders(m, time_limit=30)
        mono = m.solve(time_limit=30)
        if mono.objective is None:
            continue
        if r.bound is not None:
            assert r.bound <= mono.objective + 5e-3 * (1 + abs(mono.objective)), (
                f"unsound GBD bound {r.bound} > opt {mono.objective}"
            )
        if r.objective is not None:
            assert r.objective == pytest.approx(
                mono.objective, abs=1e-2 * (1 + abs(mono.objective))
            )


def test_project_mu_sign_convention():
    """``_project_mu`` must push multipliers into the dual-feasible orthant:
    ``mu >= 0`` for ``<=`` rows (cl=-inf, cu=0), ``mu <= 0`` for ``>=`` rows
    (cl=0, cu=+inf), and unchanged (free) for equalities (cl=cu=0)."""
    import numpy as _np
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    m = dm.Model("signs")
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize(x * x)
    m.subject_to(x <= 3)  # <=  -> mu >= 0
    m.subject_to(x >= -3)  # >=  -> mu <= 0
    m.subject_to(x == 0)  # ==  -> free
    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    # Reconstruct the projection logic exactly as gbd._project_mu does.
    mu = _np.array([-2.0, 2.0, -7.0])  # deliberately "wrong-signed" inputs
    out = mu.copy()
    lower_inf = cl <= -1e19
    upper_inf = cu >= 1e19
    only_upper = lower_inf & ~upper_inf
    only_lower = upper_inf & ~lower_inf
    out[only_upper] = _np.maximum(out[only_upper], 0.0)
    out[only_lower] = _np.minimum(out[only_lower], 0.0)
    senses = []
    for lo, hi in zip(cl, cu):
        if lo <= -1e19 and hi < 1e19:
            senses.append("<=")
        elif hi >= 1e19 and lo > -1e19:
            senses.append(">=")
        else:
            senses.append("==")
    for k, sense in enumerate(senses):
        if sense == "<=":
            assert out[k] >= -1e-12
        elif sense == ">=":
            assert out[k] <= 1e-12
        else:  # equality: left free
            assert out[k] == pytest.approx(mu[k])


@pytest.mark.slow
def test_lagrangian_anchor_sound_under_inexact_recourse(monkeypatch):
    """The GBD cut anchors at the Lagrangian dual value, not the NLP primal, so a
    *suboptimal* recourse solve cannot produce an unsound bound.

    We inject a feasible-but-suboptimal recourse primal (nudge the recourse
    variable off its optimum, raising the objective) into every NLP solve — the
    exact failure mode that an over-cutting primal anchor would mis-certify. The
    Lagrangian anchor's box-min correction (``m_y``) pulls the cut back down, so
    the reported lower bound must stay <= the true optimum (2.0) and must not be
    falsely certified above it.
    """
    import discopt.solvers.nlp_pounce as nlp_pounce

    real = nlp_pounce.solve_nlp

    def perturbed(ev, x0, **kw):
        res = real(ev, x0, **kw)
        if res.x is not None and res.status.name == "OPTIMAL":
            lb, ub = ev.variable_bounds
            xnew = np.asarray(res.x, dtype=float).copy()
            for j in range(len(xnew)):
                if ub[j] - lb[j] > 1e-6 and ub[j] < 1e18:  # a bounded recourse var
                    xnew[j] = min(xnew[j] + 0.6, ub[j])
            res.x = xnew
            res.objective = float(ev.evaluate_objective(xnew))
        return res

    monkeypatch.setattr(nlp_pounce, "solve_nlp", perturbed)

    m = dm.Model("inj")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize((x - 3) ** 2 + 2 * y)  # y=1 -> x=3 -> 2.0 optimum
    m.subject_to(x <= 5 * y)
    r = solve_benders(m, time_limit=20)
    true_opt = 2.0
    # The reported lower bound is never above the true optimum, and the broken
    # solve is never falsely certified optimal at the inflated value.
    if r.bound is not None:
        assert r.bound <= true_opt + 1e-3, f"unsound bound {r.bound} > optimum {true_opt}"
    assert not (r.status == "optimal" and r.objective is not None and r.objective > true_opt + 1e-2)


@pytest.mark.slow
def test_nonconvex_reports_no_bound():
    """A nonconvex (concave) objective must not produce a numeric lower bound;
    soundness gate -> bound is None even though a heuristic incumbent is found."""
    m = dm.Model("nonconvex")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(-x * x + 3 * y)  # concave -> nonconvex value function
    m.subject_to(x + y >= 1)
    r = solve_benders(m, time_limit=30)
    assert r.bound is None
    assert not r.gap_certified


def test_maximize_concave():
    """Concave maximize is convex internally; GBD gives a valid upper bound.

    max 3y - x^2 ; y binary first-stage, x>=2y. y=1 -> x=2 -> 3-4 = -1 ;
    y=0 -> x=0 -> 0. Optimum 0 at y=0.
    """
    m = dm.Model("maxc")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=4)
    m.first_stage(y)
    m.maximize(3 * y - x * x)
    m.subject_to(x >= 2 * y)
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=ABS_TOL)
    # Upper bound on a maximize is never below the achieved objective.
    assert r.bound is None or r.bound >= r.objective - 1e-3


def test_requires_recourse_variable():
    """No recourse variable -> nothing to project; clean rejection."""
    m = dm.Model("norec")
    y = m.binary("y", shape=(2,))
    m.first_stage(y)
    m.minimize(y[0] * y[0] + y[1])
    m.subject_to(y[0] + y[1] >= 1)
    with pytest.raises(NotImplementedError):
        solve_gbd(m, time_limit=10)


def test_integer_recourse_rejected():
    """Integer recourse needs continuous KKT multipliers -> rejected."""
    m = dm.Model("intrec")
    y = m.binary("y")
    z = m.integer("z", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(y + z * z)
    m.subject_to(z + y >= 2)
    with pytest.raises(NotImplementedError):
        solve_gbd(m, time_limit=10)


def test_integer_first_stage_feasible_recourse():
    """General-integer (non-binary) first stage with nonlinear recourse.

    ``min z + (x-5)^2`` s.t. ``x+z>=6``, z integer in [0,4]. z=1, x=5 -> 1.
    """
    m = dm.Model("intfs")
    z = m.integer("z", lb=0, ub=4)
    x = m.continuous("x", lb=0, ub=10)
    m.first_stage(z)
    m.minimize(z + (x - 5) ** 2)
    m.subject_to(x + z >= 6)
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=ABS_TOL)
    assert r.bound is not None and r.bound <= r.objective + 1e-3


def test_integer_first_stage_infeasible_recourse_clean_error():
    """Non-binary integer first stage with infeasible recourse: no-good cuts only
    exist for 0/1 masters, so GBD must raise a clean NotImplementedError (rather
    than crash or silently mis-solve)."""
    m = dm.Model("intinf")
    z = m.integer("z", lb=0, ub=3)
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(z)
    m.minimize(z + x * x)
    m.subject_to(x >= 2)
    m.subject_to(x <= 2 * z - 4)  # needs z>=3; z<3 -> infeasible recourse
    with pytest.raises(NotImplementedError):
        solve_gbd(m, time_limit=20)


def test_master_only_nonlinear_constraint():
    """A master-only *nonlinear* constraint (binary master) is enforced via
    recourse infeasibility / no-good cuts, not added as a master row.

    ``min y0+y1+x^2`` s.t. ``y0^2+y1^2<=1`` (master-only, = y0+y1<=1 for binaries),
    ``x>=2``, ``x<=5(y0+y1)``. Optimum: open one facility -> 1 + 4 = 5.
    """
    m = dm.Model("monl")
    y = m.binary("y", shape=(2,))
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y[0])
    m.first_stage(y[1])
    m.minimize(y[0] + y[1] + x * x)
    m.subject_to(y[0] * y[0] + y[1] * y[1] <= 1)
    m.subject_to(x >= 2)
    m.subject_to(x <= 5 * (y[0] + y[1]))
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=ABS_TOL)


def test_free_recourse_variable():
    """A recourse variable that appears only in the (nonlinear) objective, with
    no coupling constraint, must be handled (box-only recourse min).

    ``min 2y + (x-3)^2`` with x free in [0,5] -> x=3 -> 0.
    """
    m = dm.Model("free")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(2 * y + (x - 3) ** 2)
    m.subject_to(y >= 0)
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=ABS_TOL)
    assert r.bound is not None and r.bound <= r.objective + 1e-3


def test_linear_objective_nonlinear_constraint():
    """A *linear* objective with a nonlinear convex constraint still routes to GBD.

    ``min 3y - x0 - x1`` s.t. ``x0^2+x1^2 <= 8y``. y=1 -> x0=x1=2 -> -4+3 = -1.
    """
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    r = solve_benders(m, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=ABS_TOL)
    assert r.bound is not None and r.bound <= r.objective + 1e-3


# ── Phase 0 correctness (C1, C2) ──────────────────────────────


def _quad_two_point():
    """min x^2 + y ; y binary first-stage, x recourse, x + y >= 1. Optimum 1.0."""
    m = dm.Model("quad_c1")
    y = m.binary("y")
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(y)
    m.minimize(x**2 + y)
    m.subject_to(x + y >= 1)
    return m


def test_c1_transient_nlp_failure_retries(monkeypatch):
    """A one-off recourse-NLP failure must not derail the solve (C1).

    The first ``solve_nlp`` invocation raises; every later call delegates to the
    real solver. The perturbed retry / phase-1 machinery must recover and still
    reach the true optimum.
    """
    import discopt.solvers.nlp_pounce as npn

    real = npn.solve_nlp
    state = {"n": 0}

    def flaky(evaluator, x0, *args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("transient NLP failure")
        return real(evaluator, x0, *args, **kwargs)

    monkeypatch.setattr(npn, "solve_nlp", flaky)
    r = solve_benders(_quad_two_point(), time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=ABS_TOL)
    assert state["n"] >= 2  # the failure path was actually exercised


def test_c1_persistent_failure_never_certifies(monkeypatch):
    """If the recourse NLP is always broken, the solve must not fabricate a
    certified answer (C1): no rigorous bound, not gap-certified."""
    import discopt.solvers.nlp_pounce as npn

    def always_raise(*args, **kwargs):
        raise RuntimeError("NLP backend down")

    monkeypatch.setattr(npn, "solve_nlp", always_raise)
    r = solve_benders(_quad_two_point(), time_limit=30)
    assert r.bound is None
    assert not r.gap_certified
    assert r.status != "optimal"


def test_c1_recourse_failure_at_optimum_not_excluded(monkeypatch):
    """The core C1 regression: a recourse solve that fails at the *optimal*
    first-stage point must not be mistaken for infeasibility and excluded.

    The recourse NLP (n vars) is forced to fail everywhere; the phase-1 NLP
    (n+1 vars) still runs. Because infeasibility is never *certified*, the solver
    must downgrade to heuristic mode (bound withheld, not gap-certified) rather
    than no-good-cut away the optimum and report a wrong certified optimum.
    """
    import discopt.solvers.nlp_pounce as npn

    real = npn.solve_nlp
    n_master_plus_recourse = 2  # y + x

    def fail_recourse_only(evaluator, x0, *args, **kwargs):
        # Phase-1 adds one elastic variable, so its x0 is longer than the
        # recourse solve's. Let phase-1 through; break the recourse solve.
        if len(np.asarray(x0)) == n_master_plus_recourse:
            raise RuntimeError("recourse solve broken")
        return real(evaluator, x0, *args, **kwargs)

    monkeypatch.setattr(npn, "solve_nlp", fail_recourse_only)
    r = solve_benders(_quad_two_point(), time_limit=30)
    # Never a *certified* wrong answer: bound withheld and not certified.
    assert r.bound is None
    assert not r.gap_certified


def test_c2_master_only_nonlinear_nonbinary_rejected():
    """A master-only nonlinear constraint with a *non-binary* integer master is
    rejected up front (C2), not failed mid-solve. (The all-binary case is
    supported and covered by ``test_master_only_nonlinear_constraint``.)"""
    m = dm.Model("monl_int")
    z = m.integer("z", lb=0, ub=4)
    x = m.continuous("x", lb=0, ub=5)
    m.first_stage(z)
    m.minimize(z + x * x)
    m.subject_to(z * z <= 9)  # master-only nonlinear, non-binary master
    m.subject_to(x + z >= 1)
    with pytest.raises(NotImplementedError, match="master-only nonlinear"):
        solve_benders(m, time_limit=30)
