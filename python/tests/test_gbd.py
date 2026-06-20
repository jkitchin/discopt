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
