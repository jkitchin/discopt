"""#1614: every route names itself, and ``p*(x+y)`` is a MILP HiGHS takes.

C1 -- a parameter times a sum was flagged ``general_nl`` by the Python term
classifier (only a *literal* constant counted as scaling), so the model failed the
"exactly linear" gate and left HiGHS for the in-house MILP tree, while the
algebraically equal ``p*x+p*y`` went to HiGHS.

D-14 / B-02b -- ``SolveResult.algorithm_route`` was ``None`` on the convex-QP,
nlp-bb, spatial, Benders, Lagrangian, ``lagrangian_bound`` and ``lazy_constraints``
routes, so the engine that answered was invisible.

Every assertion here is on a value the solve returned; each test also checks the
answer, so a route name cannot pass on a wrong result.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest
from discopt._relax.term_classifier import classify_nonlinear_terms

pytest.importorskip("highspy")


def _param_times_sum(form: str, pval: float):
    m = dm.Model("r")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    p = m.parameter("p", pval)
    m.subject_to(p * (x + y) <= 3.2 if form == "factored" else p * x + p * y <= 3.2)
    m.maximize(x + 2 * y)
    return m


@pytest.mark.parametrize("pval,expected", [(1.0, 6.0), (0.5, 12.0)])
@pytest.mark.parametrize("form", ["factored", "distributed"])
def test_c1_parameter_times_sum_goes_to_highs(form, pval, expected):
    m = _param_times_sum(form, pval)
    assert classify_nonlinear_terms(m).general_nl == [], "p*(x+y) is linear in x, y"
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(expected, abs=1e-6)
    assert r.algorithm_route is not None and r.algorithm_route.startswith("highs-milp"), (
        r.algorithm_route
    )


def test_c1_parameter_product_of_variables_is_still_nonlinear():
    """Only a variable-free factor is scaling: ``p*x*y`` stays nonlinear."""
    m = dm.Model("pxy")
    x = m.continuous("x", lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    p = m.parameter("p", 1.0)
    m.subject_to(p * (x * y) <= 1)
    m.maximize(x + y)
    terms = classify_nonlinear_terms(m)
    assert terms.bilinear or terms.general_nl, "x*y under a parameter is nonlinear"
    from discopt.solver import _milp_is_exactly_linear

    assert not _milp_is_exactly_linear(m)


def _route(r) -> str:
    assert r.algorithm_route is not None, "algorithm_route is None: the route is unnamed"
    return r.algorithm_route


def test_b02b_convex_qp_default_solve_names_route():
    m = dm.Model("q")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    m.minimize(dm.sum([(x[i] - i) ** 2 for i in range(3)]))
    m.subject_to(x[0] + x[1] + x[2] >= 4)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    # unconstrained optimum 0+1+2 = 3 < 4: project onto the plane, +1/3 each
    assert r.objective == pytest.approx(3 * (1 / 3) ** 2, abs=1e-5)
    assert _route(r)


def test_d14_nlp_bb_names_route():
    m = dm.Model("nlpbb")
    x = m.continuous("x", lb=0, ub=4)
    z = m.binary("z")
    m.minimize((x - 1.5) ** 2 + z)
    m.subject_to(x <= 1 + 3 * z)
    r = m.solve(nlp_bb=True, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.25, abs=1e-4)
    assert "nlp-bb" in _route(r)


def test_d14_spatial_nonconvex_names_route():
    m = dm.Model("bistable")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(x**4 - 2 * x**2 + 0.25 * x)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective is not None and r.objective < -1.0
    assert _route(r)


def _two_stage_milp():
    m = dm.Model("two_stage")
    y = m.binary("y")
    x1 = m.continuous("x1", lb=0, ub=10)
    x2 = m.continuous("x2", lb=0, ub=10)
    m.minimize(2 * y + x1 + x2)
    m.subject_to(x1 + x2 >= 3)
    m.subject_to(x1 <= 5 * y)
    m.subject_to(x2 <= 5 * y)
    return m


def test_d14_benders_names_route_and_master_engine():
    r = _two_stage_milp().solve(decomposition="benders", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-5)
    route = _route(r)
    assert route.startswith("benders") and "master MILP on" in route, route


def test_d14_direct_solve_benders_names_route():
    """``stochastic.solve_lshaped`` calls ``solve_benders`` directly, not via solve()."""
    from discopt.decomposition.benders import solve_benders

    r = solve_benders(_two_stage_milp(), time_limit=60)
    assert r.status == "optimal"
    assert "master MILP on" in _route(r)


def _coupled_knapsack():
    m = dm.Model("lag")
    x = m.binary("x", shape=(4,))
    m.minimize(-(3 * x[0] + 5 * x[1] + 4 * x[2] + 2 * x[3]))
    c = 3 * x[0] + 4 * x[1] + 2 * x[2] + 3 * x[3] <= 6
    m.subject_to(c)
    m.mark_coupling(c)
    return m


def test_d14_lagrangian_decomposition_names_route():
    r = _coupled_knapsack().solve(decomposition="lagrangian", time_limit=60)
    assert r.bound is not None and r.bound <= -9.0 + 1e-6  # optimum is -9 (x1, x2)
    assert _route(r).startswith("lagrangian"), r.algorithm_route


def test_d14_lagrangian_bound_names_route():
    r = _coupled_knapsack().solve(lagrangian_bound=True, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-9.0, abs=1e-6)
    assert "lagrangian_bound" in _route(r), r.algorithm_route


def test_d14_lazy_constraints_names_route():
    m = dm.Model("lazy")
    x = [m.binary(f"x{i}") for i in range(2)]
    m.maximize(x[0] + x[1])
    calls = []
    r = m.solve(lazy_constraints=lambda ctx, model: calls.append(1) or [], time_limit=30)
    assert calls, "the lazy callback was never invoked"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0)
    assert _route(r)


def _convex_minlp():
    m = dm.Model("cvx")
    x = m.continuous("x", shape=(3,), lb=0, ub=4)
    z = m.binary("z", shape=(3,))
    m.minimize(dm.sum([(x[i] - 1.5 - i) ** 2 for i in range(3)]) + dm.sum([z[i] for i in range(3)]))
    for i in range(3):
        m.subject_to(x[i] <= 1 + 3 * z[i])
    return m


def test_e01c_auto_route_engine_failure_warns(monkeypatch):
    """A routed engine that RAISES falls back soundly -- and says so out loud."""
    import discopt.solvers.mip_nlp as mip_nlp

    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("simulated master failure")

    monkeypatch.setattr(mip_nlp, "solve_mip_nlp", broken)
    with pytest.warns(RuntimeWarning, match="simulated master failure"):
        r = _convex_minlp().solve(time_limit=60)
    assert calls, "the auto-route never ran; the test measured nothing"
    assert r.status == "optimal"
    # z0=0 (x0=1, cost .25), z1=1 (x1=2.5, cost 1), z2=1 (x2=3.5, cost 1)
    assert r.objective == pytest.approx(2.25, abs=1e-4)
    assert "fell back" in _route(r)


# ── A-21: a quadratic form whose Q is exactly zero is linear ──


def _zero_q_model(form: str, where: str, integer: bool):
    m = dm.Model("zq")
    make = m.integer if integer else m.continuous
    x = make("x", lb=0, ub=10)
    y = make("y", lb=0, ub=10)
    q = {
        "zero_sum_sq": 0.0 * (x + y) ** 2,
        "zero_bilinear": 0 * x * y,
        "cancelled_sq": (x - x) ** 2,
        "tiny_sq": 1e-300 * x**2,
    }[form]
    if where == "objective":
        m.maximize(3 * x + 2 * y - q)
        m.subject_to(x + y <= 4)
    else:
        m.maximize(3 * x + 2 * y)
        m.subject_to(x + y + q <= 4)
    return m


@pytest.mark.parametrize("where", ["objective", "constraint"])
@pytest.mark.parametrize("form", ["zero_sum_sq", "zero_bilinear", "cancelled_sq"])
@pytest.mark.parametrize("integer", [False, True])
def test_a21_zero_quadratic_coefficient_is_linear(form, where, integer):
    from discopt._relax.problem_classifier import ProblemClass, classify_problem

    m = _zero_q_model(form, where, integer)
    assert classify_problem(m) == (ProblemClass.MILP if integer else ProblemClass.LP)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(12.0, abs=1e-6)  # x=4, y=0
    # A MILP goes to HiGHS; an LP to the default LP route (Rust since highs-parity D1).
    assert _route(r).startswith("highs-milp" if integer else "rust-lp"), r.algorithm_route


@pytest.mark.parametrize("where", ["objective", "constraint"])
def test_a21_tiny_nonzero_coefficient_stays_quadratic(where):
    """Only an EXACT zero is dropped: 1e-300 is a (tiny) quadratic term."""
    from discopt._relax.problem_classifier import ProblemClass, classify_problem

    m = _zero_q_model("tiny_sq", where, integer=False)
    assert classify_problem(m) not in (ProblemClass.LP, ProblemClass.MILP)


# ── D-14 leftovers: convex-NLP and GP routes name themselves ──


def test_d14_convex_nlp_default_solve_names_route():
    m = dm.Model("cnlp")
    x = m.continuous("x", lb=0.1, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(dm.exp(x) + (y - 2) ** 2)
    m.subject_to(x + y >= 3)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    # x+y>=3 binds and lb=0.1 is slack: stationarity gives e^x = 2(y-2) = 2(1-x).
    import math

    xs = r.x["x"]
    assert math.exp(xs) == pytest.approx(2 * (1 - xs), abs=1e-4)
    assert r.objective == pytest.approx(math.exp(xs) + (1 - xs) ** 2, abs=1e-5)
    assert _route(r)


# ── D-26/D-27: decomposition masters on HiGHS, and an LP master's bound ──


def test_d26_benders_master_runs_on_highs():
    r = _two_stage_milp().solve(decomposition="benders", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-5)
    assert _route(r).endswith("master MILP on HiGHS"), r.algorithm_route


def test_d26_lagrangian_master_runs_on_highs(monkeypatch):
    import discopt.solvers.lp_backend as lpb

    engines = []
    original = lpb.get_decomposition_master_solver

    def spy():
        milp, engine = original()
        engines.append(engine)
        return milp, engine

    monkeypatch.setattr(lpb, "get_decomposition_master_solver", spy)
    r = _coupled_knapsack().solve(decomposition="lagrangian", time_limit=60)
    assert engines and set(engines) == {"HiGHS"}, engines
    assert r.bound is not None and r.bound <= -9.0 + 1e-6


def test_d26_highs_lp_master_bound_is_its_optimum():
    """With no integer column, HiGHS's ``mip_dual_bound`` is a stale 0.0.

    Read as the bound it capped this LP's bound at 0 < 4 (the optimum), so a
    continuous Benders master never closed its gap and stalled at
    ``iteration_limit``.
    """
    import numpy as np
    from discopt.solvers import SolveStatus
    from discopt.solvers.milp_highs import solve_milp

    r = solve_milp(
        c=np.array([1.0, 1.0]),
        A_ub=np.array([[-2.0, -1.0], [0.0, -1.0]]),
        b_ub=np.array([-8.0, 0.0]),
        bounds=[(0.0, 10.0), (-1e12, 1e20)],
    )
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(4.0, abs=1e-9)
    assert r.bound == pytest.approx(4.0, abs=1e-9)


# ── E-01(d): every MIP-NLP method takes milp_solver, and honours it ──


@pytest.mark.parametrize("method", ["fp", "goa"])
def test_e01d_mip_nlp_method_honours_milp_solver(method, monkeypatch):
    import warnings

    import discopt.solvers.lp_backend as lpb

    seen = []
    original = lpb.get_milp_solver

    def spy(prefer_pounce=False, backend="auto"):
        seen.append(backend)
        return original(prefer_pounce=prefer_pounce, backend=backend)

    monkeypatch.setattr(lpb, "get_milp_solver", spy)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*ignored.*milp_solver.*")
        r = _convex_minlp().solve(
            solver="mip-nlp", mip_nlp_method=method, milp_solver="highs", time_limit=60
        )
    assert seen, "no MILP was solved; the test measured nothing"
    assert set(seen) == {"highs"}, seen
    assert r.status in ("optimal", "feasible")
    assert r.objective == pytest.approx(2.25, abs=1e-4)
