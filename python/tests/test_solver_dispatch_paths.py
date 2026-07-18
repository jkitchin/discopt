"""Functional coverage of solve_model dispatch branches (#87).

Each test solves a 1-4 variable model end-to-end through a specific
``solve_model`` option path (non-smooth routing, warm starts, spatial-B&B
callbacks, heuristic toggles) and asserts the mathematically known optimum,
so the branch is verified — not merely executed. Every solve is seconds or
less.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.callbacks import CutResult
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke


def _bilinear_binary_model():
    """Nonconvex bilinear MINLP routed through the spatial batch B&B."""
    m = Model("spatial_mi")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y <= 4.0 + b)
    m.minimize(x + y + 2.0 * b)
    return m, x, y, b


def test_nonsmooth_abs_objective_routes_and_certifies():
    # Regression probe for #739 (fixed on main): the sampled linearity check
    # must not misroute abs() models to the LP fast path.
    # min |x| over a box straddling 0: a smooth NLP oscillates at the kink;
    # the non-smooth router must send this to the spatial B&B and certify 0.
    m = Model("absmin")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.minimize(abs(x))
    res = m.solve(time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(0.0, abs=1e-6)


def test_nonsmooth_abs_constraint_certifies():
    # Regression probe for #739 (fixed on main): the abs() constraint must
    # not be silently linearized/dropped.
    # min x s.t. |x| <= 0.5 -> optimum -0.5.
    m = Model("abscon")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.subject_to(abs(x) <= 0.5)
    m.minimize(x)
    res = m.solve(time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-0.5, abs=1e-5)


def test_warm_start_initial_solution_on_minlp():
    m, x, y, b = _bilinear_binary_model()
    # Feasible warm start: x=1, y=1, b=0 (x*y=1 >= 1, x+y=2 <= 4).
    res = m.solve(initial_solution={x: 1.0, y: 1.0, b: 0.0}, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    # Optimum: b=0 and x*y=1 with min x+y -> x=y=1, objective 2.
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_warm_start_used_as_incumbent():
    m, x, y, b = _bilinear_binary_model()
    res = m.solve(
        initial_solution={x: 1.0, y: 1.0, b: 0.0},
        use_start_as_incumbent=True,
        time_limit=60.0,
    )
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


@pytest.mark.xfail(
    reason="#740: heuristic incumbents bypass the lazy-constraints gate on the "
    "spatial MINLP path. Probe asserts the cut-respecting optimum.",
    strict=False,
)
def test_lazy_constraints_on_spatial_batch_loop():
    # Reject b=0 solutions via a lazy cut b >= 1: the certified optimum then
    # moves to the b=1 branch (x=y=1, objective 4).
    m, x, y, b = _bilinear_binary_model()
    calls = []

    def lazy(ctx, model):
        calls.append(ctx.node_bound)
        xs = ctx.x_relaxation
        if xs is not None and xs[2] < 0.5:
            return [CutResult(terms=[(b, 1.0)], sense=">=", rhs=1.0)]
        return None

    res = m.solve(lazy_constraints=lazy, time_limit=120.0)
    assert calls, "lazy constraint callback must run at integer-feasible nodes"
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(4.0, abs=1e-4)
    assert res.x["b"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.xfail(
    reason="#740: heuristic incumbents bypass the incumbent_callback veto on the "
    "spatial MINLP path. Probe asserts the veto-respecting optimum.",
    strict=False,
)
def test_incumbent_callback_veto_on_spatial_batch_loop():
    # Vetoing every b=0 incumbent forces the accepted optimum to b=1.
    m, x, y, b = _bilinear_binary_model()
    vetoed = []

    def incumbent_cb(ctx, model, sol):
        if sol["b"] < 0.5:
            vetoed.append(dict(sol))
            return False
        return True

    res = m.solve(incumbent_callback=incumbent_cb, time_limit=120.0)
    assert vetoed, "incumbent callback must be consulted"
    assert res.status in ("optimal", "feasible")
    assert res.x["b"] == pytest.approx(1.0, abs=1e-5)
    assert res.objective == pytest.approx(4.0, abs=1e-4)


def test_incumbent_callback_exception_fails_soft():
    # A buggy user callback is logged and ignored; the solve still certifies
    # the true optimum (soft-fail contract, INT-1/#413).
    m, x, y, b = _bilinear_binary_model()

    def boom(ctx, model, sol):
        raise RuntimeError("user bug")

    res = m.solve(incumbent_callback=boom, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_rens_heuristic_path():
    m, x, y, b = _bilinear_binary_model()
    res = m.solve(rens=True, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_subnlp_heuristic_path():
    m, x, y, b = _bilinear_binary_model()
    res = m.solve(subnlp_enabled=True, subnlp_frequency=1, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_maximize_bilinear_with_pounce_backend():
    # max x*y s.t. x + y == 1 -> 0.25 at x=y=0.5 (nonconvex, spatial B&B).
    m = Model("maxbil")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + y == 1.0)
    m.maximize(x * y)
    res = m.solve(nlp_solver="pounce", time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(0.25, abs=1e-4)


def test_pure_integer_nonconvex_box_search():
    # max x^2 over integer x in [-3, 3]: the integer box-search / rounding
    # machinery must find |x| = 3 (objective 9), not the stationary x = 0.
    m = Model("intbox")
    x = m.integer("x", lb=-3, ub=3)
    m.maximize(x**2)
    res = m.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(9.0, abs=1e-6)


def test_learned_relaxations_flag_is_accepted():
    # The flag loads the pretrained registry when available and must degrade
    # gracefully (same certified optimum) when no registry ships.
    m = Model("learned")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    res = m.solve(use_learned_relaxations=True, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_node_callback_receives_progress():
    m, x, y, b = _bilinear_binary_model()
    seen = []

    def node_cb(ctx, model):
        seen.append(ctx.node_count)

    res = m.solve(node_callback=node_cb, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert seen, "node callback must fire during the solve"
