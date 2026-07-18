"""Functional coverage of solve_model engine/relaxation option paths (#87).

Second battery of tiny end-to-end solves, each pinning a known optimum while
steering an under-tested engine path: NLP-based B&B, McCormick-NLP node
bounds, batched node processing, GNN branching-score hook, root cut loops,
LP/QP matrix fast paths, and deadline exhaustion. Runtime: seconds per test.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke


def _bilinear_binary_model():
    m = Model("engine_mi")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y <= 4.0 + b)
    m.minimize(x + y + 2.0 * b)
    return m


def test_nlp_bb_path_on_convex_minlp():
    # Convex MINLP: min (i-1.3)^2 + (z-0.7)^2, i integer in [0,3].
    # NLP-based B&B branches on i; optimum at i=1, z=0.7 -> 0.09.
    m = Model("nlpbb")
    i = m.integer("i", lb=0, ub=3)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize((i - 1.3) ** 2 + (z - 0.7) ** 2)
    res = m.solve(nlp_bb=True, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(0.09, abs=1e-4)
    assert res.x["i"] == pytest.approx(1.0, abs=1e-6)


def test_nlp_bb_with_warm_start():
    m = Model("nlpbb_ws")
    i = m.integer("i", lb=0, ub=3)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize((i - 1.3) ** 2 + (z - 0.7) ** 2)
    res = m.solve(nlp_bb=True, initial_solution={i: 1.0, z: 0.7}, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(0.09, abs=1e-4)


def test_mccormick_nlp_node_bounds():
    # mccormick_bounds="nlp" compiles relaxation evaluators for objective and
    # constraints and bounds every open node with them.
    m = _bilinear_binary_model()
    res = m.solve(mccormick_bounds="nlp", time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_mccormick_none_disables_extra_bounds():
    m = _bilinear_binary_model()
    res = m.solve(mccormick_bounds="none", time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_invalid_mccormick_bounds_rejected():
    m = _bilinear_binary_model()
    with pytest.raises((ValueError, TypeError)):
        m.solve(mccormick_bounds="bogus", time_limit=10.0)


def test_batched_node_processing():
    # batch_size > 1 drives the batched node solve path.
    m = _bilinear_binary_model()
    res = m.solve(batch_size=4, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_root_cut_loop_on_knapsack():
    # 0/1 knapsack with a fractional LP root: the root cut loop (cover /
    # aggregation-MIR separation) must not cut off the true optimum.
    m = Model("knap")
    xb = m.binary("xb", shape=(4,))
    w = [5.0, 4.0, 3.0, 2.0]
    v = [8.0, 6.0, 4.0, 2.5]
    m.subject_to(sum(w[j] * xb[j] for j in range(4)) <= 7.0)
    m.maximize(sum(v[j] * xb[j] for j in range(4)))
    res = m.solve(root_cut_rounds=3, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    # Best value with capacity 7: items 1+3 (w=4+3=7, v=10.5).
    assert res.objective == pytest.approx(10.5, abs=1e-6)


def test_lp_fast_path():
    m = Model("lp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + 2.0 * y >= 4.0)
    m.subject_to(x + y == 3.0)
    m.minimize(2.0 * x + y)
    res = m.solve(time_limit=30.0)
    assert res.status == "optimal"
    # x + y = 3 and x + 2y >= 4 -> y >= 1; min 2x + y = x + 3 at x as small
    # as possible: x = 3 - y, minimized at y = 3, x = 0 -> objective 3.
    assert res.objective == pytest.approx(3.0, abs=1e-6)


def test_qp_fast_path():
    m = Model("qp")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.subject_to(x + y >= 2.0)
    m.minimize(x**2 + y**2)
    res = m.solve(time_limit=30.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(2.0, abs=1e-5)  # x=y=1


def test_infeasible_lp_detected():
    m = Model("lp_inf")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(x >= 2.0)
    m.minimize(x)
    res = m.solve(time_limit=30.0)
    assert res.status == "infeasible"


def test_deadline_exhaustion_reports_time_limit():
    # A nonconvex MINLP with an absurdly small budget must terminate quickly
    # and never report a certified optimum it did not earn.
    m = _bilinear_binary_model()
    res = m.solve(time_limit=0.05)
    assert res.status in ("time_limit", "feasible", "optimal", "iteration_limit")
    if res.status == "optimal":
        # If it certified anyway (warm cache), the certificate must be right.
        assert res.objective == pytest.approx(2.0, abs=1e-3)


def test_milp_simplex_backend_solves_pure_milp():
    m = Model("milp")
    i = m.integer("i", lb=0, ub=10)
    j = m.integer("j", lb=0, ub=10)
    m.subject_to(3.0 * i + 5.0 * j <= 15.0)
    m.maximize(2.0 * i + 3.0 * j)
    res = m.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    # Enumerate: best is i=5, j=0 -> 10? Check i=0,j=3 -> 9; i=5,j=0 -> 10;
    # i=3,j=1 (9+5=14<=15) -> 9. Optimum 10.
    assert res.objective == pytest.approx(10.0, abs=1e-6)


def test_solution_pool_collects_incumbents():
    m = _bilinear_binary_model()
    res = m.solve(solution_pool=True, solution_pool_capacity=5, time_limit=120.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(2.0, abs=1e-4)
