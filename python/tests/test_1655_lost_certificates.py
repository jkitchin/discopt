"""#1655: certificates declined where the backend proved optimality.

1. Convex QP: the certificate evaluated the objective at the IPM point, which sits
   on its equality rows only to IPM tolerance; through rows scaled by 1/d that put
   the "feasible value" 3.3e-8 below the optimum, and the rigorous bound above it
   read as a failed premise. The point is projected onto its equality rows first.
2. LP: the exact dual correction converged in 9-16 rounds on a robust counterpart
   but was capped at 8, leaving the NS bound 19% below the optimum.
3. Pure-binary MILP: the #1634 presolve-free cross-solve ran for the whole budget on
   a model with integral 0/1 data, the class its hazard cannot reach.
"""

from __future__ import annotations

import math
import time
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _highs_std_form, _project_onto_equality_rows
from discopt.solvers import lp_milp_highs as L


def _quiet_solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(**kw)


def _glass() -> dm.Model:
    A = np.array(  # noqa: N806
        [
            [2.20, 2.95, 3.55, 3.10, 2.75, 1.90],
            [5.0, 400.0, 130.0, 45.0, -30.0, 0.0],
            [1.460, 1.590, 1.730, 1.610, 1.520, 1.460],
        ]
    )
    t, d = np.array([2.50, 85.0, 1.520]), np.array([0.01, 1.0, 0.002])
    lo, hi = [0.65, 0.10, 0.05, 0, 0, 0], [0.78, 0.17, 0.12, 0.06, 0.03, 0.05]
    m = dm.Model("glass")
    x = [m.continuous(f"x{i}", lb=lo[i], ub=hi[i]) for i in range(6)]
    m.subject_to(sum(x) == 1)
    e = [m.continuous(f"e{k}", lb=-np.inf, ub=np.inf) for k in range(3)]
    for k in range(3):
        m.subject_to(e[k] == (sum(A[k, i] * x[i] for i in range(6)) - t[k]) / d[k])
    m.minimize(sum(ek**2 for ek in e))
    return m


@pytest.mark.parametrize("solver", ["pounce", None])
def test_convex_qp_with_free_residuals_is_certified(solver):
    kw = {"solver": solver} if solver else {}
    r = _quiet_solve(_glass(), **kw)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.bound is not None and r.bound <= r.objective + 1e-12
    assert r.objective == pytest.approx(0.66743860, rel=1e-6)


def test_projection_repairs_equality_rows_and_never_worsens():
    A = np.array([[1.0, 1.0, 0.0], [500.0, 0.0, -1.0]])  # noqa: N806
    cl = cu = np.array([1.0, 0.0])
    lb, ub = np.array([0.0, 0.0, -np.inf]), np.array([1.0, 1.0, np.inf])
    x = np.array([0.3, 0.7 + 5e-11, 150.0])
    z = _project_onto_equality_rows(x, A, cl, cu, lb, ub)
    assert z is not None
    assert np.max(np.abs(A @ z - cl)) < np.max(np.abs(A @ x - cl))
    assert np.all(z >= lb) and np.all(z <= ub)
    # an exactly feasible point needs no repair
    assert _project_onto_equality_rows(np.array([0.3, 0.7, 150.0]), A, cl, cu, lb, ub) is None


def _robust_blend(gamma: float) -> dm.Model:
    from discopt.ro import RobustCounterpart, budget_uncertainty_set

    cost = np.array([607.0, 646, 578, 679, 717, 734, 625, 582, 640, 678, 610, 689])
    s_bar = np.array([17.9, 12.8, 19.4, 10.6, 4.8, 3.7, 15.4, 18.2, 13.0, 11.1, 15.8, 9.0])
    avail = np.array([178.0, 188, 396, 194, 357, 376, 308, 319, 225, 191, 389, 215])
    cet = np.array([45.8, 40.2, 54.2, 54.7, 42.2, 43.8, 41.3, 45.0, 46.6, 43.3, 43.9, 42.7])
    n = 12
    m = dm.Model("rblend")
    v = m.continuous("v", shape=(n,), lb=0, ub=avail)
    s = m.parameter("s", value=s_bar)
    m.minimize(dm.sum(lambda j: float(cost[j]) * v[j], over=range(n)))
    m.subject_to(dm.sum(lambda j: v[j], over=range(n)) >= 1000.0)
    m.subject_to(dm.sum(lambda j: float(cet[j] - 45.0) * v[j], over=range(n)) >= 0)
    m.subject_to(s @ v - 10.0 * dm.sum(lambda j: v[j], over=range(n)) <= 0)
    RobustCounterpart(m, budget_uncertainty_set(s, delta=0.3 * s_bar, gamma=gamma)).formulate()
    return m


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_robust_lp_is_certified(gamma):
    r = _quiet_solve(_robust_blend(gamma))
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound, r.algorithm_route)
    assert r.bound == pytest.approx(r.objective, rel=1e-9)


def _stn() -> dm.Model:
    react = {"A": {"R1": 10, "R2": 12}, "B": {"R1": 14, "R2": 16}, "C": {"R2": 12}}
    finish, margin = {"A": 6, "B": 4, "C": 8}, {"A": 14, "B": 12, "C": 11}
    orders, horizon, dt = {"A": 10, "B": 8, "C": 6}, 120, 2
    n_t = horizon // dt

    def per(p):
        return math.ceil(p / dt)

    tasks = [("react", g, j, per(p)) for g in react for j, p in react[g].items()]
    tasks += [("finish", g, "F", per(finish[g])) for g in react]
    m = dm.Model("stn")
    w = m.binary("W", shape=(len(tasks), n_t))
    for j in sorted({tk[2] for tk in tasks}):
        ks = [k for k, tk in enumerate(tasks) if tk[2] == j]
        m.subject_to(
            dm.sum([w[k, t] for k in ks for t in range(max(0, n_t - tasks[k][3] + 1), n_t)]) == 0
        )
        for t in range(n_t):
            m.subject_to(
                dm.sum([w[k, s] for k in ks for s in range(max(0, t - tasks[k][3] + 1), t + 1)])
                <= 1
            )
    n_fin = {}
    for g in react:
        rk = [k for k, tk in enumerate(tasks) if tk[:2] == ("react", g)]
        fk = [k for k, tk in enumerate(tasks) if tk[:2] == ("finish", g)]
        for t in range(n_t):
            made = [w[k, s] for k in rk for s in range(0, t - tasks[k][3] + 1)]
            used = [w[k, s] for k in fk for s in range(0, t + 1)]
            m.subject_to((dm.sum(made) if made else 0) - dm.sum(used) >= 0)
        n_fin[g] = dm.sum([w[k, t] for k in fk for t in range(n_t)])
        m.subject_to(n_fin[g] <= orders[g])
        m.subject_to(dm.sum([w[k, t] for k in rk for t in range(n_t)]) <= orders[g])
    m.maximize(dm.sum([float(margin[g]) * n_fin[g] for g in react]))
    return m


def test_pure_binary_scheduling_milp_certifies_fast():
    m = _stn()
    t0 = time.perf_counter()
    r = _quiet_solve(m, time_limit=60)
    wall = time.perf_counter() - t0
    assert r.status == "optimal" and r.gap_certified, r.algorithm_route
    assert r.objective == pytest.approx(236.0, abs=1e-6)
    assert r.bound >= 236.0 - 1e-6
    assert r.solver_stats.get("milp/presolve_cross_check_exact_class") == 1.0
    assert wall < 45.0, wall


def test_exact_class_excludes_badly_scaled_models():
    """The #1634 witnesses (non-integral, nine decades) stay cross-checked."""
    from test_1634_highs_parallel_column_presolve import WITNESSES, _model

    n = 0
    for c, A, b, ub, _ in WITNESSES.values():  # noqa: N806
        _, _, sf = _highs_std_form(_model(c, A, b, ub))
        assert not L.presolve_exact_class(sf)
        n += 1
    assert n == len(WITNESSES)
    _, _, sf = _highs_std_form(_stn())
    assert L.presolve_exact_class(sf)
