"""Unit tests for the GBD feasibility-phase NLP (decomposition C1 fix).

The elastic phase-1 evaluator must (a) compute the epigraph residuals and their
derivatives correctly, and (b) let ``certify_recourse_feasibility`` distinguish a
feasible recourse from an infeasible one at a fixed first-stage point.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.decomposition.benders._feasibility import (
    FeasibilityPhaseEvaluator,
    certify_recourse_feasibility,
)
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

try:
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
except ImportError:
    POUNCE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not POUNCE_AVAILABLE, reason="no NLP backend")


def _convex_model():
    m = dm.Model("feas")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(x + y <= 4.0)
    m.subject_to(y * y - x <= 1.0)
    m.minimize(y)
    return m


def test_phase1_jacobian_and_hessian_match_finite_differences():
    m = _convex_model()
    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    lb, ub = (np.array(a, float).copy() for a in ev.variable_bounds)
    lb[0] = ub[0] = 1.0  # pin the master column
    fe = FeasibilityPhaseEvaluator(ev, cl, cu, lb, ub)

    z = np.array([1.0, 2.0, 0.5])
    h = 1e-6
    c0 = fe.evaluate_constraints(z)
    jac_fd = np.zeros((fe.n_constraints, fe.n_variables))
    for j in range(fe.n_variables):
        zp = z.copy()
        zp[j] += h
        jac_fd[:, j] = (fe.evaluate_constraints(zp) - c0) / h
    assert np.max(np.abs(fe.evaluate_jacobian(z) - jac_fd)) < 1e-4

    lam = np.array([0.7, 0.3])
    H = fe.evaluate_lagrangian_hessian(z, 0.0, lam)
    g0 = fe.evaluate_jacobian(z).T @ lam
    hess_fd = np.zeros_like(H)
    for j in range(fe.n_variables):
        zp = z.copy()
        zp[j] += h
        hess_fd[:, j] = (fe.evaluate_jacobian(zp).T @ lam - g0) / h
    assert np.max(np.abs(H - hess_fd)) < 1e-4


def test_certify_feasible_point():
    m = _convex_model()
    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    lb, ub = (np.array(a, float).copy() for a in ev.variable_bounds)
    lb[0] = ub[0] = 1.0  # x=1: y=0 satisfies both -> feasible
    verdict, info = certify_recourse_feasibility(ev, cl, cu, lb, ub)
    assert verdict == "feasible"
    assert info["t"] <= 1e-6


def test_certify_infeasible_point():
    m = dm.Model("infeas")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(x + y <= 4.0)
    m.subject_to(y >= 3.0)
    m.minimize(y)
    ev = NLPEvaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    lb, ub = (np.array(a, float).copy() for a in ev.variable_bounds)
    lb[0] = ub[0] = 3.0  # x=3 -> y<=1 and y>=3 : infeasible
    verdict, info = certify_recourse_feasibility(ev, cl, cu, lb, ub)
    assert verdict == "infeasible"
    assert info["t"] > 1e-6
