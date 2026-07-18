"""Classifier extraction tests on .nl-loaded models (repr/autodiff paths, #87).

Models re-loaded from AMPL .nl files carry no algebraic Expression trees, so
LP/QP/QCP extraction routes through the Rust-repr / autodiff extractors
instead of the algebraic walkers. Same validation as the algebraic tests:
the extracted matrices must reproduce objective and quadratic-row values at
sampled points.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.problem_classifier import (
    ProblemClass,
    classify_problem,
    extract_lp_data,
    extract_qcp_data,
    extract_qp_data,
)
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


def _roundtrip(model):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nl", delete=False) as f:
        path = f.name
    try:
        model.to_nl(path)
        return dm.from_nl(path)
    finally:
        os.unlink(path)


def test_lp_extraction_from_nl_loaded_model():
    m = Model("lp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + 2.0 * y <= 4.0)
    m.subject_to(x - y >= -1.0)
    m.minimize(3.0 * x - y + 2.0)
    m2 = _roundtrip(m)
    assert classify_problem(m2) == ProblemClass.LP
    data = extract_lp_data(m2)
    c = np.asarray(data.c, dtype=np.float64)
    ev = NLPEvaluator(m2)
    obj0 = float(ev.evaluate_objective(np.zeros(2)))
    rng = np.random.RandomState(0)
    for _ in range(4):
        pt = rng.uniform(-1.0, 1.0, size=2)
        assert float(c[:2] @ pt) == pytest.approx(float(ev.evaluate_objective(pt)) - obj0, abs=1e-8)


def test_qp_extraction_from_nl_loaded_model():
    m = Model("qp")
    a = m.continuous("a", lb=-2.0, ub=2.0)
    b = m.continuous("b", lb=-2.0, ub=2.0)
    m.subject_to(a + b <= 2.0)
    m.minimize(2.0 * a**2 - a * b + 0.25 * b**2 + 1.5 * a)
    m2 = _roundtrip(m)
    data = extract_qp_data(m2)
    Q = np.asarray(data.Q, dtype=np.float64)[:2, :2]
    c = np.asarray(data.c, dtype=np.float64)[:2]
    ev = NLPEvaluator(m2)
    obj0 = float(ev.evaluate_objective(np.zeros(2)))
    rng = np.random.RandomState(1)
    for _ in range(5):
        pt = rng.uniform(-1.5, 1.5, size=2)
        val = float(0.5 * pt @ Q @ pt + c @ pt)
        assert val == pytest.approx(float(ev.evaluate_objective(pt)) - obj0, abs=1e-7)


def test_qcp_extraction_from_nl_loaded_model():
    m = Model("qcp")
    a = m.continuous("a", lb=-2.0, ub=2.0)
    b = m.continuous("b", lb=-2.0, ub=2.0)
    m.subject_to(a**2 + b**2 <= 1.0)
    m.subject_to(a + b >= 0.0)
    m.minimize(a + 2.0 * b + a * a)
    m2 = _roundtrip(m)
    assert classify_problem(m2) == ProblemClass.QCQP
    data = extract_qcp_data(m2)
    # Objective: 0.5 x'Qx + c'x with Q diag (2, 0), c = (1, 2).
    np.testing.assert_allclose(np.asarray(data.Q).diagonal(), [2.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(np.asarray(data.c)[:2], [1.0, 2.0], atol=1e-10)
    assert len(data.quadratic_constraints) == 1
    qc = data.quadratic_constraints[0]
    Qr = np.asarray(qc.Q, dtype=np.float64)[:2, :2]
    ar = np.asarray(qc.c, dtype=np.float64)[:2]
    rng = np.random.RandomState(2)
    for _ in range(5):
        pt = rng.uniform(-1.0, 1.0, size=2)
        row = float(0.5 * pt @ Qr @ pt + ar @ pt)
        assert row == pytest.approx(pt[0] ** 2 + pt[1] ** 2, abs=1e-7)


def test_milp_classification_from_nl_loaded_model():
    m = Model("milp")
    i = m.integer("i", lb=0, ub=5)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(i + y <= 3.0)
    m.minimize(i + y)
    m2 = _roundtrip(m)
    assert classify_problem(m2) == ProblemClass.MILP
    # And the loaded MILP still solves to the same optimum (0 at the origin).
    res = m2.solve(time_limit=30.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(0.0, abs=1e-8)
