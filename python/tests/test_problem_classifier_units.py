"""Unit tests for problem classification and LP/QP/QCP data extraction (#87).

Every extraction is validated against the model itself: the extracted
matrices must reproduce the objective (and constraint rows) at sampled
points. This pins the algebraic walkers' coefficient handling across
expression shapes (scaled squares, cross terms, divisions, negations,
array variables, sums) rather than just executing them.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.problem_classifier import (
    ProblemClass,
    classify_problem,
    extract_lp_data,
    extract_lp_data_algebraic,
    extract_qcp_data,
    extract_qp_data,
    extract_qp_data_algebraic,
)
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_problem_classes():
    m_lp = Model("lp")
    x = m_lp.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m_lp.subject_to(x[0] + x[1] <= 1.0)
    m_lp.minimize(x[0] - 2.0 * x[1])
    assert classify_problem(m_lp) == ProblemClass.LP

    m_qp = Model("qp")
    a = m_qp.continuous("a", lb=-1.0, ub=1.0)
    b = m_qp.continuous("b", lb=-1.0, ub=1.0)
    m_qp.subject_to(a + b <= 1.0)
    m_qp.minimize(a**2 + 0.5 * b**2 + a * b + a)
    assert classify_problem(m_qp) == ProblemClass.QP

    m_milp = Model("milp")
    i = m_milp.integer("i", lb=0, ub=5)
    y = m_milp.continuous("y", lb=0.0, ub=1.0)
    m_milp.subject_to(i + y <= 3.0)
    m_milp.minimize(i + y)
    assert classify_problem(m_milp) == ProblemClass.MILP

    m_nlp = Model("nlp")
    z = m_nlp.continuous("z", lb=0.1, ub=2.0)
    m_nlp.minimize(dm.exp(z) + z)
    assert classify_problem(m_nlp) == ProblemClass.NLP

    m_minlp = Model("minlp")
    w = m_minlp.continuous("w", lb=0.1, ub=2.0)
    k = m_minlp.integer("k", lb=0, ub=3)
    m_minlp.subject_to(w * k >= 0.5)
    m_minlp.minimize(w + k)
    # w*k is a quadratic (bilinear) row with an integer variable -> MIQCP.
    assert classify_problem(m_minlp) == ProblemClass.MIQCP


# ---------------------------------------------------------------------------
# LP extraction
# ---------------------------------------------------------------------------


def _lp_model():
    m = Model("lpx")
    x = m.continuous("x", lb=0.0, ub=10.0, shape=(2,))
    z = m.continuous("z", lb=-5.0, ub=5.0)
    m.subject_to(x[0] + 2.0 * x[1] - z <= 4.0)
    m.subject_to(x[0] - x[1] == 1.0)
    m.subject_to(-(z / 2.0) + x[0] >= -3.0)
    m.minimize(3.0 * x[0] - x[1] + 0.5 * z + 2.0)
    return m


@pytest.mark.parametrize("extract", [extract_lp_data, extract_lp_data_algebraic])
def test_lp_extraction_reproduces_objective(extract):
    m = _lp_model()
    data = extract(m)
    c = np.asarray(data.c, dtype=np.float64)
    rng = np.random.RandomState(0)
    from discopt._jax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(m)
    for _ in range(5):
        pt = rng.uniform(-1.0, 1.0, size=3)
        # Extraction may append slack columns; the structural prefix must
        # reproduce the objective up to the constant offset.
        lin = float(c[:3] @ pt)
        obj_at_pt = float(ev.evaluate_objective(pt))
        obj_at_zero = float(ev.evaluate_objective(np.zeros(3)))
        assert lin == pytest.approx(obj_at_pt - obj_at_zero, abs=1e-8)


# ---------------------------------------------------------------------------
# QP extraction
# ---------------------------------------------------------------------------


def _qp_model():
    m = Model("qpx")
    a = m.continuous("a", lb=-2.0, ub=2.0)
    b = m.continuous("b", lb=-2.0, ub=2.0)
    m.subject_to(a + b <= 2.0)
    # Mixed shapes: scaled square, cross term, division, negation, linear.
    m.minimize(2.0 * a**2 - a * b + (b**2) / 4.0 - (-b) + 1.5 * a)
    return m


@pytest.mark.parametrize("extract", [extract_qp_data, extract_qp_data_algebraic])
def test_qp_extraction_reproduces_objective(extract):
    m = _qp_model()
    data = extract(m)
    Q = np.asarray(data.Q, dtype=np.float64)
    c = np.asarray(data.c, dtype=np.float64)
    from discopt._jax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(m)
    rng = np.random.RandomState(1)
    obj0 = float(ev.evaluate_objective(np.zeros(2)))
    for _ in range(6):
        pt = rng.uniform(-1.5, 1.5, size=2)
        quad = float(0.5 * pt @ Q[:2, :2] @ pt + c[:2] @ pt)
        assert quad == pytest.approx(float(ev.evaluate_objective(pt)) - obj0, abs=1e-8)


def test_qp_extraction_array_variables_and_matmul():
    m = Model("qparr")
    x = m.continuous("x", lb=-1.0, ub=1.0, shape=(3,))
    m.subject_to(dm.sum(x) <= 2.0)
    m.minimize(dm.sum(x * x) + x[0] * x[1] - 2.0 * x[2])
    data = extract_qp_data(m)
    Q = np.asarray(data.Q, dtype=np.float64)[:3, :3]
    c = np.asarray(data.c, dtype=np.float64)[:3]
    from discopt._jax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(m)
    rng = np.random.RandomState(2)
    obj0 = float(ev.evaluate_objective(np.zeros(3)))
    for _ in range(6):
        pt = rng.uniform(-1.0, 1.0, size=3)
        val = float(0.5 * pt @ Q @ pt + c @ pt)
        assert val == pytest.approx(float(ev.evaluate_objective(pt)) - obj0, abs=1e-8)


# ---------------------------------------------------------------------------
# QCP extraction
# ---------------------------------------------------------------------------


def test_qcp_extraction_reproduces_quadratic_rows():
    m = Model("qcp")
    a = m.continuous("a", lb=-2.0, ub=2.0)
    b = m.continuous("b", lb=-2.0, ub=2.0)
    m.subject_to(a**2 + b**2 <= 1.0)  # quadratic row
    m.subject_to(a + b >= 0.0)  # linear row
    m.minimize(a + 2.0 * b)
    assert classify_problem(m) == ProblemClass.QCP
    data = extract_qcp_data(m)
    assert len(data.quadratic_constraints) == 1
    qc = data.quadratic_constraints[0]
    Qr = np.asarray(qc.Q, dtype=np.float64)[:2, :2]
    ar = np.asarray(qc.c, dtype=np.float64)[:2]
    rng = np.random.RandomState(3)
    for _ in range(5):
        pt = rng.uniform(-1.0, 1.0, size=2)
        # Row value 0.5 x'Qx + a'x must equal a^2+b^2 up to the stored rhs
        # convention; check the quadratic part reproduces the curvature.
        row = float(0.5 * pt @ Qr @ pt + ar @ pt)
        assert row == pytest.approx(pt[0] ** 2 + pt[1] ** 2, abs=1e-8)


def test_lp_extraction_rejects_nonlinear_model():
    m = Model("notlp")
    x = m.continuous("x", lb=0.1, ub=2.0)
    m.minimize(dm.log(x))
    with pytest.raises(Exception):
        extract_lp_data_algebraic(m)
