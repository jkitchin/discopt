"""Regression tests for issue #739 — abs() models misclassified as LP.

``ExprArena::max_degree`` treated ``UnOp::Abs`` of a linear operand as degree 1
("abs(linear) as degree 1" for structure detection), so ``|x|`` objectives and
constraints classified as LP. The LP fast path's extractors then probed the model
pointwise (unit-vector / Jacobian samples), which cannot distinguish a piecewise
linear function from a linear one: they baked in one side's slope, turning
``min |x|`` into ``min x`` (false optimal certificate) and ``|x| <= c`` into
``x <= c`` (constraint silently dropped, violated at the returned point).

The fix makes ``UnOp::Abs`` of anything variable-dependent non-polynomial
(``usize::MAX``), matching the existing ``FunctionCall(Abs)`` handling, so such
models classify NLP/MINLP and route to the spatial B&B whose piecewise
relaxation certifies them exactly.

Each solve assertion below FAILS on pre-fix ``main`` (status='optimal' with
objective -1.0 / -2.0) and passes after the fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.modeling.core import Model

# ── classification: abs must never classify as LP/QP ─────────────────────────


@pytest.mark.smoke
def test_abs_objective_not_classified_lp():
    from discopt._jax.problem_classifier import ProblemClass, classify_problem

    m = Model("absmin")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.minimize(abs(x))
    assert classify_problem(m) == ProblemClass.NLP


@pytest.mark.smoke
def test_abs_constraint_not_classified_lp():
    from discopt._jax.problem_classifier import ProblemClass, classify_problem

    m = Model("abscon")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.subject_to(abs(x) <= 0.5)
    m.minimize(x)
    assert classify_problem(m) == ProblemClass.NLP


@pytest.mark.smoke
def test_abs_nested_in_linear_sum_not_classified_lp_or_qp():
    """abs buried in an otherwise linear/quadratic body still poisons the degree."""
    from discopt._jax.problem_classifier import ProblemClass, classify_problem

    m = Model("absnested")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(2.0 * y + abs(x - 1.0) <= 1.5)
    m.minimize(x + y)
    assert classify_problem(m) == ProblemClass.NLP


@pytest.mark.smoke
def test_abs_objective_rust_repr_not_linear():
    """Probe the Rust fix directly: |x| is neither linear nor quadratic."""
    model_to_repr = pytest.importorskip("discopt._rust").model_to_repr

    m = Model("absrepr")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.minimize(abs(x))
    repr_ = model_to_repr(m, None)
    assert not repr_.is_objective_linear()
    assert not repr_.is_objective_quadratic()


@pytest.mark.smoke
def test_plain_linear_model_still_classified_lp():
    """Control: the abs refusal must not over-reach onto genuinely linear models."""
    from discopt._jax.problem_classifier import ProblemClass, classify_problem

    m = Model("lin")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.subject_to(x >= -0.5)
    m.minimize(2.0 * x + 1.0)
    assert classify_problem(m) == ProblemClass.LP


# ── end-to-end: the false-optimal repros from the issue ──────────────────────


@pytest.mark.smoke
def test_abs_objective_solves_to_true_optimum():
    """min |x| over [-1, 2]: true optimum 0 at x=0 (pre-fix: -1.0 at x=-1)."""
    m = Model("absmin")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.minimize(abs(x))
    res = m.solve(time_limit=30.0)
    assert res.status == "optimal"
    assert abs(res.objective - 0.0) < 1e-5, f"got {res.objective}, want 0.0"
    xv = float(np.asarray(res.x["x"]).reshape(()))
    # The reported objective must match the model at the returned point.
    assert abs(abs(xv) - res.objective) < 1e-5


@pytest.mark.smoke
def test_abs_constraint_not_dropped():
    """min x s.t. |x| <= 0.5 over [-2, 2]: true optimum -0.5 (pre-fix: -2.0,
    with the constraint violated at the returned point)."""
    m = Model("abscon")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.subject_to(abs(x) <= 0.5)
    m.minimize(x)
    res = m.solve(time_limit=30.0)
    assert res.status == "optimal"
    xv = float(np.asarray(res.x["x"]).reshape(()))
    assert abs(xv) <= 0.5 + 1e-5, f"|x| <= 0.5 violated at returned point x={xv}"
    assert abs(res.objective - (-0.5)) < 1e-4, f"got {res.objective}, want -0.5"
