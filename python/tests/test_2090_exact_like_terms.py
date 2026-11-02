"""#2090: like terms added in binary64 changed a row, and the wrong optimum was certified.

``t + x - t`` with ``t = 2**53 * x`` is ``x`` exactly, but every float collector
(the LP extractor, the Rust polynomial canonicalise, the arena walk, pointwise
evaluation) computes ``2**53 + 1 -> 2**53`` and reads ``0 * x``. Rows and objectives
are now collected exactly at the solve doors
(:mod:`discopt.modeling._exact_polynomial`) and rewritten to their canonical form
only when float collection could round.
"""

from __future__ import annotations

from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling._exact_polynomial import (
    _sum_is_order_exact,
    canonicalize_expression,
)
from discopt.modeling.core import Constant, SumOverExpression

M = 2.0**53


# ── the three reproducers from the issue ───────────────────────────────────────


@pytest.mark.smoke
def test_linear_row_certifies_true_optimum():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    t = M * x
    m.subject_to(((t + x) - t) - x + x + y == 3)
    m.minimize(x + 2 * y + 7)
    r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(10.0, abs=1e-6)
    assert r.bound <= 10.0 + 1e-6
    np.testing.assert_allclose([r.value(x), r.value(y)], [3.0, 0.0], atol=1e-6)


@pytest.mark.smoke
def test_coefficient_row_certifies_true_optimum():
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=10)
    t = M * z
    m.subject_to(t + 0.5 * z - t + z <= 3)
    m.maximize(z)
    r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-6)
    assert r.bound >= 2.0 - 1e-6
    assert 1.5 * r.value(z) <= 3 + 1e-6


@pytest.mark.smoke
def test_minlp_row_certifies_true_optimum():
    m = dm.Model()
    xs, selectors = [], []
    for i in range(2):
        xi = m.continuous(f"x{i}", lb=3, ub=6)
        a, b = m.binary(f"a{i}"), m.binary(f"b{i}")
        m.subject_to(a + b == 1)
        m.subject_to(xi == 3 * a + 6 * b)
        xs.append(xi)
        selectors.append(a)
    m.subject_to(xs[0] + xs[1] <= 9)
    m.minimize(6000 / xs[0] + 12000 / xs[1])
    a = selectors[0]
    t = M * a
    m.subject_to(((t + a) - t) - a == 0)
    r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(4000.0, rel=1e-6)
    assert r.bound <= 4000.0 * (1 + 1e-6)


# ── the canonicaliser ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ordinary_bodies_are_returned_unchanged():
    """No rewrite when every summation order is exact: the same object comes back."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    bodies = [
        x[0] + 2 * x[1] - 3,
        x[0] + x[0] + 0.5 * x[0] - 1.25,  # like terms, but exact in every order
        (x[0] + 1) * (y - 1) + 3 * y,
        x[0] * y + y * x[0],
        x[0] / 3 + y,  # a non-dyadic single contribution is not a sum
        dm.exp(x[0] + x[1]) + y**2,
        np.array([1.0, 2.0, 3.0]) @ x + y,
        dm.sum(x) + y - 1.2 - 0.5,
    ]
    for body in bodies:
        assert canonicalize_expression(body) is body, body


@pytest.mark.unit
def test_cancelling_like_terms_are_collected_exactly():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    out = canonicalize_expression(((M * x + x) - M * x) - x + x + y)
    assert out is not None
    # Evaluate the canonical body pointwise in floats: it is x + y exactly.
    vals = {"x": 0.25, "y": 0.5}
    assert _eval(out, vals) == 0.75


@pytest.mark.unit
def test_cancellation_is_caught_inside_a_function_argument():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    body = dm.exp((M * x + x) - M * x)
    out = canonicalize_expression(body)
    assert out is not body
    assert _eval(out, {"x": 1.0}) == pytest.approx(np.e)


@pytest.mark.unit
def test_cancellation_inside_a_constant_divisor():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    body = x / (Constant(M) + 1.0 - M)  # the divisor is exactly 1, 0 in floats
    out = canonicalize_expression(body)
    assert out is not body
    assert _eval(out, {"x": 0.5}) == 0.5


@pytest.mark.unit
def test_quadratic_like_terms_and_full_cancellation():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    out = canonicalize_expression(M * x * y + x * y - M * (y * x))
    assert _eval(out, {"x": 0.5, "y": 0.5}) == 0.25
    zero = canonicalize_expression(M * x + x - M * x - x)
    assert isinstance(zero, Constant) and float(zero.value) == 0.0


@pytest.mark.unit
def test_rows_are_rewritten_in_place_keeping_identity():
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=10)
    c = M * z + 0.5 * z - M * z + z <= 3
    m.subject_to(c, name="row")
    m.maximize(z)
    m.validate()  # the solve door
    assert m._constraints[-1] is c
    assert isinstance(c.body, SumOverExpression)
    assert c._exact_body is c.body  # cached: a re-solve does not walk it again
    assert _eval(c.body, {"z": 2.0}) == 0.0  # 1.5 * 2 - 3


@pytest.mark.unit
def test_constraint_family_fast_path_is_exact():
    """``Model.constraint``'s single-variable fast path reads the canonical body."""
    m = dm.Model()
    s = m.set("I", [0, 1])
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.constraint(s, lambda i: M * x[i] + 0.5 * x[i] - M * x[i] + x[i] <= 3, name="cap")
    m.maximize(x[0] + x[1])
    r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(4.0, abs=1e-6)


@pytest.mark.unit
def test_objective_like_terms():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    big = 2.0**54  # 2**54 - 1 is not a binary64, so the float row reads 0 * x
    m.minimize((big * x - x) - big * x + 3)  # exactly -x + 3
    r = m.solve(time_limit=60, threads=1)
    assert r.objective == pytest.approx(2.0, abs=1e-6)
    assert r.value(x) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
    "contribs, exact",
    [
        ([1, 2, 3], True),
        ([M, 1, -M], False),  # M + 1 rounds in that order
        ([M, -M], True),
        ([0.5, 0.25, 1e10], True),
        ([0.1, 0.2], False),
        ([Fraction(1, 3), Fraction(1, 3)], False),
        ([7], True),
    ],
)
def test_order_exactness_certificate(contribs, exact):
    assert _sum_is_order_exact([Fraction(c) for c in contribs]) is exact


def _eval(expr, values: dict) -> float:
    """Pointwise float evaluation of a small scalar expression tree."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        UnaryOp,
        Variable,
    )

    if isinstance(expr, Constant):
        return float(expr.value)
    if isinstance(expr, Variable):
        return values[expr.name]
    if isinstance(expr, IndexExpression):
        return values[expr.base.name][expr.index]
    if isinstance(expr, SumOverExpression):
        return float(sum(_eval(t, values) for t in expr.terms))
    if isinstance(expr, UnaryOp):
        assert expr.op == "neg"
        return -_eval(expr.operand, values)
    if isinstance(expr, FunctionCall):
        return float(getattr(np, expr.func_name)(*[_eval(a, values) for a in expr.args]))
    assert isinstance(expr, BinaryOp)
    a, b = _eval(expr.left, values), _eval(expr.right, values)
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "**": a**b}[expr.op]


@pytest.mark.unit
def test_solve_model_door_without_validate():
    """``solve_model`` is public and skips ``Model.validate``; it collects too."""
    from discopt.solver import solve_model

    m = dm.Model()
    z = m.continuous("z", lb=0, ub=10)
    m.subject_to(M * z + 0.5 * z - M * z + z <= 3)
    m.maximize(z)
    r = solve_model(m, time_limit=60, threads=1)
    assert r.objective == pytest.approx(2.0, abs=1e-6)


@pytest.mark.unit
def test_disjunct_and_indicator_rows_are_collected():
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=10)
    d = m.either_or([[M * z + 0.5 * z - M * z + z <= 3], [z <= 1]], name="d")
    m.maximize(z)
    m.validate()
    row = d.disjuncts[0][0]
    assert _eval(row.body, {"z": 2.0}) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "contribs, benign",
    [
        ([M, 1, -M], False),  # the #2090 cancellation: kappa ~ 2**54
        ([0.1, 0.2], True),  # ulp-level rounding, no cancellation
        ([1.25, 0.8], True),  # ex1222's constants
        ([12.96, -18.0, -18.0, 25.0], True),  # nvs04's expanded square, kappa ~ 35
        ([Fraction(1, 3), Fraction(-1, 3)], False),  # inexact terms cancelling to 0
        ([M, -M], True),  # exact in every order, whatever kappa
    ],
)
def test_benign_sum_criterion(contribs, benign):
    from discopt.modeling._exact_polynomial import _sum_is_benign

    assert _sum_is_benign([Fraction(c) for c in contribs]) is benign


@pytest.mark.unit
def test_ulp_level_like_terms_are_not_rewritten():
    """Ordinary decimal data with like terms keeps its structure (smoke regressions)."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    for body in [
        0.1 * x + 0.2 * x + y,
        -9.53e-05 * x * y - 9.53e-05 * y * x,
        x + 1.25 + 0.8,
        (x - 1.8) ** 2 + 0.3 * x,
    ]:
        assert canonicalize_expression(body) is body, body
