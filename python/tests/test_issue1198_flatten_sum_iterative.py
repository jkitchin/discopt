"""Issue #1198 — `_flatten_sum` must not recurse once per additive term.

A long linear row parses to a left-deep ``((a+b)+c)+d…`` chain whose depth
equals its term count, so the recursive additive walk used by the nonlinear
bound-tightening rules exhausted the Python stack on models with a few thousand
terms (`t1000` in MINLPLib). The `RecursionError` escaped
`tighten_nonlinear_bounds` and the instance dropped out of any panel touching
this path — a silent size cliff, not a soundness bug.

The walk is now iterative over an explicit ``(expr, scale)`` stack. These tests
fail before that rewrite (`RecursionError` at ~1000 terms) and pass after. The
equivalence test pins the rewrite to the recursive semantics it replaced: the
same leaves, in the same order, with the same scales.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.nonlinear_bound_tightening import (
    BinaryOp,
    SumOverExpression,
    UnaryOp,
    _flatten_sum,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

# Comfortably past CPython's default recursion limit (1000), so the pre-fix walk
# dies here regardless of how many frames the test harness already occupies.
DEEP_N = 20_000


def _left_deep_sum(terms):
    """Build ``((t0 + t1) + t2) + …`` — the shape a long linear row parses to."""
    expr = terms[0]
    for term in terms[1:]:
        expr = expr + term
    return expr


def _recursive_flatten_sum(expr, scale: float, out: list[tuple[float, object]]) -> None:
    """The pre-#1198 recursive walk, kept here as the equivalence oracle."""
    if isinstance(expr, SumOverExpression):
        for term in expr.terms:
            _recursive_flatten_sum(term, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _recursive_flatten_sum(expr.left, scale, out)
        _recursive_flatten_sum(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _recursive_flatten_sum(expr.left, scale, out)
        _recursive_flatten_sum(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _recursive_flatten_sum(expr.operand, -scale, out)
        return
    out.append((scale, expr))


def test_flatten_sum_handles_a_left_deep_sum_of_20000_terms():
    """The additive walk's depth must be bounded by the heap, not the C stack."""
    m = Model("deep_sum")
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(DEEP_N)]

    terms: list[tuple[float, object]] = []
    _flatten_sum(_left_deep_sum(xs), 1.0, terms)  # pre-fix: RecursionError

    assert len(terms) == DEEP_N
    # Order and scale are preserved all the way down the chain.
    assert [t is x for (_, t), x in zip(terms, xs)] == [True] * DEEP_N
    assert all(scale == 1.0 for scale, _ in terms)


def test_flatten_sum_handles_a_left_deep_difference():
    """`-` alternates the carried scale; depth must not matter to that either."""
    m = Model("deep_diff")
    xs = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(DEEP_N)]

    expr = xs[0]
    for x in xs[1:]:
        expr = expr - x

    terms: list[tuple[float, object]] = []
    _flatten_sum(expr, 1.0, terms)

    assert len(terms) == DEEP_N
    assert terms[0][0] == 1.0
    assert all(scale == -1.0 for scale, _ in terms[1:])


def test_flatten_sum_matches_the_recursive_walk_term_for_term():
    """Bit-for-bit equivalence with the walk this replaced, on mixed structure.

    Sums, differences, `neg`, `SumOverExpression`, non-additive leaves (squares,
    products, scaled variables) and both signs of the incoming scale.
    """
    m = Model("mixed")
    xs = [m.continuous(f"x{i}", lb=-2.0, ub=3.0) for i in range(6)]

    exprs = [
        xs[0] + xs[1] - xs[2],
        -(xs[0] ** 2 + xs[1] ** 2),
        (xs[0] - (xs[1] + xs[2])) - (-(xs[3] * xs[4])),
        2.5 * xs[0] + (-1.25) * xs[1] - (xs[2] ** 2 - xs[3]),
        sum(xs[1:], xs[0]),
        -(-(xs[0] + xs[1]) - xs[2]) + xs[3] ** 2,
        _left_deep_sum([x**2 for x in xs]) - _left_deep_sum(xs),
    ]

    comparisons = 0
    for expr in exprs:
        for scale in (1.0, -1.0, 0.75, -3.0):
            expected: list[tuple[float, object]] = []
            actual: list[tuple[float, object]] = []
            _recursive_flatten_sum(expr, scale, expected)
            _flatten_sum(expr, scale, actual)

            assert len(actual) == len(expected)
            for (exp_scale, exp_term), (act_scale, act_term) in zip(expected, actual):
                assert act_term is exp_term  # same leaf, same position
                assert act_scale == exp_scale  # exact, not approx: bounds move on a ulp
                comparisons += 1

    # §6: prove the probe fired — a silently empty comparison loop reads as a pass.
    assert comparisons > 0


def test_tighten_nonlinear_bounds_survives_a_deep_row():
    """End to end: the rule that hit this on `t1000` now derives its bounds.

    ``sum_i x_i^2 <= 9`` over a left-deep row of 20k squares implies
    ``|x_i| <= 3``; pre-fix the `RecursionError` escaped the whole pass.
    """
    m = Model("deep_sos")
    xs = [m.continuous(f"x{i}", lb=-100.0, ub=100.0) for i in range(DEEP_N)]
    m.subject_to(_left_deep_sum([x**2 for x in xs]) <= 9.0)
    m.minimize(xs[0])

    lb0, ub0 = flat_variable_bounds(m)
    tlb, tub, stats = tighten_nonlinear_bounds(m, lb0.copy(), ub0.copy())

    assert not stats.infeasible
    assert "sum_of_squares_upper_bound" in stats.applied_rules

    # Nothing looser than the input box, and the radius-3 ball bounds are found.
    assert np.all(tlb >= lb0 - 1e-12)
    assert np.all(tub <= ub0 + 1e-12)
    assert tub == pytest.approx(np.full(DEEP_N, 3.0))
    assert tlb == pytest.approx(np.full(DEEP_N, -3.0))
