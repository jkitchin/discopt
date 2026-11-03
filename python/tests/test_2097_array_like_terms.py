"""#2097: like terms inside array expressions are collected exactly.

Before the fix, ``canonicalize_expression`` declined every body whose static shape
was not ``()`` and treated ``dm.sum`` / ``@`` of an array expression as an opaque
atom, so ``dm.sum(t + x - t + y)`` with ``t = 2**53 * x`` kept the binary64
cancellation of #2090 and certified a wrong optimum (``optimal``,
``gap_certified=True``). Each model here certified a wrong value on ``a624e0f``.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling._exact_polynomial import (
    ArrayRowCancellationError,
    canonicalize_expression,
)
from discopt.modeling.core import IndexExpression

M = 2.0**53

pytestmark = pytest.mark.smoke


def _solve(m):
    return m.solve(time_limit=60, threads=1)


def _assert_max(r, expected):
    """A certified maximum must equal ``expected``; its bound must not undercut it."""
    assert r.objective == pytest.approx(expected, rel=1e-6, abs=1e-6)
    if r.gap_certified:
        assert r.bound >= expected - 1e-6


def test_sum_of_array_row_with_equal_shapes():
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    y = m.continuous("y", shape=(3,), lb=0, ub=10)
    t = M * x
    m.subject_to(dm.sum(t + x - t + y) <= 6)  # exactly sum(x) + sum(y) <= 6
    m.maximize(dm.sum(x))
    _assert_max(_solve(m), 6.0)


def test_sum_of_array_row_with_a_constant():
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    t = M * x
    m.subject_to(dm.sum(t + x - t + 1) <= 6)  # exactly sum(x) + 3 <= 6
    m.maximize(dm.sum(x))
    _assert_max(_solve(m), 3.0)


def test_shape_one_array_row():
    m = dm.Model()
    x = m.continuous("x", shape=(1,), lb=0, ub=10)
    t = M * x
    m.subject_to(t + 0.5 * x - t + x <= 3)  # exactly 1.5 x <= 3
    m.maximize(dm.sum(x))
    _assert_max(_solve(m), 2.0)


def test_array_row_of_several_elements():
    m = dm.Model()
    x = m.continuous("x", shape=(4,), lb=0, ub=10)
    t = M * x
    m.subject_to(t + 0.5 * x - t + x <= np.array([3.0, 1.5, 6.0, 0.75]))
    m.maximize(dm.sum(x))
    _assert_max(_solve(m), 2.0 + 1.0 + 4.0 + 0.5)


@pytest.mark.parametrize(
    "name,expected",
    [
        # Exactly x0^2 <= 2.
        ("x@Q@x", math.sqrt(2.0)),
        # Exactly x0^2 - x1^2 <= 2 (dm.sum(x**2) also holds x1^2), so x0 = 10.
        ("dm.sum(x**2)", 10.0),
    ],
)
def test_reduction_shares_a_monomial_with_visible_terms(name, expected):
    """The second entry point of the class (#2097 comment): an array reduction
    downstream expands, merging with the visible ``±B x0^2`` terms."""
    B = 2.0**54
    Q = np.array([[1.0, 0.0], [0.0, 0.0]])
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    opaque = x @ Q @ x if name == "x@Q@x" else dm.sum(x**2)
    m.subject_to(B * x[0] ** 2 - opaque - B * x[0] ** 2 + 2 * x[0] ** 2 <= 2)
    m.maximize(x[0])
    _assert_max(_solve(m), expected)


def test_ordinary_array_rows_are_left_as_the_same_object():
    m = dm.Model()
    x = m.continuous("x", shape=(5,), lb=0, ub=1)
    y = m.continuous("y", shape=(5,), lb=0, ub=1)
    X = m.continuous("X", shape=(3, 5), lb=0, ub=1)
    A = np.arange(25.0).reshape(5, 5)
    for body in [
        x[1:] - x[:-1],
        A @ x - np.ones(5),
        dm.sum(X, axis=0) - 1,
        x - y,
        x**2 + 2 * y - 1,
        dm.sum(x**2) - 3,
        dm.sum(A[0] * x) + dm.sum(y) - 1,
    ]:
        assert canonicalize_expression(body) is body


def _indices(node, out):
    from discopt.modeling.core import Expression

    if isinstance(node, IndexExpression):
        out.append(node.index)
    for attr in ("left", "right", "operand", "base"):
        child = getattr(node, attr, None)
        if isinstance(child, Expression):
            _indices(child, out)
    for t in getattr(node, "terms", None) or ():
        _indices(t, out)
    return out


def test_rewritten_array_row_is_exact_and_slice_only():
    """The rewrite reads the same flat slots as the written row and uses only
    basic (int/slice) indices -- a gather would disable the Rust FBBT model."""
    m = dm.Model()
    x = m.continuous("x", shape=(6,), lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    body = M * x[1:] + x[:-1] - M * x[1:] + 0.25 * x[1:] + z - 2
    new = canonicalize_expression(body)
    assert new is not body
    idxs = _indices(new, [])
    assert idxs  # the walk reached the slices (CLAUDE.md §6)
    for idx in idxs:
        parts = idx if isinstance(idx, tuple) else (idx,)
        assert all(isinstance(p, (int, slice)) for p in parts), idx

    rng = np.random.default_rng(0)
    for _ in range(5):
        xv = rng.random(6)
        zv = rng.random()
        exact = xv[:-1] + 0.25 * xv[1:] + zv - 2
        got = _eval(new, {id(x): xv, id(z): np.float64(zv)})
        np.testing.assert_allclose(got, exact, rtol=0, atol=1e-15)


def _eval(node, env):
    from discopt.modeling.core import BinaryOp, Constant, Variable

    if isinstance(node, Constant):
        return np.asarray(node.value, dtype=np.float64)
    if isinstance(node, Variable):
        return env[id(node)]
    if isinstance(node, IndexExpression):
        return np.asarray(_eval(node.base, env))[node.index]
    if isinstance(node, BinaryOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        return {"+": a + b, "-": a - b, "*": a * b, "**": a**b}[node.op]
    raise AssertionError(type(node))


def test_misaligned_array_row_is_refused_loudly():
    """No slice-aligned exact form exists; leaving it would certify the rounded
    row, so the solve refuses instead of answering."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    m.subject_to(M * x + x - M * x + x[::-1] <= 1)
    m.maximize(dm.sum(x))
    with pytest.raises(ArrayRowCancellationError):
        m.solve(time_limit=10, threads=1)


def test_array_fast_pass_is_linear_in_scalar_terms():
    """A body of unknown static shape takes the array pre-pass; with n scalar
    single-leaf terms over one Variable it compared every pair (O(n^2) numpy
    broadcasts): test_1619's 3000-term QP objective spent 43 s there and timed
    out its CI shard. The 0-d slots now go through a set."""
    import time

    from discopt.modeling._exact_polynomial import _fast_clean_array

    n = 6000
    m = dm.Model()
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    from discopt.modeling.core import SumOverExpression

    # An explicit list: ``dm.sum`` of ``2 * x[j]`` would collapse into the affine
    # ``dm.sum(C * x)`` block, which this pass keys coarsely.
    body = SumOverExpression([x[j] ** 2 for j in range(n)] + [2 * x[j] for j in range(n)])
    t = time.perf_counter()
    assert _fast_clean_array(body, {}, []) is True  # x_j**2 and x_j never collide
    assert time.perf_counter() - t < 5.0
    dup = SumOverExpression([x[j] for j in range(n)] + [x[n - 1]])
    assert _fast_clean_array(dup, {}, []) is False  # x[n-1] twice: like terms
