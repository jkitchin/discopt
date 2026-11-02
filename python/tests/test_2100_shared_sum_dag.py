"""#2100: a sum shared through its own terms is walked as a DAG, not a tree.

``row = row + row`` repeated ``d`` times is ``d + 1`` nodes and ``2**d`` paths.
``_flatten_sum`` (bound tightening), the exact like-term pass and
``_linearize_affine_expr`` all walked it as a tree, so ``solve(time_limit=1)``
took 11 s at ``d = 24``, 43 s at ``d = 26`` and never returned at ``d = 64``
(measured on ``a624e0f``). The exact pass now collects such a row through its
DAG and rewrites it to ``2**d * x``; the other two walks are bounded.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import pytest
from discopt._relax import nonlinear_bound_tightening as nbt
from discopt.modeling._exact_polynomial import (
    SharedSumOverflowError,
    canonicalize_expression,
)

pytestmark = pytest.mark.smoke


def _shared(base, depth):
    row = base
    for _ in range(depth):
        row = row + row
    return row


def _tree_flatten(expr, scale, out):
    """The pre-#2100 tree walk, for the bit-identity check below the limit."""
    stack = [(expr, scale)]
    while stack:
        node, s = stack.pop()
        if isinstance(node, dm.core.SumOverExpression):
            stack.extend((t, s) for t in reversed(node.terms))
        elif isinstance(node, dm.core.BinaryOp) and node.op in "+-":
            stack.append((node.right, s if node.op == "+" else -s))
            stack.append((node.left, s))
        elif isinstance(node, dm.core.UnaryOp) and node.op == "neg":
            stack.append((node.operand, -s))
        else:
            out.append((s, node))


@pytest.mark.parametrize("depth", [24, 64])
def test_solve_returns_promptly(depth, monkeypatch):
    # 2**64 is past HiGHS's 1e15 coefficient limit for the pure-LP route (a loud
    # refusal, the same for a hand-written ``2**64 * x <= 1``); the in-house LP
    # takes it, which is what this test is about.
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(_shared(x, depth) <= 1)
    m.minimize(x)
    t = time.perf_counter()
    r = m.solve(time_limit=1, threads=1)
    wall = time.perf_counter() - t
    assert wall < 20, wall  # was 11 s at depth 24 and no return at 64
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-9)


def test_shared_row_is_rewritten_to_its_collected_form():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    new = canonicalize_expression(_shared(x - 3 * y, 64) - 1)
    out: list = []
    _tree_flatten(new, 1.0, out)
    assert len(out) == 3, out  # 2**64 * x, -3 * 2**64 * y, -1: no sharing left
    consts = sorted(
        float(n.left.value) for _, n in out if isinstance(n, dm.core.BinaryOp) and n.op == "*"
    )
    assert consts == [float(-3 * 2**64), float(2**64)]


@pytest.mark.parametrize(
    "build, expected",
    [
        # exp(2**-40 * 2**40 x) <= 2, min x -> 0
        (lambda m, x, y: (dm.exp(_shared(x, 40) * 2.0**-40) <= 2, x), 0.0),
        # 2**40 x y <= 1, min -x - y -> -1 (x = 1, y ~ 0)
        (lambda m, x, y: (_shared(x * y, 40) <= 1, -x - y), -1.0),
        # objective 2**40 (x + y) with x + y >= 0.5 -> 2**39
        (lambda m, x, y: (x + y >= 0.5, _shared(x + y, 40)), 2.0**39),
    ],
)
def test_shared_sum_in_other_positions(build, expected):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    row, obj = build(m, x, y)
    m.subject_to(row)
    m.minimize(obj)
    t = time.perf_counter()
    r = m.solve(time_limit=5, threads=1)
    assert time.perf_counter() - t < 30
    assert r.status == "optimal"
    assert r.objective == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_unrepresentable_shared_sum_is_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(_shared(x, 1100) <= 1)  # 2**1100 has no binary64
    m.minimize(x)
    with pytest.raises(SharedSumOverflowError):
        m.solve(time_limit=1, threads=1)


def test_flatten_sum_is_unchanged_below_the_limit():
    m = dm.Model()
    x = m.continuous("x")
    y = m.continuous("y")
    for expr in [
        (x + 2 * y) - (x - y) + -(y + x),
        _shared(x - y, 10),  # shared, but 1024 occurrences: the tree walk
        dm.core.SumOverExpression([x, -y, x]),
    ]:
        want: list = []
        _tree_flatten(expr, 1.5, want)
        got: list = []
        nbt._flatten_sum(expr, 1.5, got)
        assert [(s, id(n)) for s, n in got] == [(s, id(n)) for s, n in want]


def test_flatten_sum_collects_a_shared_dag_exactly():
    m = dm.Model()
    x = m.continuous("x")
    y = m.continuous("y")
    out: list = []
    nbt._flatten_sum(_shared(x - 3 * y, 64) + y, 1.0, out)
    by = {id(n): s for s, n in out}
    assert by[id(x)] == float(2**64)
    # -3 * y is a leaf (``*`` is not an additive edge here); y itself appears once.
    assert sorted(by.values()) == sorted([float(2**64), float(-(2**64)), 1.0])


def test_linearize_affine_expr_is_bounded():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(x <= 1)
    m.minimize(x)
    meta = nbt.build_flat_variable_metadata(m)
    small = nbt._linearize_affine_expr(2.0 * _shared(x, 5), 1.0, meta, 1)
    assert small is not None and small[0][0] == 64.0
    assert nbt._linearize_affine_expr(2.0 * _shared(x, 64), 1.0, meta, 1) is None
