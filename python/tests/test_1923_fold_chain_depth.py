"""#1923: a shallow objective must not become a recursion-limit-deep expression.

``fold_affine_constants`` rebuilt every affine form as a left-folded chain, one
level per term. Re-distributing an already-distributed Goldstein-Price objective
(``ex8_1_3``) folded 1,372 terms into a 1,374-level chain, and the recursive term
estimator (then the distributor) raised ``RecursionError`` at the default limit.
"""

import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.term_classifier import (
    _FOLD_CHAIN_MAX,
    distribute_products,
    estimate_distributed_terms,
    fold_affine_constants,
)
from discopt.modeling.core import BinaryOp, UnaryOp


def _depth(e):
    best, stack = 0, [(e, 1)]
    while stack:
        n, d = stack.pop()
        best = max(best, d)
        if isinstance(n, BinaryOp):
            stack += [(n.left, d + 1), (n.right, d + 1)]
        elif isinstance(n, UnaryOp):
            stack.append((n.operand, d + 1))
    return best


def _goldstein_price(x, y):
    a = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    b = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    return a * b


@pytest.mark.unit
def test_redistributing_goldstein_price_stays_shallow():
    m = dm.Model("gp")
    x = m.continuous("x")
    y = m.continuous("y")
    e1 = distribute_products(_goldstein_price(x, y))
    folded = fold_affine_constants(e1)
    assert _depth(folded) < 100
    e2 = distribute_products(e1)  # RecursionError before the fix
    assert _depth(e2) < 100
    assert estimate_distributed_terms(folded) > _FOLD_CHAIN_MAX


@pytest.mark.unit
def test_term_estimate_is_iterative_on_a_deep_chain():
    m = dm.Model("chain")
    xs = [m.continuous(f"x{i}") for i in range(5)]
    e = xs[0]
    for i in range(5000):  # far deeper than the default recursion limit
        e = e + xs[i % 5] * xs[(i + 1) % 5]
    assert estimate_distributed_terms(e) == 5001


@pytest.mark.unit
def test_long_fold_is_balanced_and_equal():
    m = dm.Model("long")
    xs = [m.continuous(f"x{i}", lb=-1, ub=1) for i in range(400)]
    e = 1.0 + xs[0]
    for i, v in enumerate(xs[1:], start=1):
        e = (e - 2.0 * v + 0.5) if i % 3 == 0 else (e + v + v)
    f = fold_affine_constants(e)
    assert _depth(f) <= 12
    rng = np.random.default_rng(0)
    vals = {id(v): val for v, val in zip(xs, rng.uniform(-1, 1, len(xs)))}

    def ev(n):
        if isinstance(n, BinaryOp):
            a, b = ev(n.left), ev(n.right)
            return {"+": a + b, "-": a - b, "*": a * b}[n.op]
        if isinstance(n, UnaryOp):
            return -ev(n.operand)
        if id(n) in vals:
            return vals[id(n)]
        return float(np.asarray(n.value))

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    assert ev(f) == pytest.approx(ev(e), rel=1e-12, abs=1e-9)


@pytest.mark.slow
@pytest.mark.correctness
def test_goldstein_price_free_variables_solves():
    m = dm.Model("goldstein_price")
    x = m.continuous("x", lb=-float("inf"), ub=float("inf"))
    y = m.continuous("y", lb=-float("inf"), ub=float("inf"))
    m.minimize(_goldstein_price(x, y))
    r = m.solve(time_limit=8)  # RecursionError before the fix
    assert r.objective == pytest.approx(3.0, abs=1e-5)
