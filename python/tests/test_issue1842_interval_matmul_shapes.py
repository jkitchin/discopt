"""#1842: interval matmul enclosures had the wrong shape outside ``(m,k) @ (k,)``.

``_eval_matmul`` enclosed only a matrix times a vector; every other product
fell through to an unbounded interval of shape ``A.shape[:-1]`` -- 0-d for
``(k,) @ (k, n)`` -- so ``(x + c @ M - 10)[0]`` raised ``IndexError: array is
0-dimensional`` in the root relaxation build. With the shape alone fixed the
constant ``sqrt(sum(M @ M))`` still had no finite enclosure and the dual bound
stayed at the variable box (status ``feasible``, bound 20, 8000+ nodes in 15 s).
All 1-D / 2-D products now get the sound interval dot product, and the sibling
fallbacks (``**`` with a non-scalar exponent, multi-argument calls, unknown
binary ops) report the broadcast shape.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt.modeling.core import Constant

pytestmark = pytest.mark.smoke

M = np.array([[3.0, -1.0], [-2.0, 1.0]])


def test_issue_repro_solves_to_optimum():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    row = (x + Constant(np.ones(2)) @ Constant(M) - 10)[0] + dm.sqrt(
        dm.sum(Constant(M) @ Constant(M))
    )
    m.subject_to(row <= 0)
    m.maximize(x)
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    # (ones @ M)[0] = 1, sum(M @ M) = 2.
    assert float(r.value(x)) == pytest.approx(9.0 - np.sqrt(2.0), abs=1e-6)


SHAPES = [((3,), (3, 2)), ((2, 3), (3, 4)), ((3,), (3,)), ((2, 3), (3,))]


@pytest.mark.parametrize("sa,sb", SHAPES, ids=[f"{a}@{b}" for a, b in SHAPES])
def test_matmul_enclosure_shape_and_soundness(sa, sb):
    rng = np.random.default_rng(1842)
    m = dm.Model()
    a = m.continuous("a", shape=sa, lb=-2.0, ub=3.0)
    b = m.continuous("b", shape=sb, lb=-1.5, ub=0.5)
    enc = evaluate_interval(a @ b, m)
    want = np.matmul(np.zeros(sa), np.zeros(sb)).shape
    assert enc.lo.shape == want and enc.hi.shape == want
    assert np.all(np.isfinite(enc.lo)) and np.all(np.isfinite(enc.hi))
    checks = 0
    for _ in range(2000):
        av = rng.uniform(-2.0, 3.0, sa)
        bv = rng.uniform(-1.5, 0.5, sb)
        val = av @ bv
        assert np.all(enc.lo <= val) and np.all(val <= enc.hi)
        checks += 1
    # Corners attain the interval dot-product extremes; check them too.
    for av in (np.full(sa, -2.0), np.full(sa, 3.0)):
        for bv in (np.full(sb, -1.5), np.full(sb, 0.5)):
            val = av @ bv
            assert np.all(enc.lo <= val) and np.all(val <= enc.hi)
            checks += 1
    assert checks == 2004


def test_constant_matmul_enclosure_is_tight():
    enc = evaluate_interval(dm.sum(Constant(M) @ Constant(M)), None)
    assert float(enc.lo) == pytest.approx(2.0) and float(enc.hi) == pytest.approx(2.0)
    assert float(enc.lo) <= 2.0 <= float(enc.hi)


def test_batched_matmul_unbounded_with_product_shape():
    m = dm.Model()
    a = m.continuous("a", shape=(2, 2, 3), lb=0, ub=1)
    enc = evaluate_interval(a @ Constant(np.ones((3, 4))), m)
    assert enc.lo.shape == (2, 2, 4)


def test_sibling_fallbacks_report_broadcast_shape():
    m = dm.Model()
    x = m.continuous("x", lb=1, ub=2)
    enc = evaluate_interval(x ** Constant(np.array([1.0, 2.0, 3.0])), m)
    assert enc.lo.shape == (3,)
