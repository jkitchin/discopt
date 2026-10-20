"""#1814: ``Model.complementarity`` on a deep operand raised ``RecursionError``.

The MPEC lowering runs when the relation is added, at build time, outside the
solve-time recursion gate (#1808). Two recursive walkers sat on that path: the
interval evaluator (``_elem_shape`` sizes the operand) and ``_is_linear`` (whether
the operand needs a lift). Both now use an explicit stack, including through a
``dm.sum`` nested one level per step.
"""

from __future__ import annotations

import sys
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt._relax.gdp_reformulate import _is_linear

N = 2500


@pytest.fixture
def default_recursion_limit():
    """CPython's default limit: pytest raises it, which hid the overflow."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def _deep_pair():
    m = dm.Model("mpec")
    x = m.continuous("x", shape=N, lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    e = x[0]
    for i in range(1, N):
        e = e + x[i]
    return m, x, y, e


@pytest.mark.smoke
def test_walkers_handle_a_deep_chain(default_recursion_limit):
    m, x, _, e = _deep_pair()
    enc = evaluate_interval(e, m)
    assert float(enc.lo) <= 0.0 and float(enc.lo) > -1e-9
    assert float(enc.hi) >= N and float(enc.hi) < N + 1e-9
    assert _is_linear(e)
    assert not _is_linear(e * x[0])  # a bilinear top node is still refused
    assert not _is_linear(e + x[0] * x[1])  # ...and a bilinear leaf deep in a chain


@pytest.mark.smoke
def test_shared_subexpression_enclosure_unchanged():
    m = dm.Model("dag")
    x = m.continuous("x", shape=3, lb=-1, ub=2)
    s = x[0] * x[1] + x[2]
    enc = evaluate_interval(s * s - s, m)
    # Reference: the same walk on each subterm, combined by interval arithmetic.
    s_enc = evaluate_interval(s, m)
    ref = s_enc * s_enc - s_enc
    assert np.array_equal(np.asarray(enc.lo), np.asarray(ref.lo))
    assert np.array_equal(np.asarray(enc.hi), np.asarray(ref.hi))


@pytest.mark.smoke
def test_complementarity_on_deep_operand_builds(default_recursion_limit):
    m, _, y, e = _deep_pair()
    m.complementarity(e, y)


def _nested_sum(shape, n):
    """``dm.sum`` nested one level per step: one ``SumOverExpression`` per level.

    ``"affine"`` keeps every level on the #1712 fast path's pattern except that a
    term is itself a sum, so the fast path declines at every level;
    ``"bilinear"`` adds a bilinear term per level as well.
    """
    m = dm.Model("nested")
    x = m.continuous("x", shape=n + 1, lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    e = x[0]
    for i in range(1, n):
        if shape == "affine":
            e = dm.sum([e, x[i % 3]])
        else:
            e = dm.sum([e * 1.0, x[i] * x[i + 1]])
    return m, y, e


NESTED = [("affine", 600), ("bilinear", 600), ("bilinear", 2500)]


@pytest.mark.smoke
@pytest.mark.parametrize("shape,n", NESTED)
def test_nested_sum_operand_evaluates(shape, n, default_recursion_limit):
    m, _, e = _nested_sum(shape, n)
    enc = evaluate_interval(e, m)
    # x in [0, 1]: each level adds one term in [0, 1], plus x[0]. Every level's
    # add (and product) rounds outward, ~1e-9 over 2500 levels.
    assert float(enc.lo) <= 0.0 and float(enc.lo) > -1e-6
    assert float(enc.hi) >= n and float(enc.hi) < n + 1e-6


@pytest.mark.smoke
@pytest.mark.parametrize("shape,n", NESTED)
@pytest.mark.parametrize("method", ["gdp", "sos1"])
def test_complementarity_on_nested_sum_builds(shape, n, method, default_recursion_limit):
    m, y, e = _nested_sum(shape, n)
    m.complementarity(e, y, method=method)


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{}, {"method": "sos1"}])
def test_complementarity_on_deep_operand_solves(kw, default_recursion_limit):
    m, x, y, e = _deep_pair()
    m.complementarity(e, y, **kw)
    m.maximize(y - dm.sum(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60)
    # e = sum(x) >= 0 complementary to y: the optimum puts x = 0 and y = 1.
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-6)
