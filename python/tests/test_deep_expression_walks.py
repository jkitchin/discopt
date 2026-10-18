"""Walks over a loop-built expression must not depend on Python's recursion limit.

``e = e + x[i]`` in a loop builds a chain as deep as the loop is long. #1748 made
``Model.clone()`` and ``pickle`` handle such chains; two recursive walks remained:

* ``discopt.transformations.model_fingerprint`` (and so ``check()`` /
  ``diff_models``) rendered expressions with a recursive ``_canon`` and raised
  ``RecursionError`` at a few thousand terms;
* ``copy.deepcopy`` of an expression on its own, outside a model, did the same.

Both tests below fail on the parent commit with ``RecursionError``. The
fingerprint text must also be unchanged -- it is compared across calls by
``check()`` -- which ``test_canonical_text_is_unchanged`` pins on small cases.
"""

from __future__ import annotations

import copy

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest


def _chain(n: int):
    m = dm.Model("deep")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    e = x[0]
    for i in range(1, n):
        e = e + (i % 7) * x[i]
    m.minimize(e)
    m.subject_to(e >= 3)
    return m, e


@pytest.mark.parametrize("n", [5000, 50000])
def test_fingerprint_of_a_loop_built_model(n):
    m, _ = _chain(n)
    fp = dt.model_fingerprint(m)
    assert dt.model_fingerprint(m.clone()) == fp


@pytest.mark.parametrize("n", [5000, 50000])
def test_deepcopy_of_a_lone_loop_built_expression(n):
    _, e = _chain(n)
    c = copy.deepcopy(e)
    assert c is not e and c.left is not e.left
    assert dt._canon(c) == dt._canon(e)


def test_deepcopy_of_an_expression_keeps_its_sharing():
    m = dm.Model("s")
    x = m.continuous("x", lb=0, ub=1)
    s = dm.exp(x) + 1
    e = s * s
    c = copy.deepcopy(e)
    assert c.left is c.right
    assert c.left is not s


def test_canonical_text_is_unchanged():
    """The iterative renderer emits exactly the text the recursive one did."""
    m = dm.Model("t")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", value=np.array([1.0, 2.5]))
    cases = {
        "x": dt._canon(x[0] + 2.0 * x[1]),
        "call": dt._canon(dm.exp(x[2]) - dm.sum(x[0:2])),
        "matmul": dt._canon(p @ x[0:2]),
        "data": dt._canon(("a", 1, None, [1.5, -0.0], slice(1, None, 2), ())),
    }
    assert cases == {
        "x": "(var(x)[0] + (2.0 * var(x)[1]))",
        "call": "(exp(var(x)[2]) - sum(var(x)[slice(0, 2, None)], axis=None))",
        "matmul": "(param(p=array<<f8(2,)>[1.0, 2.5]) @ var(x)[slice(0, 2, None)])",
        "data": "('a', 1, None, [1.5, -0.0], slice(1, None, 2), (,),)",
    }
