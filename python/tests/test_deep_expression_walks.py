"""Walks over a loop-built expression must not depend on Python's recursion limit.

``e = e + x[i]`` in a loop builds a chain as deep as the loop is long. #1748 made
``Model.clone()`` and ``pickle`` handle such chains; three recursive walks remained:

* ``discopt.transformations.model_fingerprint`` (and so ``check()`` /
  ``diff_models``) rendered expressions with a recursive ``_canon`` and raised
  ``RecursionError`` at a few thousand terms;
* ``copy.deepcopy`` of an expression on its own, outside a model, did the same;
* so did ``pickle`` of an expression on its own, past a few thousand terms.

The deep cases below fail on ``main`` with ``RecursionError``. The
fingerprint text must also be unchanged -- it is compared across calls by
``check()`` -- which ``test_canonical_text_is_unchanged`` pins on small cases.
"""

from __future__ import annotations

import copy
import pickle

import discopt.modeling as dm
import discopt.modeling.core as core
import discopt.transformations as dt
import numpy as np
import pytest

#: The state key the children-first list rode in before WP-7 moved it into the
#: reduce arguments; asserted absent from every unpickled node.
_CHILDREN_FIRST_KEY = "_pickle_children_first"


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


@pytest.mark.parametrize("n", [5000, 50000])
def test_pickle_of_a_lone_loop_built_expression(n):
    _, e = _chain(n)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        c = pickle.loads(pickle.dumps(e, protocol=protocol))
        assert dt._canon(c) == dt._canon(e)
        assert _CHILDREN_FIRST_KEY not in vars(c)


def test_unpickling_a_node_takes_no_python_setstate():
    """WP-7: the children-first list rides in the outer node's reduce *arguments*.

    It rode in the node's state until then, and stripping it back out took a
    Python ``Expression.__setstate__`` on every node of every unpickle -- measured
    at +126 % on unpickling a 200k-node expression against the pre-#1764 reader
    (``docs/dev/data/serialize-pickle-ab-2026-10-19.jsonl``). Now every node but
    ``Variable`` (read-only bounds, #1332) unpickles through the C default, and the
    outer node's state is exactly what ``object.__reduce_ex__`` gives.
    """
    assert "__setstate__" not in vars(core.Expression)
    _, e = _chain(50)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        rv = e.__reduce_ex__(protocol)
        assert rv[0] is core._new_after_children, protocol
        assert rv[2] == object.__reduce_ex__(e, protocol)[2], protocol
        # The scope the reduce opened closes only when the sentinel is pickled.
        core._PICKLE_SCOPES.open = 0


def test_pickled_expression_keeps_identity_with_its_model():
    """The model a pickled expression drags along holds the SAME unpickled node."""
    m, e = _chain(5000)
    c = pickle.loads(pickle.dumps(e))
    v = c.right.right.base
    assert v.model is not m
    assert v.model._objective.expression is c


def test_pickled_expression_keeps_its_sharing():
    m = dm.Model("s")
    x = m.continuous("x", lb=0, ub=1)
    s = dm.exp(x) + 1
    c = pickle.loads(pickle.dumps(s * s))
    assert c.left is c.right


def test_copy_copy_of_an_expression_stays_shallow():
    _, e = _chain(50)
    c = copy.copy(e)
    assert c is not e and c.left is e.left and c.right is e.right
    assert _CHILDREN_FIRST_KEY not in vars(c)


def test_children_first_scopes_close():
    """Every scope a pickle opens is closed by the time it returns."""
    open_before = getattr(core._PICKLE_SCOPES, "open", 0)
    m, e = _chain(2000)
    for _ in range(3):
        pickle.loads(pickle.dumps(e))
        pickle.loads(pickle.dumps(m))
    assert getattr(core._PICKLE_SCOPES, "open", 0) == open_before
