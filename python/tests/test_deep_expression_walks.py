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
from pathlib import Path

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


def test_model_deepcopy_does_not_rewalk_each_node(monkeypatch):
    """#1822: ``Model.__deepcopy__`` copies its nodes children-first itself.

    It used to hand each node to ``copy.deepcopy``, which dispatched to
    ``Expression.__deepcopy__`` and re-walked that node's (already copied)
    children -- once per node, 26-44 % of a model copy. The only nodes still
    copied through the override are the ``x[i]`` a variable memoises in its
    ``_elem_cache``, reached from the variable's own state (one level deep: their
    base is that variable, already in the memo).
    """
    m, _ = _chain(500)
    calls = []
    orig = core.Expression.__deepcopy__

    def counting(self, memo):
        calls.append(self)
        return orig(self, memo)

    monkeypatch.setattr(core.Expression, "__deepcopy__", counting)
    c = copy.deepcopy(m)
    monkeypatch.undo()
    stray = [
        n
        for n in calls
        if not (isinstance(n, core.IndexExpression) and isinstance(n.base, core.Variable))
    ]
    assert not stray, sorted({type(n).__name__ for n in stray})
    # The copy is still the same graph, built from fresh objects.
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)
    assert c._objective.expression is not m._objective.expression
    assert c._constraints[0].body.right is c._objective.expression  # body is 3 - e
    assert c._variables[0] is not m._variables[0]


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


def test_a_pre_wp7_lone_expression_pickle_loses_its_list_on_the_next_copy():
    """A lone expression pickled before WP-7 carried the list in its root's state.

    Without ``Expression.__setstate__`` it unpickles with the list still in the
    root's ``__dict__``; the next pickle, deepcopy or copy drops it. The fixture is
    real: ``x[0] * x[1] + exp(x[2]) - 2`` pickled (protocol 4) by the tree at
    ``1c0f74b3``, the last commit with the state-borne list.
    """
    blob = (
        Path(__file__).parent
        / "data"
        / "pickle_fixtures"
        / "lone_expression_pickled_by_1c0f74b3.pkl"
    ).read_bytes()
    legacy = pickle.loads(blob)
    # The residue the fix is for: present on load, so this test exercises it.
    assert _CHILDREN_FIRST_KEY in vars(legacy)
    model = next(
        n for n in core._descendants_children_first(legacy) if isinstance(n, core.Variable)
    ).model
    checked = 0
    copies = [
        pickle.loads(pickle.dumps(legacy, protocol=p)) for p in range(pickle.HIGHEST_PROTOCOL + 1)
    ]
    copies += [copy.deepcopy(legacy), copy.copy(legacy)]
    for c in copies:
        assert type(c) is type(legacy)
        for node in core._descendants_children_first(c):
            assert _CHILDREN_FIRST_KEY not in vars(node)
            checked += 1
        assert vars(c).keys() == vars(legacy).keys() - {_CHILDREN_FIRST_KEY}
    assert checked > len(copies)
    assert _CHILDREN_FIRST_KEY.encode() not in pickle.dumps(legacy)
    # The model it dragged along re-pickles to the same model.
    again = pickle.loads(pickle.dumps(model))
    assert dt.model_fingerprint(again) == dt.model_fingerprint(model)
    assert again.solve(time_limit=10).objective == pytest.approx(-1.0, abs=1e-6)


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
