"""Construction-cost invariants of the modeling layer (#1215).

discopt's modeling layer is the foundation every other layer sits on -- the
solvers, the exporters, mb-DoE, and any process-modelling layer above it -- and
it must stay usable at 100k variables/constraints. Measured against Pyomo 6.10.1
on a 40-form x 5 000-instance model, it was 2.98x slower and 1.93x heavier per
row, and an attribution put **96% of a build in expression-node creation**, with
two specific causes:

* every ``x[i]`` allocated a fresh ``IndexExpression`` (1.99 us against Pyomo's
  0.18 us, an 11.3x gap) where Pyomo returns the one canonical ``VarData``;
* every node was thought to carry an expensive per-instance ``__dict__``. That
  second reading was *falsified* by measurement -- see
  ``test_expression_nodes_are_not_slotted`` -- and the remaining gap is in the
  ``Constraint`` wrapper and ``Model.constraint``\'s bookkeeping, not the nodes.

These are structural properties, not timings, so they are testable without a
benchmark: this file pins the properties, and
``discopt_benchmarks/scripts/bench_model_construction.py`` measures the effect.
A timing assertion here would be flaky on a shared runner and is deliberately
absent.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import (
    Constant,
    IndexExpression,
)

pytestmark = pytest.mark.smoke


def test_integer_index_returns_the_canonical_node():
    """``x[i]`` must be the same object every time, as Pyomo's ``VarData`` is.

    Identity here is not cosmetic: expressions hash by identity, so the
    ``id()``-keyed memos carried by every walker, relaxation and tape lowering
    only hit across rows when the element node is shared.
    """
    m = dm.Model("t")
    x = m.continuous("x", shape=(4, 3), lb=0.0, ub=1.0)
    assert x[0] is x[0]
    assert x[2] is x[2]
    assert x[0] is not x[1]
    assert x[1, 2] is x[1, 2]
    assert x[1, 2] is not x[2, 1]


def test_index_type_is_never_silently_changed():
    """``x[0]`` and ``x[(0,)]`` must not collapse onto one another.

    Consumers branch on ``IndexExpression.index`` being an ``int`` versus a
    ``tuple`` (``_resolve_var_index`` in the ``.nl`` writer does), so a cache key
    that normalised the two together would hand back a node whose index has a
    different type than the caller wrote.
    """
    m = dm.Model("t")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=1.0)
    assert isinstance(x[0].index, int)
    assert isinstance(x[(0,)].index, tuple)
    assert x[0] is not x[(0,)]


def test_out_of_range_index_still_raises_before_anything_is_cached():
    """The bounds guard must run on every call, including a repeat."""
    m = dm.Model("t")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
    for _ in range(2):
        with pytest.raises(IndexError):
            x[7]
    assert x[2] is x[2]


def test_slices_and_numpy_ints_stay_uncached_but_work():
    """Uncached index forms must keep working, just without canonicalisation."""
    m = dm.Model("t")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=1.0)
    sl = x[0:2]
    assert isinstance(sl, IndexExpression)
    a, b = x[np.int64(0)], x[np.int64(0)]
    assert isinstance(a, IndexExpression) and isinstance(b, IndexExpression)


def test_expression_nodes_are_not_slotted():
    """Slotting the node classes is a measured dead end -- do not re-add it.

    Blaming the per-node ``__dict__`` for ~104 B looked obvious and was wrong:
    CPython 3.12 gives same-shape instances a key-sharing dict, so slotting the
    hot node classes moved retained memory 144.9 -> 143.1 MB (1.2%) on the
    200 000-row panel with no time change. Slots would buy that 1% at the price
    of forbidding attribute assignment on any ``Expression`` subclass, including
    in plugins -- and ``_relax/scalarize.py`` already memoises onto nodes.
    """
    m = dm.Model("t")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    for node in (x[0], x[0] + x[1], dm.exp(x[0]), Constant(1.0)):
        assert hasattr(node, "__dict__"), f"{type(node).__name__} was slotted"


def test_expression_nodes_remain_weak_referenceable():
    """Guards the slotting dead end from a partial re-introduction.

    Slotting without declaring ``__weakref__`` silently removes
    weak-referenceability, which would break ``solver.py``'s ``WeakKeyDictionary``
    caches with an opaque ``TypeError`` far from the cause.
    """
    import weakref

    m = dm.Model("t")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    for node in (x[0], x[0] + x[1], dm.exp(x[0])):
        assert weakref.ref(node)() is node


def test_scalarize_memo_still_writable():
    """``_relax/scalarize.py`` memoises a shape onto the node; slots must allow it."""
    m = dm.Model("t")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    node = x[0] + x[1]
    node._scalarize_shape = ()
    assert node._scalarize_shape == ()


def test_shared_element_nodes_do_not_change_the_model():
    """Canonical elements must not alter what the model means.

    The same element appearing in several rows is now one node; the solved
    answer must be unchanged.
    """
    m = dm.Model("share")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=4.0)
    m.subject_to(x[0] + x[1] <= 3.0, name="a")
    m.subject_to(x[0] + x[2] <= 2.0, name="b")
    m.subject_to(x[0] * x[0] <= 4.0, name="c")
    m.maximize(x[0] + x[1] + x[2])
    res = m.solve(time_limit=60)
    assert res.status in ("optimal", "feasible")
    # x0 <= 2 (from c), and maximising gives x0 + (3 - x0) + (2 - x0) = 5 - x0,
    # so the optimum takes x0 = 0 and reaches 5.
    assert res.objective == pytest.approx(5.0, abs=1e-5)


def test_parameter_rebind_normalises_and_keeps_the_model_lowerable():
    """``p.value = 7.0`` is the fitting-loop idiom and must not break lowering.

    ``value`` was a plain attribute, so ``__init__`` established the documented
    ndarray invariant and the next assignment broke it, leaving a bare float that
    ``model_to_repr`` refused with ``TypeError: 'float' object cannot be
    converted to 'PyArray<T, D>'``.
    """
    from discopt._rust import model_to_repr

    m = dm.Model("p")
    x = m.continuous("x", lb=0.1, ub=5.0)
    p = m.parameter("p", value=2.0)
    m.minimize(p * x * x)

    p.value = 7.0
    assert isinstance(p.value, np.ndarray)
    assert p.shape == ()
    model_to_repr(m, getattr(m, "_builder", None))  # must not raise


def test_parameter_shape_change_is_refused():
    """A new shape would invalidate every arena node and tape already built."""
    m = dm.Model("p")
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="shape"):
        p.value = 3.0
    np.testing.assert_array_equal(p.value, np.array([1.0, 2.0]))
