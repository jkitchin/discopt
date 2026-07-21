"""Regression tests for issue #816.

Two independent behaviors on modeling ``Expression`` objects:

1. **Bug (correctness/hang).** ``Variable`` supported ``__getitem__`` but not
   ``__len__``/``__iter__``, and out-of-range indexing returned an expression
   instead of raising ``IndexError``. Python then fell back to the legacy
   sequence-iteration protocol (``x[0]``, ``x[1]``, … until ``IndexError``),
   which never terminated — ``list(x)``/``sum(x)``/``np.array(x)`` hung forever.
   Fix: out-of-range integer indexing raises ``IndexError`` at build time, and
   ``Expression`` defines ``__len__``/``__iter__`` over the leading axis.

2. **Enhancement (shape propagation + assembly).** ``.shape`` now propagates
   through indexing/slicing, element-wise ``BinaryOp``/``UnaryOp`` and
   element-wise ``FunctionCall`` (``sqrt``, ``exp``, …), and ``dm.concatenate`` /
   ``dm.stack`` assemble a vector of scalar expressions from pieces.

All fast; run on every PR.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model

# ─────────────────────────────────────────────────────────────
# Bug: iteration hang + silent out-of-bounds indexing
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_816_out_of_bounds_index_raises():
    m = Model()
    x = m.continuous("x", shape=(5,), lb=0.1, ub=10.0)
    with pytest.raises(IndexError):
        _ = x[99]
    with pytest.raises(IndexError):
        _ = x[-99]
    # In-range (including negative) indices still build an expression.
    assert x[0] is not None
    assert x[-1] is not None


@pytest.mark.unit
def test_816_out_of_bounds_index_raises_2d():
    m = Model()
    X = m.continuous("X", shape=(3, 4), lb=0.0, ub=1.0)
    with pytest.raises(IndexError):
        _ = X[3]
    with pytest.raises(IndexError):
        _ = X[1, 9]
    assert X[2, 3] is not None  # last valid element


@pytest.mark.unit
def test_816_len_and_iter():
    m = Model()
    x = m.continuous("x", shape=(5,), lb=0.1, ub=10.0)
    assert hasattr(x, "__len__")
    assert hasattr(x, "__iter__")
    assert len(x) == 5
    elems = list(x)  # used to hang forever
    assert len(elems) == 5
    # Leading-axis iteration matches explicit indexing.
    assert [repr(e) for e in x] == [repr(x[i]) for i in range(5)]


@pytest.mark.unit
def test_816_iteration_does_not_hang_for_builtins():
    m = Model()
    x = m.continuous("x", shape=(4,), lb=0.0, ub=1.0)
    # sum() over the builtin protocol must terminate and fold to a DAG.
    total = sum(x)
    assert total is not None
    # np.array over the sequence protocol must terminate (object array).
    arr = np.array(list(x), dtype=object)
    assert arr.shape == (4,)


@pytest.mark.unit
def test_816_len_iter_2d_yields_rows():
    m = Model()
    X = m.continuous("X", shape=(3, 4), lb=0.0, ub=1.0)
    assert len(X) == 3
    rows = list(X)
    assert len(rows) == 3
    assert all(r.shape == (4,) for r in rows)


@pytest.mark.unit
def test_816_scalar_indexing_stays_lenient():
    """Indexing a scalar variable is a pre-existing DSL leniency (the LLM builder
    treats a scalar/size-1 variable as indexable); #816 only tightens *array*
    indexing, so scalar indexing must NOT start raising."""
    m = Model()
    s = m.continuous("s", lb=0.0, ub=1.0)  # shape ()
    # Builds an expression instead of raising; shape stays unknown (not inferred).
    assert s[0] is not None
    assert not hasattr(s[0], "shape")


@pytest.mark.unit
def test_816_scalar_len_iter_raise_not_hang():
    m = Model()
    s = m.continuous("s", lb=0.0, ub=1.0)  # shape ()
    with pytest.raises(TypeError):
        len(s)
    with pytest.raises(TypeError):
        iter(s)


@pytest.mark.unit
def test_816_unknown_shape_len_iter_raise_not_hang():
    m = Model()
    a = m.continuous("a", shape=(3,), lb=0.0, ub=1.0)
    red = dm.sum(a)  # reduction → statically unknown shape
    with pytest.raises(TypeError):
        len(red)
    with pytest.raises(TypeError):
        iter(red)


# ─────────────────────────────────────────────────────────────
# Enhancement: .shape propagation
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_816_shape_propagation_through_index_and_ops():
    m = Model()
    x = m.continuous("x", shape=(5,), lb=0.1, ub=10.0)
    assert x.shape == (5,)
    assert x[0].shape == ()
    assert x[:-1].shape == (4,)
    assert x[1:].shape == (4,)
    assert (x[1:] - x[:-1]).shape == (4,)
    assert dm.sqrt(x * x + 0.01).shape == (5,)
    assert dm.sqrt(x[:-1] * x[:-1] + 0.01).shape == (4,)


@pytest.mark.unit
def test_816_shape_propagation_2d_slices():
    m = Model()
    X = m.continuous("X", shape=(3, 4), lb=0.0, ub=1.0)
    assert X[1].shape == (4,)
    assert X[1, 2].shape == ()
    assert X[:, 0].shape == (3,)
    assert (X[:, 1:] - X[:, :-1]).shape == (3, 3)


@pytest.mark.unit
def test_816_shape_unknown_is_attribute_error_not_silent():
    """Shape-unknown nodes must keep ``getattr(x, 'shape', default)`` working."""
    m = Model()
    a = m.continuous("a", shape=(3,), lb=0.0, ub=1.0)
    red = dm.sum(a)
    assert not hasattr(red, "shape")
    assert getattr(red, "shape", ()) == ()
    # A matmul also has statically-unknown shape.
    mm = np.ones((2, 3)) @ a
    assert not hasattr(mm, "shape")
    assert getattr(mm, "shape", ()) == ()


@pytest.mark.unit
def test_816_elementwise_functions_carry_shape():
    m = Model()
    x = m.continuous("x", shape=(4,), lb=0.1, ub=5.0)
    for fn in (dm.exp, dm.log, dm.sin, dm.cos, dm.tanh, dm.sqrt):
        assert fn(x).shape == (4,)
    # element-wise binary min/max broadcast their operand shapes
    assert dm.maximum(x, 0.0).shape == (4,)


# ─────────────────────────────────────────────────────────────
# Enhancement: concatenate / stack
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_816_concatenate_assembles_vector():
    m = Model()
    p = m.continuous("p", shape=(6,), lb=1.0, ub=10.0)
    interior = p[1:] - p[:-1]  # shape (5,)
    # method-of-lines pattern: boundary value, interior values, boundary value
    full = dm.concatenate([[p[0]], interior, [p[-1]]])
    assert full.shape == (7,)
    # Each element is a scalar expression the solver already handles.
    assert full[0].shape == ()


@pytest.mark.unit
def test_816_stack_adds_axis():
    m = Model()
    p = m.continuous("p", shape=(6,), lb=1.0, ub=10.0)
    st = dm.stack([p[:-1], p[1:]])
    assert st.shape == (2, 5)
    assert st[0, 0].shape == ()


@pytest.mark.unit
def test_816_concatenate_requires_known_shape():
    m = Model()
    a = m.continuous("a", shape=(3,), lb=0.0, ub=1.0)
    red = dm.sum(a)  # shapeless
    with pytest.raises(TypeError):
        dm.concatenate([red, a])


@pytest.mark.unit
def test_816_shared_residual_reused_across_backends():
    """The issue's single-source-of-truth goal: one residual, two backends.

    The physics is written once with ``sqrt`` injected; passing ``np.sqrt``
    evaluates numerically and passing ``dm.sqrt`` builds a shaped discopt
    expression — proving the array-style code is reusable verbatim.
    """

    def momentum_residual(q, p_left, p_right, area, dx, K, eps, sqrt):
        dpdx = area * (p_right - p_left) / dx
        friction = K * q * sqrt(q * q + eps * eps) / (0.5 * (p_left + p_right))
        return -(dpdx + friction)

    # numpy leg
    qn = np.array([0.2, 0.3, 0.4])
    pl = np.array([2.0, 2.1, 2.2])
    pr = np.array([2.1, 2.2, 2.3])
    rn = momentum_residual(qn, pl, pr, 2.0, 0.5, 0.1, 1e-3, np.sqrt)
    assert rn.shape == (3,)

    # discopt leg — identical code, shaped expression out
    m = Model()
    q = m.continuous("q", shape=(3,), lb=0.1, ub=5.0)
    p = m.continuous("p", shape=(4,), lb=1.0, ub=10.0)
    rd = momentum_residual(q, p[:-1], p[1:], 2.0, 0.5, 0.1, 1e-3, dm.sqrt)
    assert rd.shape == (3,)
