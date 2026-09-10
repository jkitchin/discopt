"""The modelling guide's promises about array-valued bodies must hold (#1215).

`docs/notebooks/modeling_guide.ipynb` now names the array-valued constraint body
as discopt's default idiom and makes specific, checkable claims about it. The
book is built with `execute_notebooks: "off"`, so its cells are not re-run at
build time and nothing else would notice if one of those claims stopped being
true. These tests are what notices.

Each test corresponds to a sentence in the guide:

* "one Python object describing n rows" -> one `Constraint` per family;
* "the identical model ... byte-for-byte the same" -> `.nl` equality with the
  per-element form, which covers row ORDER, not just row count;
* "every math function ... is elementwise over an array body" -> the function
  list in the guide's API reference, exercised one by one;
* "the right-hand side can be a numpy array ... or a scalar";
* "`name='c'` names the family; the rows become `c[0]`, `c[1]`, ...";
* "reductions work too", "matrix products work", "integrality is unaffected".

The point is not to re-test the relaxation layer. It is that a documented idiom
with no test is a promise with no guard.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export import to_nl
from discopt.modeling import Model

N = 6


def _model():
    m = Model("doc")
    x = m.continuous("x", shape=(N,), lb=0.5, ub=4.0)
    y = m.continuous("y", shape=(N,), lb=0.5, ub=4.0)
    z = m.binary("z", shape=(N,))
    return m, x, y, z


def _rows(model):
    return int(to_nl(model).split("\n")[1].split()[1])


def test_an_array_body_is_one_constraint_object_for_n_rows():
    m, x, y, _ = _model()
    m.subject_to(dm.exp(x) + y <= 9.0, name="c")
    m.minimize(dm.sum(x))
    assert len(m._constraints) == 1
    assert _rows(m) == N


def test_array_and_per_element_forms_write_byte_identical_nl():
    """Covers row order, which is how a solver's `.sol` maps duals back."""
    rhs = 3.0 + np.arange(N) * 0.5

    vec, x, y, _ = _model()
    vec.subject_to(dm.exp(x) * 0.1 + y <= rhs, name="c")
    vec.subject_to(x * y >= 1.0, name="d")
    vec.minimize(dm.sum(x) + dm.sum(y))

    elem, xe, ye, _ = _model()
    for i in range(N):
        elem.subject_to(dm.exp(xe[i]) * 0.1 + ye[i] <= float(rhs[i]), name=f"c_{i}")
    for i in range(N):
        elem.subject_to(xe[i] * ye[i] >= 1.0, name=f"d_{i}")
    elem.minimize(dm.sum(xe) + dm.sum(ye))

    assert len(vec._constraints) == 2
    assert len(elem._constraints) == 2 * N
    text = to_nl(vec)
    assert text == to_nl(elem)
    # An all-empty match would pass the line above; pin the size too.
    assert _rows(vec) == 2 * N


@pytest.mark.parametrize(
    "label,body",
    [
        ("exp", lambda x, y, z: dm.exp(x) + y),
        ("log", lambda x, y, z: dm.log(x) + y),
        ("sqrt", lambda x, y, z: dm.sqrt(x) + y),
        ("sin", lambda x, y, z: dm.sin(x) + y),
        ("cos", lambda x, y, z: dm.cos(x) + y),
        ("abs", lambda x, y, z: dm.abs(x - y)),
        ("pow", lambda x, y, z: x**2 + y),
        ("bilinear", lambda x, y, z: x * y),
        ("div", lambda x, y, z: x / y),
        ("affine", lambda x, y, z: x + 2.0 * y),
        ("ndarray coeff", lambda x, y, z: np.arange(N) * x + y),
        ("binary term", lambda x, y, z: x * y + z),
    ],
)
def test_documented_functions_are_elementwise_over_an_array_body(label, body):
    m, x, y, z = _model()
    m.subject_to(body(x, y, z) <= 9.0, name="c")
    m.minimize(dm.sum(x))
    assert len(m._constraints) == 1, label
    assert _rows(m) == N, label


def test_rhs_may_be_an_array_or_a_scalar():
    for rhs in (np.linspace(1.0, 2.0, N), 9.0):
        m, x, y, _ = _model()
        m.subject_to(x + y <= rhs, name="c")
        m.minimize(dm.sum(x))
        assert _rows(m) == N


def test_a_named_family_expands_to_indexed_row_names():
    """`name="c"` names the FAMILY; the written rows are `c_0`, `c_1`, ...

    The model itself keeps one `Constraint` called `c` -- the expansion happens
    in the writer, not in the model -- so the names have to be read out of an
    output format that carries them. `.nl` does not (row names live in a
    separate `.row` file), so this reads the LP.
    """
    from discopt.export import to_lp

    m, x, y, _ = _model()
    m.subject_to(x + y <= 9.0, name="c")
    m.minimize(dm.sum(x))

    assert [c.name for c in m._constraints] == ["c"]
    written = to_lp(m)
    for i in range(N):
        assert f"c_{i}:" in written, f"row c_{i} missing from the LP"
    assert "c_%d:" % N not in written


def test_a_full_reduction_is_one_row():
    m, x, _, _ = _model()
    m.subject_to(dm.sum(x) <= 9.0, name="c")
    m.minimize(dm.sum(x))
    assert _rows(m) == 1


def test_an_axis_reduction_reduces_a_2d_family():
    m = Model("axis")
    X = m.continuous("X", shape=(2, 3), lb=0.5, ub=4.0)
    m.subject_to(dm.sum(X, axis=1) <= 9.0, name="c")
    m.minimize(dm.sum(X))
    assert _rows(m) == 2


def test_a_matrix_product_body_gives_one_row_per_matrix_row():
    m, x, _, _ = _model()
    m.subject_to(np.eye(N) @ x <= np.ones(N), name="c")
    m.minimize(dm.sum(x))
    assert len(m._constraints) == 1
    assert _rows(m) == N


def test_a_2d_variable_takes_an_array_body():
    m = Model("twod")
    X = m.continuous("X", shape=(2, 3), lb=0.5, ub=4.0)
    m.subject_to(X + 1.0 <= 9.0, name="c")
    m.minimize(dm.sum(X))
    assert _rows(m) == 6


def test_variable_has_no_reshape_so_the_guide_says_declare_the_shape():
    """The guide tells the reader to declare the shape; hold that reason."""
    m = Model("noreshape")
    x = m.continuous("x", shape=(6,), lb=0.5, ub=4.0)
    assert not hasattr(x, "reshape")


@pytest.mark.slow
def test_both_idioms_solve_to_the_same_objective():
    rhs = 3.0 + np.arange(4) * 0.5

    def build(vectorised):
        m = Model("s")
        x = m.continuous("x", shape=(4,), lb=0.5, ub=4.0)
        y = m.continuous("y", shape=(4,), lb=0.5, ub=4.0)
        if vectorised:
            m.subject_to(dm.exp(x) * 0.1 + y <= rhs, name="c")
            m.subject_to(x * y >= 1.0, name="d")
        else:
            for i in range(4):
                m.subject_to(dm.exp(x[i]) * 0.1 + y[i] <= float(rhs[i]), name=f"c{i}")
                m.subject_to(x[i] * y[i] >= 1.0, name=f"d{i}")
        m.minimize(dm.sum(x) + dm.sum(y))
        return m.solve()

    elem, vec = build(False), build(True)
    assert elem.status == vec.status
    assert elem.objective == pytest.approx(vec.objective, abs=1e-9)
