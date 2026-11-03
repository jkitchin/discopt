"""A size-1 array Variable keeps its declared shape on the POUNCE tape.

``_nl_expr_compiler`` lowered every one-element Variable to a 0-d array, so a
variable declared ``shape=(1,)`` (or ``(1, 1)``) lost its axes. Any array
operation that needs them -- fancy indexing ``x[np.array([0])]``, ``A @ x`` --
then failed with ``UnsupportedForTape`` and the model fell back to the legacy
JAX evaluator, which raises ``MissingJaxError`` when the optional ``jax`` extra
is not installed. The JAX path itself keeps the shape
(``x_flat[off : off + size].reshape(shape)``), so the tape now does too.

Found while building one constraint per MPS row with ``coeffs @ x[cols]``: a
model whose integer block had a single column could not be solved without JAX.
"""

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm
from discopt._nl_expr_compiler import compile_to_nl_array


@pytest.mark.smoke
@pytest.mark.parametrize("shape", [(1,), (1, 1)])
def test_size1_variable_keeps_shape(shape):
    m = dm.Model("t")
    x = m.continuous("x", shape=shape, lb=-5, ub=5)
    arr = compile_to_nl_array(x, m)
    assert arr.shape == shape


@pytest.mark.smoke
def test_scalar_variable_stays_0d():
    m = dm.Model("t")
    x = m.continuous("x", lb=-5, ub=5)
    assert compile_to_nl_array(x, m).shape == ()


@pytest.mark.smoke
def test_fancy_index_and_matmul_on_size1_variable():
    m = dm.Model("t")
    k = m.integer("k", shape=(1,), lb=0, ub=10)
    z = m.binary("z", shape=(2,))
    body = np.array([5.0]) @ k[np.array([0])] + np.array([-1.0, -4.0]) @ z[np.array([0, 1])]
    arr = compile_to_nl_array(body, m)
    assert arr.shape == ()
    x = np.array([2.0, 1.0, 1.0])  # k, z0, z1 in declaration order
    assert arr[()].eval(x) == pytest.approx(5 * 2 - 1 - 4)


@pytest.mark.smoke
def test_size1_fancy_index_model_solves_on_tape():
    """End to end: 5k - z0 - 4 z1 = -3 with z0 + z1 <= 1 has no integer solution.

    Before the fix this raised ``MissingJaxError`` without the jax extra (and
    silently loaded JAX with it).
    """
    m = dm.Model("t")
    k = m.integer("k", shape=(1,), lb=0, ub=10)
    z = m.binary("z", shape=(2,))
    m.subject_to(np.array([5.0]) @ k[np.array([0])] - z[0] - 4 * z[1] == -3)
    m.subject_to(z[0] + z[1] <= 1)
    m.minimize(0 * k[0])
    assert m.solve(time_limit=30).status == "infeasible"
