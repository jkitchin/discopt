"""Tests for the public parametric-compilation API (discopt.parametric).

This module is the stable contract external plugins (discopt-doe, ...) build
on, so these tests pin the public names, the flat-vector layout, and the
differentiability guarantees — independently of the ``_jax`` internals that
implement them (covered in ``test_parametric.py``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import subprocess
import sys
from types import SimpleNamespace

import discopt.modeling as dm
import jax
import numpy as np
import pytest
from discopt.parametric import (
    compile_expression,
    compile_response_function,
    extract_x_flat,
    flatten_params,
    param_total_size,
    variable_slices,
    variable_total_size,
)

pytestmark = pytest.mark.smoke


def _model():
    """y = a*x1 + b*x2**2 with one Parameter c: exercises vars and params."""
    m = dm.Model()
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-10, ub=10)
    c = m.parameter("c", value=3.0)
    x = m.continuous("x", lb=0, ub=5)
    y = a * x + b * x**2 + c
    return m, y


class TestCompileExpression:
    def test_value_matches_analytic(self):
        m, y = _model()
        fn = compile_expression(y, m)
        # x_flat layout: a, b, x (declaration order); p_flat: c
        x_flat = np.array([2.0, 0.5, 4.0])
        p_flat = np.array([3.0])
        # 2*4 + 0.5*16 + 3 = 19
        assert float(fn(x_flat, p_flat)) == pytest.approx(19.0)

    def test_gradient_wrt_variables(self):
        m, y = _model()
        fn = compile_expression(y, m)
        x_flat = np.array([2.0, 0.5, 4.0])
        p_flat = np.array([3.0])
        g = jax.grad(fn, argnums=0)(x_flat, p_flat)
        # dy/da = x = 4, dy/db = x^2 = 16, dy/dx = a + 2bx = 2 + 4 = 6
        np.testing.assert_allclose(np.asarray(g), [4.0, 16.0, 6.0])

    def test_gradient_wrt_parameters(self):
        m, y = _model()
        fn = compile_expression(y, m)
        g = jax.grad(fn, argnums=1)(np.array([2.0, 0.5, 4.0]), np.array([3.0]))
        np.testing.assert_allclose(np.asarray(g), [1.0])  # dy/dc = 1


class TestCompileResponseFunction:
    def test_ordering_matches_dict(self):
        m = dm.Model()
        x = m.continuous("x", lb=0, ub=10)
        fn = compile_response_function({"double": 2 * x, "square": x * x}, m)
        out = np.asarray(fn(np.array([3.0]), np.zeros(0)))
        np.testing.assert_allclose(out, [6.0, 9.0])
        assert fn.response_names == ["double", "square"]
        assert fn.n_responses == 2

    def test_empty_responses_raises(self):
        m = dm.Model()
        m.continuous("x", lb=0, ub=1)
        with pytest.raises(ValueError):
            compile_response_function({}, m)


class TestFlatVectorLayout:
    def test_variable_slices_layout(self):
        m = dm.Model()
        m.continuous("s", lb=0, ub=1)
        m.continuous("v", lb=0, ub=1, shape=(3,))
        m.continuous("t", lb=0, ub=1)
        slices = variable_slices(m)
        assert slices == {"s": slice(0, 1), "v": slice(1, 4), "t": slice(4, 5)}
        assert variable_total_size(m) == 5

    def test_extract_x_flat_consistent_with_slices(self):
        m = dm.Model()
        m.continuous("s", lb=0, ub=10)
        m.continuous("v", lb=0, ub=10, shape=(2,))
        result = SimpleNamespace(x={"s": 1.5, "v": np.array([2.5, 3.5])})
        x_flat = np.asarray(extract_x_flat(result, m))
        slices = variable_slices(m)
        np.testing.assert_allclose(x_flat[slices["s"]], [1.5])
        np.testing.assert_allclose(x_flat[slices["v"]], [2.5, 3.5])

    def test_extract_x_flat_no_solution_raises(self):
        m = dm.Model()
        m.continuous("s", lb=0, ub=1)
        with pytest.raises(ValueError):
            extract_x_flat(SimpleNamespace(x=None), m)

    def test_flatten_params_concatenates_in_order(self):
        m = dm.Model()
        m.parameter("p1", value=1.0)
        m.parameter("p2", value=np.array([2.0, 3.0]))
        np.testing.assert_allclose(np.asarray(flatten_params(m)), [1.0, 2.0, 3.0])
        assert param_total_size(m) == 3

    def test_flatten_params_empty(self):
        m = dm.Model()
        m.continuous("x", lb=0, ub=1)
        p_flat = np.asarray(flatten_params(m))
        assert p_flat.shape == (0,)
        assert param_total_size(m) == 0


def test_import_is_jax_free():
    """`import discopt.parametric` must not pull jax (plugin CLI light floor)."""
    code = "import sys; import discopt.parametric; sys.exit(1 if 'jax' in sys.modules else 0)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
