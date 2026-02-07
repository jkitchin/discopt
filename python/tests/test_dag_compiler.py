"""Tests for the JAX DAG compiler."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/jaxminlp_benchmarks")
sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

from jaxminlp_api.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    _wrap,
)
from jaxminlp_api import examples

from discopt._jax.dag_compiler import (
    compile_constraint,
    compile_expression,
    compile_objective,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _flat_size(model: Model) -> int:
    """Total size of the flat variable vector."""
    return sum(v.size for v in model._variables)


def _random_interior_point(model: Model, rng: np.random.Generator) -> jnp.ndarray:
    """Generate a random point strictly inside variable bounds."""
    parts = []
    for v in model._variables:
        lb = np.clip(v.lb.flatten(), -1e3, 1e3)
        ub = np.clip(v.ub.flatten(), -1e3, 1e3)
        # Ensure lb < ub for sampling
        width = np.maximum(ub - lb, 1e-6)
        vals = lb + rng.uniform(0.05, 0.95, size=v.size) * width
        parts.append(vals)
    return jnp.array(np.concatenate(parts), dtype=jnp.float64)


# ─────────────────────────────────────────────────────────────
# Individual Node Type Tests
# ─────────────────────────────────────────────────────────────

class TestConstant:
    def test_scalar_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)  # dummy objective
        expr = Constant(3.14)
        fn = compile_expression(expr, m)
        result = fn(jnp.array([0.5]))
        assert jnp.allclose(result, 3.14)

    def test_array_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        arr = np.array([1.0, 2.0, 3.0])
        expr = Constant(arr)
        fn = compile_expression(expr, m)
        result = fn(jnp.array([0.5]))
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))


class TestVariable:
    def test_scalar_variable(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        fn = compile_expression(x, m)
        result = fn(jnp.array([5.0]))
        assert jnp.allclose(result, 5.0)

    def test_array_variable(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        fn = compile_expression(x, m)
        x_flat = jnp.array([1.0, 2.0, 3.0])
        result = fn(x_flat)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_variable_offset(self):
        m = Model("test")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        y = m.continuous("y", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        fn = compile_expression(y, m)
        x_flat = jnp.array([1.0, 2.0, 10.0, 20.0, 30.0])
        result = fn(x_flat)
        assert jnp.allclose(result, jnp.array([10.0, 20.0, 30.0]))

    def test_2d_variable(self):
        m = Model("test")
        x = m.continuous("x", shape=(2, 3), lb=0, ub=10)
        m.minimize(x[0, 0])
        fn = compile_expression(x, m)
        x_flat = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = fn(x_flat)
        assert result.shape == (2, 3)
        assert jnp.allclose(result, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


class TestBinaryOp:
    def test_add(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        expr = x + y
        fn = compile_expression(expr, m)
        assert jnp.allclose(fn(jnp.array([3.0, 4.0])), 7.0)

    def test_sub(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x - y)
        fn = compile_expression(x - y, m)
        assert jnp.allclose(fn(jnp.array([7.0, 3.0])), 4.0)

    def test_mul(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x * y)
        fn = compile_expression(x * y, m)
        assert jnp.allclose(fn(jnp.array([3.0, 4.0])), 12.0)

    def test_div(self):
        m = Model("test")
        x = m.continuous("x", lb=1, ub=10)
        y = m.continuous("y", lb=1, ub=10)
        m.minimize(x / y)
        fn = compile_expression(x / y, m)
        assert jnp.allclose(fn(jnp.array([6.0, 3.0])), 2.0)

    def test_pow(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x ** 2)
        fn = compile_expression(x ** 2, m)
        assert jnp.allclose(fn(jnp.array([3.0])), 9.0)

    def test_constant_on_left(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(2.0 * x)
        fn = compile_expression(2.0 * x, m)
        assert jnp.allclose(fn(jnp.array([5.0])), 10.0)


class TestUnaryOp:
    def test_neg(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(-x)
        fn = compile_expression(-x, m)
        assert jnp.allclose(fn(jnp.array([3.0])), -3.0)

    def test_abs(self):
        m = Model("test")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(abs(x))
        fn = compile_expression(abs(x), m)
        assert jnp.allclose(fn(jnp.array([-3.0])), 3.0)


class TestFunctionCall:
    def test_exp(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x)
        from jaxminlp_api.core import exp
        fn = compile_expression(exp(x), m)
        assert jnp.allclose(fn(jnp.array([1.0])), jnp.exp(1.0))

    def test_log(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import log
        fn = compile_expression(log(x), m)
        assert jnp.allclose(fn(jnp.array([jnp.e])), 1.0, atol=1e-5)

    def test_log2(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import log2
        fn = compile_expression(log2(x), m)
        assert jnp.allclose(fn(jnp.array([8.0])), 3.0)

    def test_log10(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10000)
        m.minimize(x)
        from jaxminlp_api.core import log10
        fn = compile_expression(log10(x), m)
        assert jnp.allclose(fn(jnp.array([1000.0])), 3.0)

    def test_sqrt(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x)
        from jaxminlp_api.core import sqrt
        fn = compile_expression(sqrt(x), m)
        assert jnp.allclose(fn(jnp.array([9.0])), 3.0)

    def test_sin(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import sin
        fn = compile_expression(sin(x), m)
        assert jnp.allclose(fn(jnp.array([jnp.pi / 2])), 1.0, atol=1e-5)

    def test_cos(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import cos
        fn = compile_expression(cos(x), m)
        assert jnp.allclose(fn(jnp.array([0.0])), 1.0)

    def test_tan(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        from jaxminlp_api.core import tan
        fn = compile_expression(tan(x), m)
        assert jnp.allclose(fn(jnp.array([0.0])), 0.0, atol=1e-6)

    def test_abs_func(self):
        m = Model("test")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import abs_ as abs_fn
        fn = compile_expression(abs_fn(x), m)
        assert jnp.allclose(fn(jnp.array([-5.0])), 5.0)

    def test_sign(self):
        m = Model("test")
        x = m.continuous("x", lb=-10, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import sign
        fn = compile_expression(sign(x), m)
        assert jnp.allclose(fn(jnp.array([-5.0])), -1.0)

    def test_minimum(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import minimum
        fn = compile_expression(minimum(x, y), m)
        assert jnp.allclose(fn(jnp.array([3.0, 5.0])), 3.0)

    def test_maximum(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x)
        from jaxminlp_api.core import maximum
        fn = compile_expression(maximum(x, y), m)
        assert jnp.allclose(fn(jnp.array([3.0, 5.0])), 5.0)

    def test_prod(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=1, ub=10)
        m.minimize(x[0])
        from jaxminlp_api.core import prod
        fn = compile_expression(prod(x), m)
        assert jnp.allclose(fn(jnp.array([2.0, 3.0, 4.0])), 24.0)

    def test_norm2(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        from jaxminlp_api.core import norm
        fn = compile_expression(norm(x), m)
        assert jnp.allclose(fn(jnp.array([3.0, 4.0, 0.0])), 5.0)


class TestIndexExpression:
    def test_scalar_index(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        fn = compile_expression(x[1], m)
        assert jnp.allclose(fn(jnp.array([10.0, 20.0, 30.0])), 20.0)

    def test_tuple_index_2d(self):
        m = Model("test")
        x = m.continuous("x", shape=(2, 3), lb=0, ub=10)
        m.minimize(x[0, 0])
        fn = compile_expression(x[1, 2], m)
        x_flat = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert jnp.allclose(fn(x_flat), 6.0)

    def test_slice_index(self):
        m = Model("test")
        x = m.continuous("x", shape=(2, 3), lb=0, ub=10)
        m.minimize(x[0, 0])
        # x[0, :] should give first row
        fn = compile_expression(x[0, :], m)
        x_flat = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = fn(x_flat)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))


class TestMatMulExpression:
    def test_matrix_vector(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        expr = MatMulExpression(Constant(A), x)
        fn = compile_expression(expr, m)
        x_flat = jnp.array([3.0, 5.0, 7.0])
        result = fn(x_flat)
        assert jnp.allclose(result, jnp.array([3.0, 5.0]))

    def test_matmul_operator(self):
        m = Model("test")
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(x[0])
        c = np.array([2.0, 3.0])
        # numpy's @ intercepts before __rmatmul__, so wrap explicitly
        expr = Constant(c) @ x
        fn = compile_expression(expr, m)
        assert jnp.allclose(fn(jnp.array([4.0, 5.0])), 23.0)


class TestSumExpression:
    def test_sum_all(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        from jaxminlp_api.core import sum as jm_sum
        expr = jm_sum(x)
        fn = compile_expression(expr, m)
        assert jnp.allclose(fn(jnp.array([1.0, 2.0, 3.0])), 6.0)

    def test_sum_axis(self):
        m = Model("test")
        x = m.continuous("x", shape=(2, 3), lb=0, ub=10)
        m.minimize(x[0, 0])
        from jaxminlp_api.core import sum as jm_sum
        expr = jm_sum(x, axis=0)
        fn = compile_expression(expr, m)
        x_flat = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = fn(x_flat)
        assert jnp.allclose(result, jnp.array([5.0, 7.0, 9.0]))


class TestSumOverExpression:
    def test_indexed_sum(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        m.minimize(x[0])
        from jaxminlp_api.core import sum as jm_sum
        c = np.array([2.0, 3.0, 4.0])
        expr = jm_sum(lambda i: c[i] * x[i], over=range(3))
        fn = compile_expression(expr, m)
        assert jnp.allclose(fn(jnp.array([1.0, 2.0, 3.0])), 2 + 6 + 12)


class TestParameter:
    def test_parameter_baked_in(self):
        m = Model("test")
        p = m.parameter("price", value=42.0)
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(p * x)
        fn = compile_expression(p * x, m)
        assert jnp.allclose(fn(jnp.array([2.0])), 84.0)

    def test_array_parameter(self):
        m = Model("test")
        p = m.parameter("prices", value=np.array([10.0, 20.0]))
        x = m.continuous("x", shape=(2,), lb=0, ub=10)
        m.minimize(x[0])
        from jaxminlp_api.core import sum as jm_sum
        expr = jm_sum(lambda i: p[i] * x[i], over=range(2))
        fn = compile_expression(expr, m)
        assert jnp.allclose(fn(jnp.array([1.0, 2.0])), 50.0)


class TestConstraintCompilation:
    def test_le_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        con = (x + y <= 5)  # body = (x + y) - 5
        fn = compile_constraint(con, m)
        # At x=2, y=1: body = (2+1) - 5 = -2
        assert jnp.allclose(fn(jnp.array([2.0, 1.0])), -2.0)

    def test_eq_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        con = (x == y)  # body = x - y
        fn = compile_constraint(con, m)
        assert jnp.allclose(fn(jnp.array([3.0, 3.0])), 0.0)

    def test_ge_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        con = (x + y >= 5)  # internally stored as 5 - (x+y) <= 0, body = 5 - (x+y)
        fn = compile_constraint(con, m)
        # At x=4, y=3: body = 5 - 7 = -2
        assert jnp.allclose(fn(jnp.array([4.0, 3.0])), -2.0)


# ─────────────────────────────────────────────────────────────
# Tests on the 7 example models
# ─────────────────────────────────────────────────────────────

def _build_all_examples():
    """Build all 7 example models."""
    return [
        ("simple_minlp", examples.example_simple_minlp),
        ("pooling_haverly", examples.example_pooling_haverly),
        ("process_synthesis", examples.example_process_synthesis),
        ("portfolio", examples.example_portfolio),
        ("reactor_design", examples.example_reactor_design),
        ("facility_location", examples.example_facility_location),
        ("parametric", examples.example_parametric),
    ]


class TestJITTraceability:
    """Verify that jax.make_jaxpr succeeds for all 7 example objectives."""

    @pytest.mark.parametrize("name,builder", _build_all_examples())
    def test_jaxpr_traces(self, name, builder, capsys):
        model = builder()
        fn = compile_objective(model)
        n = _flat_size(model)
        x = jnp.ones(n)
        # make_jaxpr should succeed without error
        jaxpr = jax.make_jaxpr(fn)(x)
        assert jaxpr is not None


class TestGradientCorrectness:
    """Verify jax.grad returns finite values at random interior points."""

    @pytest.mark.parametrize("name,builder", _build_all_examples())
    def test_grad_finite(self, name, builder, capsys):
        model = builder()
        fn = compile_objective(model)
        n = _flat_size(model)

        # Wrap to ensure scalar output for grad
        def scalar_fn(x):
            val = fn(x)
            # Some objectives may already be scalar; ensure it
            return jnp.sum(val)

        grad_fn = jax.grad(scalar_fn)
        rng = np.random.default_rng(42)

        n_points = 100
        for i in range(n_points):
            x = _random_interior_point(model, rng)
            g = grad_fn(x)
            assert jnp.all(jnp.isfinite(g)), (
                f"Non-finite gradient at point {i} for {name}: {g}"
            )


class TestNumericalCorrectness:
    """Verify compiled functions produce correct numerical results."""

    def test_simple_minlp_objective(self):
        """minimize x1^2 + x2^2 + x3 at x1=1, x2=2, x3=0 => 5.0"""
        m = examples.example_simple_minlp()
        fn = compile_objective(m)
        # x1=1.0, x2=2.0, x3=0.0
        x = jnp.array([1.0, 2.0, 0.0])
        result = fn(x)
        assert jnp.allclose(result, 5.0), f"Expected 5.0, got {result}"

    def test_simple_minlp_gradient(self):
        """grad of x1^2 + x2^2 + x3 at (1,2,0) => [2, 4, 1]"""
        m = examples.example_simple_minlp()
        fn = compile_objective(m)
        grad_fn = jax.grad(fn)
        x = jnp.array([1.0, 2.0, 0.0])
        g = grad_fn(x)
        expected = jnp.array([2.0, 4.0, 1.0])
        assert jnp.allclose(g, expected, atol=1e-5), f"Expected {expected}, got {g}"

    def test_simple_minlp_constraints(self):
        """Test constraints of the simple MINLP example."""
        m = examples.example_simple_minlp()
        # Constraint 0: x1 + x2 >= 1 => stored as 1 - (x1+x2) <= 0
        # body = 1 - x1 - x2
        con0 = m._constraints[0]
        fn0 = compile_constraint(con0, m)
        # At x1=0.5, x2=0.6, x3=0: body should be 1 - 0.5 - 0.6 = -0.1
        # Actually, >= is: body = _wrap(other) - self = 1 - (x1+x2)
        x = jnp.array([0.5, 0.6, 0.0])
        val = fn0(x)
        assert jnp.allclose(val, -0.1, atol=1e-5), f"Expected -0.1, got {val}"

    def test_parametric_objective(self):
        """Test that parameters are baked in correctly."""
        m = examples.example_parametric()
        fn = compile_objective(m)
        # The parametric model has: price_A=50, price_B=30
        # x is blend[2,2] (4 slots), y is use_source[2] (2 slots)
        # Objective: price_A * sum(x[0,:]) + price_B * sum(x[1,:]) + 1000 * sum(y)
        # Set x = [[1,1],[2,2]], y = [1,1]
        # = 50*(1+1) + 30*(2+2) + 1000*(1+1)
        # = 100 + 120 + 2000 = 2220
        x_flat = jnp.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
        result = fn(x_flat)
        assert jnp.allclose(result, 2220.0, atol=1e-3), f"Expected 2220.0, got {result}"


class TestCompileObjectiveAndConstraint:
    def test_compile_objective_no_objective(self):
        m = Model("empty")
        with pytest.raises(ValueError, match="no objective"):
            compile_objective(m)

    def test_compile_objective_returns_callable(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x ** 2)
        fn = compile_objective(m)
        assert callable(fn)
        assert jnp.allclose(fn(jnp.array([3.0])), 9.0)


class TestUnhandledExpression:
    def test_unknown_type_raises(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)

        class FakeExpression(Expression):
            pass

        with pytest.raises(TypeError, match="Unhandled expression type"):
            compile_expression(FakeExpression(), m)


class TestJIT:
    """Verify compiled functions work under jax.jit."""

    def test_jit_simple(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x ** 2 + y ** 2)
        fn = jax.jit(compile_objective(m))
        result = fn(jnp.array([3.0, 4.0]))
        assert jnp.allclose(result, 25.0)

    @pytest.mark.parametrize("name,builder", _build_all_examples())
    def test_jit_examples(self, name, builder, capsys):
        model = builder()
        fn = jax.jit(compile_objective(model))
        n = _flat_size(model)
        rng = np.random.default_rng(123)
        x = _random_interior_point(model, rng)
        result = fn(x)
        assert jnp.all(jnp.isfinite(result)), f"Non-finite JIT result for {name}: {result}"
