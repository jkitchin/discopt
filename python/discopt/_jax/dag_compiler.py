"""
DAG Compiler: Expression tree -> jax.numpy callable.

Walks the Expression DAG defined in discopt.modeling.core and produces a pure
jax.numpy function that is jax.jit and jax.grad compatible.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

# Import expression types from the modeling API
from discopt.modeling.core import (
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
)


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting offset of a variable in the flat x vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _compile_node(expr: Expression, model: Model) -> Callable:
    """
    Recursively compile an Expression node into a function f(x_flat) -> value.

    Each returned function takes a single 1D jax array (the flat variable vector)
    and returns a jax scalar or array.
    """
    if isinstance(expr, Constant):
        val = jnp.array(expr.value)

        def fn(x_flat):
            return val

        return fn

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        size = expr.size
        shape = expr.shape
        if shape == () or (len(shape) == 1 and shape[0] == 1 and shape == ()):
            # Scalar variable: single slot
            def fn(x_flat):
                return x_flat[offset]

            return fn
        else:
            # Array variable: slice and reshape
            def fn(x_flat, _offset=offset, _size=size, _shape=shape):
                return x_flat[_offset : _offset + _size].reshape(_shape)

            return fn

    if isinstance(expr, Parameter):
        val = jnp.array(expr.value)

        def fn(x_flat):
            return val

        return fn

    if isinstance(expr, BinaryOp):
        left_fn = _compile_node(expr.left, model)
        right_fn = _compile_node(expr.right, model)
        op = expr.op
        if op == "+":

            def fn(x_flat):
                return left_fn(x_flat) + right_fn(x_flat)
        elif op == "-":

            def fn(x_flat):
                return left_fn(x_flat) - right_fn(x_flat)
        elif op == "*":

            def fn(x_flat):
                return left_fn(x_flat) * right_fn(x_flat)
        elif op == "/":

            def fn(x_flat):
                return left_fn(x_flat) / right_fn(x_flat)
        elif op == "**":

            def fn(x_flat):
                return left_fn(x_flat) ** right_fn(x_flat)
        else:
            raise ValueError(f"Unknown binary operator: {op!r}")
        return fn

    if isinstance(expr, UnaryOp):
        operand_fn = _compile_node(expr.operand, model)
        op = expr.op
        if op == "neg":

            def fn(x_flat):
                return -operand_fn(x_flat)
        elif op == "abs":

            def fn(x_flat):
                return jnp.abs(operand_fn(x_flat))
        else:
            raise ValueError(f"Unknown unary operator: {op!r}")
        return fn

    if isinstance(expr, FunctionCall):
        arg_fns = [_compile_node(a, model) for a in expr.args]
        name = expr.func_name

        # Single-argument functions
        _unary_funcs = {
            "exp": jnp.exp,
            "log": jnp.log,
            "log2": jnp.log2,
            "log10": jnp.log10,
            "sqrt": jnp.sqrt,
            "sin": jnp.sin,
            "cos": jnp.cos,
            "tan": jnp.tan,
            "atan": jnp.arctan,
            "sinh": jnp.sinh,
            "cosh": jnp.cosh,
            "asin": jnp.arcsin,
            "acos": jnp.arccos,
            "tanh": jnp.tanh,
            "asinh": jnp.arcsinh,
            "acosh": jnp.arccosh,
            "atanh": jnp.arctanh,
            "erf": lambda x: __import__("jax").scipy.special.erf(x),
            "log1p": jnp.log1p,
            "abs": jnp.abs,
            "sign": jnp.sign,
        }

        if name in _unary_funcs:
            jax_fn = _unary_funcs[name]
            a_fn = arg_fns[0]

            def fn(x_flat, _jax_fn=jax_fn, _a_fn=a_fn):
                return _jax_fn(_a_fn(x_flat))

            return fn

        if name == "min":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat):
                return jnp.minimum(a_fn(x_flat), b_fn(x_flat))

            return fn

        if name == "max":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat):
                return jnp.maximum(a_fn(x_flat), b_fn(x_flat))

            return fn

        if name == "prod":
            a_fn = arg_fns[0]

            def fn(x_flat):
                return jnp.prod(a_fn(x_flat))

            return fn

        if name == "norm2":
            a_fn = arg_fns[0]

            def fn(x_flat):
                return jnp.linalg.norm(a_fn(x_flat), ord=2)

            return fn

        raise ValueError(f"Unknown function: {name!r}")

    if isinstance(expr, IndexExpression):
        base_fn = _compile_node(expr.base, model)
        idx = expr.index

        def fn(x_flat, _idx=idx):
            return base_fn(x_flat)[_idx]

        return fn

    if isinstance(expr, MatMulExpression):
        left_fn = _compile_node(expr.left, model)
        right_fn = _compile_node(expr.right, model)

        def fn(x_flat):
            return left_fn(x_flat) @ right_fn(x_flat)

        return fn

    if isinstance(expr, SumExpression):
        operand_fn = _compile_node(expr.operand, model)
        axis = expr.axis

        def fn(x_flat, _axis=axis):
            return jnp.sum(operand_fn(x_flat), axis=_axis)

        return fn

    if isinstance(expr, SumOverExpression):
        term_fns = [_compile_node(t, model) for t in expr.terms]

        def fn(x_flat):
            result = term_fns[0](x_flat)
            for t_fn in term_fns[1:]:
                result = result + t_fn(x_flat)
            return result

        return fn

    raise TypeError(f"Unhandled expression type: {type(expr).__name__}")


def compile_expression(expr: Expression, model: Model) -> Callable:
    """
    Compile an Expression DAG into a pure jax.numpy function.

    Args:
        expr: The expression to compile.
        model: The Model containing variable definitions (needed for index mapping).

    Returns:
        A function f(x_flat) -> scalar/array where x_flat is a 1D jax array
        containing all variable values concatenated in model._variables order.
        Parameters use their current .value during compilation.

    The returned function is compatible with jax.jit, jax.grad, and jax.vmap.
    """
    return _compile_node(expr, model)


def compile_objective(model: Model) -> Callable:
    """Compile the model's objective into a jax.numpy function f(x_flat) -> scalar."""
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    return compile_expression(model._objective.expression, model)


def compile_constraint(constraint: Constraint, model: Model) -> Callable:
    """Compile a constraint body into a jax.numpy function f(x_flat) -> scalar/array."""
    return compile_expression(constraint.body, model)
