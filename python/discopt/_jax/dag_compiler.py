"""
DAG Compiler: Expression tree -> jax.numpy callable.

Walks the Expression DAG defined in discopt.modeling.core and produces a pure
jax.numpy function that is jax.jit and jax.grad compatible.

Two entry-point families are provided:

* ``compile_expression`` / ``compile_objective`` / ``compile_constraint`` return
  ``fn(x_flat)``. Parameter values are snapshotted at compile time (legacy
  behavior; kept for callers that do not rebuild between solves).

* ``compile_expression_params`` / ``compile_objective_params`` /
  ``compile_constraint_params`` return ``fn(x_flat, params)`` where ``params``
  is a tuple of jax arrays aligned with ``model._parameters``. The JIT trace
  depends only on shapes, so mutating ``Parameter.value`` between calls hits
  the XLA cache instead of forcing a recompile. Use this for reusable
  evaluators (e.g., NMPC closed-loop solves).

Common-subexpression handling: the Expression object graph is a DAG — a node may
be shared by many parents. Lowering memoizes both *compilation* (each distinct
node builds one closure, via ``memo`` keyed by ``id(expr)``) and *evaluation*
(each distinct node computes one value per call, via a per-call ``cache`` keyed
by ``id(expr)``). A node reachable by k references is therefore traced and run
once, not k times. Without this a linear DAG lowered in time exponential in its
sharing depth (issue #383).
"""

from __future__ import annotations

from typing import Callable, cast

import jax.numpy as jnp

# Import expression types from the modeling API
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    CustomCall,
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
    """Compute the starting offset of a variable in the flat x vector.

    Delegates to the model's memoized prefix-sum table so per-leaf offset
    resolution during the DAG build is O(1) rather than O(n) (issue #654).
    """
    return model._flat_var_offset(var)


_CACHE_MISS = object()


def _compile_node(
    expr: Expression, model: Model, param_index: dict, memo: dict | None = None
) -> Callable:
    """Compile an Expression node into ``f(x_flat, params, cache) -> value``.

    The expression is a DAG: a node may be shared by many parents. ``memo`` (keyed
    by ``id(expr)``) ensures each distinct node is *compiled* once; the per-call
    ``cache`` (also keyed by ``id(expr)``) ensures it is *evaluated* once per call.
    So a node reachable by k references is traced and run a single time — without
    this a linear DAG lowered in time exponential in its sharing depth (#383).
    ``cache`` is a fresh dict supplied per top-level evaluation by the ``compile_*``
    wrappers.
    """
    if memo is None:
        memo = {}
    key = id(expr)
    existing = memo.get(key)
    if existing is not None:
        return cast(Callable, existing)

    raw = _raw_node(expr, model, param_index, memo)

    def node(x_flat, params, cache, _k=key, _raw=raw):
        v = cache.get(_k, _CACHE_MISS)
        if v is _CACHE_MISS:
            v = _raw(x_flat, params, cache)
            cache[_k] = v
        return v

    memo[key] = node
    return node


def _raw_node(expr: Expression, model: Model, param_index: dict, memo: dict) -> Callable:
    """Build the uncached compute closure ``f(x_flat, params, cache)`` for one node.

    Children are compiled via :func:`_compile_node` (sharing ``memo``) and called
    with the same ``cache`` so their values memoize across references.
    """
    if isinstance(expr, Constant):
        val = jnp.array(expr.value)

        def fn(x_flat, params, cache):
            return val

        return fn

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        size = expr.size
        shape = expr.shape
        if shape == () or (len(shape) == 1 and shape[0] == 1 and shape == ()):
            # Scalar variable: single slot
            def fn(x_flat, params, cache):
                return x_flat[offset]

            return fn
        else:
            # Array variable: slice and reshape
            def fn(x_flat, params, cache, _offset=offset, _size=size, _shape=shape):
                return x_flat[_offset : _offset + _size].reshape(_shape)

            return fn

    if isinstance(expr, Parameter):
        idx = param_index[id(expr)]

        def fn(x_flat, params, cache, _i=idx):
            return params[_i]

        return fn

    if isinstance(expr, BinaryOp):
        left_fn = _compile_node(expr.left, model, param_index, memo)
        right_fn = _compile_node(expr.right, model, param_index, memo)
        op = expr.op
        if op == "+":

            def fn(x_flat, params, cache):
                return left_fn(x_flat, params, cache) + right_fn(x_flat, params, cache)
        elif op == "-":

            def fn(x_flat, params, cache):
                return left_fn(x_flat, params, cache) - right_fn(x_flat, params, cache)
        elif op == "*":

            def fn(x_flat, params, cache):
                return left_fn(x_flat, params, cache) * right_fn(x_flat, params, cache)
        elif op == "/":

            def fn(x_flat, params, cache):
                return left_fn(x_flat, params, cache) / right_fn(x_flat, params, cache)
        elif op == "**":

            def fn(x_flat, params, cache):
                return left_fn(x_flat, params, cache) ** right_fn(x_flat, params, cache)
        else:
            raise ValueError(f"Unknown binary operator: {op!r}")
        return fn

    if isinstance(expr, UnaryOp):
        operand_fn = _compile_node(expr.operand, model, param_index, memo)
        op = expr.op
        if op == "neg":

            def fn(x_flat, params, cache):
                return -operand_fn(x_flat, params, cache)
        elif op == "abs":

            def fn(x_flat, params, cache):
                return jnp.abs(operand_fn(x_flat, params, cache))
        else:
            raise ValueError(f"Unknown unary operator: {op!r}")
        return fn

    if isinstance(expr, FunctionCall):
        arg_fns = [_compile_node(a, model, param_index, memo) for a in expr.args]
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
            "sigmoid": lambda x: __import__("jax").nn.sigmoid(x),
            "softplus": lambda x: jnp.logaddexp(x, 0.0),
            "abs": jnp.abs,
            "sign": jnp.sign,
            "entropy": lambda x: x * jnp.log(jnp.maximum(x, 1e-300)),
        }

        if name in _unary_funcs:
            jax_fn = _unary_funcs[name]
            a_fn = arg_fns[0]

            def fn(x_flat, params, cache, _jax_fn=jax_fn, _a_fn=a_fn):
                return _jax_fn(_a_fn(x_flat, params, cache))

            return fn

        if name == "min":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, params, cache):
                return jnp.minimum(a_fn(x_flat, params, cache), b_fn(x_flat, params, cache))

            return fn

        if name == "atan2":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, params, cache):
                return jnp.arctan2(a_fn(x_flat, params, cache), b_fn(x_flat, params, cache))

            return fn

        if name == "signpower":
            # GAMS signpower(x, a) = sign(x) * |x|**a.
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, params, cache):
                xv = a_fn(x_flat, params, cache)
                return jnp.sign(xv) * jnp.abs(xv) ** b_fn(x_flat, params, cache)

            return fn

        if name == "centropy":
            # GAMS centropy(x, y) = x * log(x / y), with the x -> 0+ limit 0.
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, params, cache):
                xv = a_fn(x_flat, params, cache)
                yv = b_fn(x_flat, params, cache)
                return xv * jnp.log(jnp.maximum(xv, 1e-300) / yv)

            return fn

        if name == "max":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_flat, params, cache):
                return jnp.maximum(a_fn(x_flat, params, cache), b_fn(x_flat, params, cache))

            return fn

        if name == "prod":
            a_fn = arg_fns[0]

            def fn(x_flat, params, cache):
                return jnp.prod(a_fn(x_flat, params, cache))

            return fn

        if name.startswith("norm"):
            # norm{p}: p-norm of an array argument (norm1, norm2, ...).
            a_fn = arg_fns[0]
            suffix = name[len("norm") :]
            try:
                ord_p: float = (
                    float(suffix)
                    if suffix not in ("", "inf")
                    else (jnp.inf if suffix == "inf" else 2.0)
                )
            except ValueError as exc:
                raise ValueError(f"Unsupported norm order: {name!r}") from exc

            def fn(x_flat, params, cache, _ord=ord_p):
                return jnp.linalg.norm(a_fn(x_flat, params, cache), ord=_ord)

            return fn

        raise ValueError(f"Unknown function: {name!r}")

    if isinstance(expr, CustomCall):
        # Opaque AD-only user function: trace the stored callable through JAX so
        # value + autodiff gradients/Hessians come for free on the local NLP
        # path. No relaxation rule exists (see relaxation_compiler / solver
        # guards), so this branch is only reached on the continuous NLP path.
        custom_arg_fns = tuple(_compile_node(a, model, param_index, memo) for a in expr.args)
        user_fn = expr.fn

        def fn(x_flat, params, cache, _user_fn=user_fn, _arg_fns=custom_arg_fns):
            return _user_fn(*[a(x_flat, params, cache) for a in _arg_fns])

        return fn

    if isinstance(expr, IndexExpression):
        base_fn = _compile_node(expr.base, model, param_index, memo)
        idx = expr.index

        def fn(x_flat, params, cache, _idx=idx):
            return base_fn(x_flat, params, cache)[_idx]

        return fn

    if isinstance(expr, MatMulExpression):
        left_fn = _compile_node(expr.left, model, param_index, memo)
        right_fn = _compile_node(expr.right, model, param_index, memo)

        def fn(x_flat, params, cache):
            return left_fn(x_flat, params, cache) @ right_fn(x_flat, params, cache)

        return fn

    if isinstance(expr, SumExpression):
        operand_fn = _compile_node(expr.operand, model, param_index, memo)
        axis = expr.axis

        def fn(x_flat, params, cache, _axis=axis):
            return jnp.sum(operand_fn(x_flat, params, cache), axis=_axis)

        return fn

    if isinstance(expr, SumOverExpression):
        term_fns = [_compile_node(t, model, param_index, memo) for t in expr.terms]

        def fn(x_flat, params, cache):
            result = term_fns[0](x_flat, params, cache)
            for t_fn in term_fns[1:]:
                result = result + t_fn(x_flat, params, cache)
            return result

        return fn

    raise TypeError(f"Unhandled expression type: {type(expr).__name__}")


def _build_param_index(model: Model) -> dict:
    """Map ``id(Parameter)`` to its position in ``model._parameters``."""
    return {id(p): i for i, p in enumerate(model._parameters)}


def _snapshot_params(model: Model) -> tuple:
    """Snapshot current parameter values as a tuple of jax arrays."""
    return tuple(jnp.asarray(p.value) for p in model._parameters)


# ---------------------------------------------------------------------------
# Param-aware entry points: returned callables take (x_flat, params).
# ---------------------------------------------------------------------------


def compile_expression_params(
    expr: Expression, model: Model, param_index: dict | None = None
) -> Callable:
    """Compile an Expression DAG into ``fn(x_flat, params)``.

    ``params`` is a tuple of jax arrays aligned with ``model._parameters``.
    The JIT trace is parameter-value-agnostic, so the XLA cache is hit across
    repeated solves that only mutate ``Parameter.value``.
    """
    if param_index is None:
        param_index = _build_param_index(model)
    root = _compile_node(expr, model, param_index)

    def fn(x_flat, params):
        # Fresh per-call value cache so shared DAG nodes evaluate once (#383).
        return root(x_flat, params, {})

    return fn


def compile_objective_params(model: Model, param_index: dict | None = None) -> Callable:
    """Compile the model's objective into ``fn(x_flat, params) -> scalar``."""
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    return compile_expression_params(model._objective.expression, model, param_index)


def compile_constraint_params(
    constraint: Constraint, model: Model, param_index: dict | None = None
) -> Callable:
    """Compile a constraint body into ``fn(x_flat, params) -> scalar/array``."""
    return compile_expression_params(constraint.body, model, param_index)


# ---------------------------------------------------------------------------
# Legacy entry points: returned callables take (x_flat) and snapshot parameter
# values at compile time. Preserved for callers that rebuild per solve.
# ---------------------------------------------------------------------------


def compile_expression(expr: Expression, model: Model) -> Callable:
    """
    Compile an Expression DAG into a pure jax.numpy function.

    Args:
        expr: The expression to compile.
        model: The Model containing variable definitions (needed for index mapping).

    Returns:
        A function f(x_flat) -> scalar/array where x_flat is a 1D jax array
        containing all variable values concatenated in model._variables order.
        Parameter values are snapshotted at compile time; mutate
        ``Parameter.value`` and recompile to pick up changes, or use
        :func:`compile_expression_params` to thread parameters at call time.

    The returned function is compatible with jax.jit, jax.grad, and jax.vmap.
    """
    inner = compile_expression_params(expr, model)
    snapshot = _snapshot_params(model)

    def fn(x_flat):
        return inner(x_flat, snapshot)

    return fn


def compile_objective(model: Model) -> Callable:
    """Compile the model's objective into a jax.numpy function f(x_flat) -> scalar."""
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    return compile_expression(model._objective.expression, model)


def compile_constraint(constraint: Constraint, model: Model) -> Callable:
    """Compile a constraint body into a jax.numpy function f(x_flat) -> scalar/array."""
    return compile_expression(constraint.body, model)
