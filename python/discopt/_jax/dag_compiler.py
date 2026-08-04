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
be shared by many parents. Lowering assigns each distinct node (keyed by
``id(expr)``) exactly one *tape slot*, so it is both compiled once and evaluated
once per call. A node reachable by k references is therefore traced and run once,
not k times. Without this a linear DAG lowered in time exponential in its sharing
depth (issue #383).

Depth handling: the DAG walk and the evaluation are both **iterative** (issue
#925). A plain Python ``sum``/``+=`` over a list of terms builds a *left-nested*
``BinaryOp`` chain whose depth equals the term count, so a recursive lowering —
or a lowering into nested child closures, which pushes the same depth onto the
stack at *call* time — raised ``RecursionError`` on well-formed models of only a
few hundred terms. Instead the DAG is flattened once into a post-order tape of
``(kernel, child_slots)`` entries; both the flattening and the per-call
evaluation are ordinary loops, so compile and eval depth are bounded by the heap
rather than by the C stack.
"""

from __future__ import annotations

from typing import Callable

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


def _build_tape(expr: Expression, model: Model, param_index: dict) -> list[tuple]:
    """Flatten an Expression DAG into a post-order evaluation tape.

    Returns a list of ``(kernel, child_slots)`` entries in dependency order: every
    entry's children occupy strictly earlier slots, and the root is the final
    entry. Each distinct node (keyed by ``id(expr)``) gets exactly one slot, so a
    node shared by k parents is compiled and evaluated once (#383).

    The walk uses an explicit stack rather than recursion so a deep chain — a
    left-nested ``sum`` over thousands of terms — costs heap, not C stack (#925).
    """
    tape: list[tuple] = []
    slot_of: dict[int, int] = {}  # id(expr) -> tape slot, set on emission
    pending: dict[int, tuple] = {}  # id(expr) -> (children, kernel), set on first visit
    stack: list[tuple[Expression, bool]] = [(expr, False)]

    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if expanded:
            # Reached at most once per node: the ``(node, True)`` marker is pushed
            # only on the first visit, which ``pending`` below makes exclusive.
            children, kernel = pending[key]
            tape.append((kernel, tuple(slot_of[id(c)] for c in children)))
            slot_of[key] = len(tape) - 1
            continue
        if key in slot_of:
            continue  # already emitted through another parent
        if key in pending:
            # Visited but not yet emitted => this node is its own ancestor. The
            # modeling layer never builds one, and silently skipping would leave
            # a dangling child slot, so refuse loudly.
            raise ValueError(
                f"Cyclic expression graph at {type(node).__name__} node — "
                "expression DAGs must be acyclic."
            )
        children, kernel = _node_kernel(node, model, param_index)
        pending[key] = (children, kernel)
        stack.append((node, True))
        for child in reversed(children):
            stack.append((child, False))

    return tape


def _evaluate_tape(tape: list[tuple], x_flat, params):
    """Run a tape built by :func:`_build_tape` and return the root value.

    ``values[i]`` holds slot ``i``'s value for this call only; the loop is flat, so
    evaluation depth is independent of the expression's nesting depth (#925).
    """
    values: list = [None] * len(tape)
    for i, (kernel, child_slots) in enumerate(tape):
        values[i] = kernel(x_flat, params, [values[s] for s in child_slots])
    return values[-1]


def _node_kernel(expr: Expression, model: Model, param_index: dict) -> tuple[tuple, Callable]:
    """Build one node's ``(children, kernel)`` pair.

    ``children`` are the node's direct operand expressions, in evaluation order.
    ``kernel(x_flat, params, a)`` computes the node's value from ``a``, the list of
    already-computed child values aligned with ``children``. Kernels never call
    each other, which is what keeps evaluation iterative.
    """
    if isinstance(expr, Constant):
        val = jnp.array(expr.value)

        def fn(x_flat, params, a):
            return val

        return (), fn

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        size = expr.size
        shape = expr.shape
        if shape == () or (len(shape) == 1 and shape[0] == 1 and shape == ()):
            # Scalar variable: single slot
            def fn(x_flat, params, a):
                return x_flat[offset]

            return (), fn
        else:
            # Array variable: slice and reshape
            def fn(x_flat, params, a, _offset=offset, _size=size, _shape=shape):
                return x_flat[_offset : _offset + _size].reshape(_shape)

            return (), fn

    if isinstance(expr, Parameter):
        idx = param_index[id(expr)]

        def fn(x_flat, params, a, _i=idx):
            return params[_i]

        return (), fn

    if isinstance(expr, BinaryOp):
        op = expr.op
        if op == "+":

            def fn(x_flat, params, a):
                return a[0] + a[1]
        elif op == "-":

            def fn(x_flat, params, a):
                return a[0] - a[1]
        elif op == "*":

            def fn(x_flat, params, a):
                return a[0] * a[1]
        elif op == "/":

            def fn(x_flat, params, a):
                return a[0] / a[1]
        elif op == "**":

            def fn(x_flat, params, a):
                return a[0] ** a[1]
        else:
            raise ValueError(f"Unknown binary operator: {op!r}")
        return (expr.left, expr.right), fn

    if isinstance(expr, UnaryOp):
        op = expr.op
        if op == "neg":

            def fn(x_flat, params, a):
                return -a[0]
        elif op == "abs":

            def fn(x_flat, params, a):
                return jnp.abs(a[0])
        else:
            raise ValueError(f"Unknown unary operator: {op!r}")
        return (expr.operand,), fn

    if isinstance(expr, FunctionCall):
        args = tuple(expr.args)
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

            def fn(x_flat, params, a, _jax_fn=jax_fn):
                return _jax_fn(a[0])

            return args, fn

        if name == "min":

            def fn(x_flat, params, a):
                return jnp.minimum(a[0], a[1])

            return args, fn

        if name == "atan2":

            def fn(x_flat, params, a):
                return jnp.arctan2(a[0], a[1])

            return args, fn

        if name == "signpower":
            # GAMS signpower(x, a) = sign(x) * |x|**a.
            def fn(x_flat, params, a):
                xv = a[0]
                return jnp.sign(xv) * jnp.abs(xv) ** a[1]

            return args, fn

        if name == "centropy":
            # GAMS centropy(x, y) = x * log(x / y), with the x -> 0+ limit 0.
            def fn(x_flat, params, a):
                xv, yv = a[0], a[1]
                return xv * jnp.log(jnp.maximum(xv, 1e-300) / yv)

            return args, fn

        if name == "max":

            def fn(x_flat, params, a):
                return jnp.maximum(a[0], a[1])

            return args, fn

        if name == "prod":

            def fn(x_flat, params, a):
                return jnp.prod(a[0])

            return args, fn

        if name.startswith("norm"):
            # norm{p}: p-norm of an array argument (norm1, norm2, ...).
            suffix = name[len("norm") :]
            try:
                ord_p: float = (
                    float(suffix)
                    if suffix not in ("", "inf")
                    else (jnp.inf if suffix == "inf" else 2.0)
                )
            except ValueError as exc:
                raise ValueError(f"Unsupported norm order: {name!r}") from exc

            def fn(x_flat, params, a, _ord=ord_p):
                return jnp.linalg.norm(a[0], ord=_ord)

            return args, fn

        raise ValueError(f"Unknown function: {name!r}")

    if isinstance(expr, CustomCall):
        # Opaque AD-only user function: trace the stored callable through JAX so
        # value + autodiff gradients/Hessians come for free on the local NLP
        # path. No relaxation rule exists (see relaxation_compiler / solver
        # guards), so this branch is only reached on the continuous NLP path.
        user_fn = expr.fn

        def fn(x_flat, params, a, _user_fn=user_fn):
            return _user_fn(*a)

        return tuple(expr.args), fn

    if isinstance(expr, IndexExpression):
        idx = expr.index

        def fn(x_flat, params, a, _idx=idx):
            return a[0][_idx]

        return (expr.base,), fn

    if isinstance(expr, MatMulExpression):

        def fn(x_flat, params, a):
            return a[0] @ a[1]

        return (expr.left, expr.right), fn

    if isinstance(expr, SumExpression):
        axis = expr.axis

        def fn(x_flat, params, a, _axis=axis):
            return jnp.sum(a[0], axis=_axis)

        return (expr.operand,), fn

    if isinstance(expr, SumOverExpression):

        def fn(x_flat, params, a):
            result = a[0]
            for v in a[1:]:
                result = result + v
            return result

        return tuple(expr.terms), fn

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
    tape = _build_tape(expr, model, param_index)

    def fn(x_flat, params):
        # Each tape slot is computed once per call, so shared DAG nodes evaluate
        # once (#383) and depth costs heap rather than C stack (#925).
        return _evaluate_tape(tape, x_flat, params)

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
