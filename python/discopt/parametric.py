"""Public parametric-compilation API.

Stable contract for code built on top of discopt — external plugins
(``discopt-doe``, ``discopt-mkm``, ...) and in-tree consumers such as
:mod:`discopt.estimate` — to compile model :class:`~discopt.modeling.core.Expression`
objects into JAX-differentiable functions of ``(x_flat, p_flat)``.

The flat-vector contract
------------------------
``x_flat`` concatenates the values of every model :class:`Variable` in
declaration order, each flattened in C order; ``p_flat`` does the same for
every model :class:`Parameter`. :func:`variable_slices` exposes the layout of
``x_flat``; :func:`extract_x_flat` and :func:`flatten_params` build the two
vectors from a solved result and the model's nominal parameter values.

All functions import JAX lazily, so ``import discopt.parametric`` does not pull
``jax`` (or any other heavy dependency) into the process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import jax.numpy as jnp

    from discopt.modeling.core import Expression, Model

__all__ = [
    "compile_expression",
    "compile_response_function",
    "extract_x_flat",
    "flatten_params",
    "param_total_size",
    "variable_slices",
    "variable_total_size",
]


def compile_expression(expr: Expression, model: Model) -> Callable:
    """Compile a single expression into ``f(x_flat, p_flat) -> value``.

    The returned function is a pure JAX computation: it can be transformed
    with ``jax.jit``, ``jax.grad``, ``jax.jacobian``, ``jax.vmap``, etc.
    Unlike the standard DAG compiler, Parameter values are *not* baked in as
    constants — they are read from ``p_flat``, so derivatives with respect to
    parameters are available via ``argnums=1``.

    Parameters
    ----------
    expr : Expression
        The model expression to compile.
    model : Model
        The discopt model that owns the expression's variables and parameters
        (defines the ``x_flat``/``p_flat`` layout).

    Returns
    -------
    callable
        Function ``f(x_flat, p_flat) -> jnp.ndarray`` evaluating the
        expression at the given flat vectors.
    """
    from discopt._jax.differentiable import _compile_parametric_node

    fn: Callable = _compile_parametric_node(expr, model)
    return fn


def compile_response_function(
    responses: dict[str, Expression],
    model: Model,
) -> Callable:
    """Compile named expressions into a stacked JAX-differentiable function.

    Parameters
    ----------
    responses : dict[str, Expression]
        Named response expressions. Each should evaluate to a scalar.
    model : Model
        The discopt model containing the variables and parameters.

    Returns
    -------
    callable
        Function ``f(x_flat, p_flat) -> jnp.ndarray`` returning a 1-D array of
        response values in ``responses`` order, with ``response_names`` and
        ``n_responses`` attributes attached.

    Raises
    ------
    ValueError
        If ``responses`` is empty.
    """
    from discopt._jax.parametric import compile_response_function as _impl

    return _impl(responses, model)


def extract_x_flat(result, model: Model) -> jnp.ndarray:
    """Extract the flat variable vector from a solved result.

    Parameters
    ----------
    result : SolveResult
        A solved result with ``result.x`` populated.
    model : Model
        The discopt model (defines the ``x_flat`` layout).

    Returns
    -------
    jnp.ndarray
        1-D array of all variable values at the solution.

    Raises
    ------
    ValueError
        If the result has no solution.
    """
    from discopt._jax.parametric import extract_x_flat as _impl

    return _impl(result, model)


def flatten_params(model: Model) -> jnp.ndarray:
    """Concatenate all model Parameter values into ``p_flat``.

    Returns a zero-length array when the model has no Parameters, so the
    result is always a valid second argument for compiled functions.

    Parameters
    ----------
    model : Model
        The discopt model.

    Returns
    -------
    jnp.ndarray
        1-D array of all parameter values in declaration order.
    """
    from discopt._jax.differentiable import _flatten_params

    return _flatten_params(model)


def param_total_size(model: Model) -> int:
    """Total number of scalar parameter values in the model."""
    from discopt._jax.differentiable import _param_total_size

    return _param_total_size(model)


def variable_total_size(model: Model) -> int:
    """Total number of scalar variable values in the model."""
    return sum(v.size for v in model._variables)


def variable_slices(model: Model) -> dict[str, slice]:
    """Map each variable's name to its slice of ``x_flat``.

    Variable names are unique within a model (they key ``result.x``), so this
    fully describes the ``x_flat`` layout: declaration order, each variable
    flattened C-order to ``v.size`` entries.

    Parameters
    ----------
    model : Model
        The discopt model.

    Returns
    -------
    dict[str, slice]
        ``{variable_name: slice(start, stop)}`` into ``x_flat``.
    """
    out: dict[str, slice] = {}
    offset = 0
    for v in model._variables:
        out[v.name] = slice(offset, offset + v.size)
        offset += v.size
    return out
