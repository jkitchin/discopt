"""Parametric response compiler for model-based DoE and parameter estimation.

Compiles arbitrary model expressions into JAX-differentiable functions of
``(x_flat, p_flat)``, where ``x_flat`` is the concatenated variable vector
and ``p_flat`` is the concatenated parameter vector.

The key function :func:`compile_response_function` returns a function that
maps ``(x_flat, p_flat)`` to a stacked vector of response values. This
enables exact Jacobian computation via ``jax.jacobian(fn, argnums=1)``
for Fisher Information Matrix calculation.

Example
-------
>>> import jax
>>> fn = compile_response_function({"y1": expr1, "y2": expr2}, model)
>>> J = jax.jacobian(fn, argnums=1)(x_flat, p_flat)  # (2, n_params)
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from discopt._jax.differentiable import (
    _compile_parametric_node,
    _flatten_params,
    _param_total_size,
)
from discopt.modeling.core import Expression, Model


def compile_response_function(
    responses: dict[str, Expression],
    model: Model,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compile named expressions into a JAX-differentiable response function.

    Parameters
    ----------
    responses : dict[str, Expression]
        Named response expressions from the model. Each expression should
        evaluate to a scalar at a given ``(x, p)`` point.
    model : Model
        The discopt model containing the variables and parameters.

    Returns
    -------
    callable
        Function ``f(x_flat, p_flat) -> jnp.ndarray`` returning a 1-D array
        of response values in the same order as ``responses``.

    Raises
    ------
    ValueError
        If ``responses`` is empty.
    """
    if not responses:
        raise ValueError("responses must be non-empty")

    names = list(responses.keys())
    compiled_fns = [_compile_parametric_node(responses[n], model) for n in names]

    def response_fn(x_flat: jnp.ndarray, p_flat: jnp.ndarray) -> jnp.ndarray:
        vals = [fn(x_flat, p_flat) for fn in compiled_fns]
        return jnp.stack(vals)

    # Attach metadata for introspection
    response_fn.response_names = names  # type: ignore[attr-defined]
    response_fn.n_responses = len(names)  # type: ignore[attr-defined]

    return response_fn


def flatten_params(model: Model) -> jnp.ndarray:
    """Concatenate all model parameter values into a flat JAX array.

    Convenience wrapper around the internal ``_flatten_params``.

    Parameters
    ----------
    model : Model
        The discopt model.

    Returns
    -------
    jnp.ndarray
        1-D array of all parameter values.
    """
    return _flatten_params(model)


def param_total_size(model: Model) -> int:
    """Total number of scalar parameter values in the model.

    Parameters
    ----------
    model : Model
        The discopt model.

    Returns
    -------
    int
        Total scalar parameter count.
    """
    return _param_total_size(model)


def variable_total_size(model: Model) -> int:
    """Total number of scalar variable values in the model.

    Parameters
    ----------
    model : Model
        The discopt model.

    Returns
    -------
    int
        Total scalar variable count.
    """
    return sum(v.size for v in model._variables)


def extract_x_flat(result, model: Model) -> jnp.ndarray:
    """Extract the flat variable vector from a SolveResult.

    Parameters
    ----------
    result : SolveResult
        A solved result with ``result.x`` populated.
    model : Model
        The discopt model.

    Returns
    -------
    jnp.ndarray
        1-D array of all variable values at the solution.

    Raises
    ------
    ValueError
        If the result has no solution.
    """
    if result.x is None:
        raise ValueError("No solution available in result")

    parts = []
    for v in model._variables:
        val = np.asarray(result.x[v.name], dtype=np.float64).ravel()
        parts.append(val)
    return jnp.array(np.concatenate(parts), dtype=jnp.float64)
