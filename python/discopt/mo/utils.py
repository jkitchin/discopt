"""Utilities for multi-objective optimization.

Provides:

* :func:`ideal_point` -- per-objective best values, computed by ``k`` single-
  objective solves.
* :func:`nadir_point` -- payoff-table estimate of the worst Pareto value per
  objective.
* :func:`evaluate_expression` -- evaluate a discopt ``Expression`` at a
  solution dict, used to cross-evaluate objectives at anchor points.
* :func:`normalize_objectives` -- affine ideal/nadir normalization for
  scalarizations and indicators.

Sense handling conventions
--------------------------

Objectives are passed to scalarizers in their natural sense (an Expression)
plus a parallel ``senses`` list of ``"min"`` / ``"max"`` strings. All internal
math converts to an all-minimize convention by negating maximize objectives,
but results stored in :class:`~discopt.mo.pareto.ParetoFront` retain original
senses.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _as_senses(
    senses: Optional[Iterable[str]],
    k: int,
) -> list[str]:
    """Normalize the ``senses`` argument into a length-``k`` list of strings."""
    if senses is None:
        return ["min"] * k
    out = list(senses)
    if len(out) != k:
        raise ValueError(f"senses has length {len(out)}, expected {k}")
    for s in out:
        if s not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max', got {s!r}")
    return out


def _flatten_solution(model, x_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate per-variable arrays into the flat layout used by dag_compiler."""
    parts: list[np.ndarray] = []
    for v in model._variables:
        if v.name not in x_dict:
            raise KeyError(
                f"Solution dict missing variable {v.name!r}; available keys: {sorted(x_dict)}"
            )
        arr = np.asarray(x_dict[v.name], dtype=np.float64).reshape(-1)
        if arr.size != v.size:
            raise ValueError(f"Variable {v.name} expected {v.size} values, got {arr.size}")
        parts.append(arr)
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def evaluate_expression(expr, model, x_dict: dict[str, np.ndarray]) -> float:
    """Evaluate a scalar discopt Expression at a given solution.

    Compiles ``expr`` through the JAX DAG compiler using the model's current
    parameter values, then calls it on the flat solution vector.
    """
    from discopt._jax.dag_compiler import compile_expression

    fn = compile_expression(expr, model)
    x_flat = _flatten_solution(model, x_dict)
    return float(np.asarray(fn(x_flat)))


def ideal_point(
    model,
    objectives: list,
    *,
    senses: Optional[Iterable[str]] = None,
    warm_start: bool = True,
    **solve_kwargs,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Compute the ideal point and per-objective anchor solutions.

    For each objective ``f_i`` in order, sets the model objective to
    ``minimize f_i`` (or ``maximize f_i``) and solves, restoring the original
    model objective at the end. Returns the vector of per-objective best
    values (in original senses) and the list of anchor solution dicts.

    Parameters
    ----------
    model : discopt.modeling.Model
        Model carrying the decision variables and constraints. Any existing
        objective is restored before return.
    objectives : list of Expression
        Length-``k`` list of objective expressions.
    senses : iterable of {"min", "max"}, optional
        One sense per objective; default is all ``"min"``.
    warm_start : bool, default True
        Pass each anchor's solution as ``initial_solution`` to the next anchor
        solve.
    **solve_kwargs
        Forwarded to :meth:`discopt.modeling.Model.solve`.

    Returns
    -------
    ideal : numpy.ndarray
        Shape ``(k,)`` array of best per-objective values.
    anchors : list of dict
        ``anchors[i]`` is the variable-value dict at the optimum of ``f_i``.

    Raises
    ------
    RuntimeError
        If any of the ``k`` single-objective solves fails to return a
        feasible solution.
    """
    senses_list = _as_senses(senses, len(objectives))
    saved_objective = model._objective
    ideal = np.zeros(len(objectives), dtype=np.float64)
    anchors: list[dict[str, np.ndarray]] = []
    last_x: Optional[dict[str, np.ndarray]] = None

    try:
        for i, (expr, sense) in enumerate(zip(objectives, senses_list)):
            if sense == "min":
                model.minimize(expr)
            else:
                model.maximize(expr)

            kwargs = dict(solve_kwargs)
            if warm_start and last_x is not None and "initial_solution" not in kwargs:
                kwargs["initial_solution"] = _x_to_var_dict(model, last_x)

            result = model.solve(**kwargs)
            if result.x is None or result.objective is None:
                raise RuntimeError(
                    f"ideal-point solve for objective {i} failed: status={result.status}"
                )
            ideal[i] = float(result.objective)
            anchors.append({k: np.asarray(v).copy() for k, v in result.x.items()})
            last_x = result.x
    finally:
        model._objective = saved_objective

    return ideal, anchors


def _x_to_var_dict(model, x_dict: dict[str, np.ndarray]) -> dict:
    """Convert a name-keyed solution dict to the Variable-keyed form that
    :meth:`Model.solve` accepts as ``initial_solution``."""
    name_to_var = {v.name: v for v in model._variables}
    return {name_to_var[n]: x_dict[n] for n in x_dict if n in name_to_var}


def nadir_point(
    model,
    objectives: list,
    anchors: list[dict[str, np.ndarray]],
    *,
    senses: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Payoff-table estimate of the nadir point.

    Evaluates every objective ``f_j`` at every anchor solution ``x^{(i)}*``
    and takes, for each ``j``, the worst value across anchors (max for
    minimization, min for maximization). This is the standard approximation
    for ``k = 2`` and a (potentially loose) heuristic for ``k >= 3``
    [Miettinen 1999].

    Parameters
    ----------
    model : discopt.modeling.Model
    objectives : list of Expression
    anchors : list of dict
        Solution dicts returned by :func:`ideal_point`, length ``k``.
    senses : iterable of {"min", "max"}, optional

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)`` nadir estimate in original senses.
    """
    senses_list = _as_senses(senses, len(objectives))
    k = len(objectives)
    if len(anchors) != k:
        raise ValueError(f"Expected {k} anchors, got {len(anchors)}")

    payoff = np.zeros((k, k), dtype=np.float64)  # payoff[i, j] = f_j at anchor i
    for i in range(k):
        for j in range(k):
            payoff[i, j] = evaluate_expression(objectives[j], model, anchors[i])

    nadir = np.zeros(k, dtype=np.float64)
    for j in range(k):
        col = payoff[:, j]
        nadir[j] = float(col.max() if senses_list[j] == "min" else col.min())
    return nadir


def normalize_objectives(
    objectives: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    *,
    senses: Optional[Iterable[str]] = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Affine normalize objectives into ``[0, 1]`` using the ideal/nadir range.

    Converts each objective so that 0 corresponds to the ideal and 1 to the
    nadir, regardless of sense. The output is always oriented so that smaller
    is better.

    Parameters
    ----------
    objectives : numpy.ndarray
        Array of shape ``(n, k)`` or ``(k,)``.
    ideal, nadir : numpy.ndarray
        Length-``k`` reference vectors in original senses.
    senses : iterable of {"min", "max"}, optional
        Defaults to all ``"min"``.
    epsilon : float
        Small floor on the denominator to avoid division by zero when ideal
        and nadir coincide.

    Returns
    -------
    numpy.ndarray
        Normalized array with the same shape as *objectives*.
    """
    objectives = np.asarray(objectives, dtype=np.float64)
    k = objectives.shape[-1]
    senses_list = _as_senses(senses, k)
    signs = np.array([1.0 if s == "min" else -1.0 for s in senses_list], dtype=np.float64)
    span = signs * (nadir - ideal)
    span = np.where(np.abs(span) < epsilon, epsilon, span)
    return np.asarray(signs * (objectives - ideal) / span, dtype=np.float64)
