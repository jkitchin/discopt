"""
Warm-start utilities for discopt.

Validates and flattens user-provided initial solutions so they can be
injected into the NLP solver (as starting point) and the B&B tree
(as initial incumbent / upper bound).
"""

from __future__ import annotations

import logging
import warnings
from typing import Union

import numpy as np

from discopt.modeling.core import Model, Variable, VarType

logger = logging.getLogger(__name__)


def validate_initial_solution(
    model: Model,
    initial_solution: dict[Variable, Union[float, np.ndarray, list]],
    *,
    tol_bounds: float = 1e-6,
    tol_integrality: float = 1e-5,
) -> np.ndarray:
    """Validate an initial solution and return it as a flat numpy vector.

    Parameters
    ----------
    model : Model
        The optimization model whose variables the solution refers to.
    initial_solution : dict
        Mapping from Variable objects to their values (scalars, lists, or
        numpy arrays).
    tol_bounds : float
        Tolerance for variable-bound violations.  Values outside
        ``[lb - tol, ub + tol]`` trigger a warning and are clamped.
    tol_integrality : float
        Tolerance for integrality violations on INTEGER / BINARY variables.
        Non-integer values trigger a warning and are rounded.

    Returns
    -------
    np.ndarray
        Flat solution vector (length = total number of scalar variables
        in the model) suitable for passing to the NLP evaluator.

    Raises
    ------
    TypeError
        If *initial_solution* is not a dict or contains non-Variable keys.
    ValueError
        If a Variable does not belong to the model, or a value has the
        wrong shape or non-finite entries.
    """
    if not isinstance(initial_solution, dict):
        raise TypeError(
            "initial_solution must be a dict mapping Variable objects to values, "
            f"got {type(initial_solution).__name__}"
        )

    # Build a set of model variables for membership checks
    model_vars = {id(v): v for v in model._variables}

    # Validate keys
    for key in initial_solution:
        if not isinstance(key, Variable):
            raise TypeError(
                "initial_solution keys must be Variable objects, "
                f"got {type(key).__name__} for key {key!r}"
            )
        if id(key) not in model_vars:
            raise ValueError(
                f"Variable '{key.name}' is not part of this model. "
                "Use the Variable objects returned by m.continuous(), "
                "m.binary(), or m.integer()."
            )

    # Build flat vector: start from midpoint of bounds, then overlay
    # provided values.
    n_vars = sum(v.size for v in model._variables)
    x_flat = np.zeros(n_vars, dtype=np.float64)
    offset = 0
    provided_mask = np.zeros(n_vars, dtype=bool)

    for v in model._variables:
        size = v.size
        if v in initial_solution:
            val = np.asarray(initial_solution[v], dtype=np.float64)

            # Shape validation
            expected_shape = v.shape
            if expected_shape == ():
                # Scalar variable: accept scalar or (1,) array
                if val.shape not in ((), (1,)):
                    raise ValueError(
                        f"Variable '{v.name}' is scalar but got value with shape {val.shape}"
                    )
                val = val.flatten()
            else:
                if val.shape != expected_shape:
                    # Try to reshape from flat
                    try:
                        val = val.reshape(expected_shape)
                    except ValueError:
                        raise ValueError(
                            f"Variable '{v.name}' has shape {expected_shape} "
                            f"but got value with shape {val.shape}"
                        )
                val = val.flatten()

            # Bounds checking and clamping
            lb_flat = v.lb.flatten()
            ub_flat = v.ub.flatten()

            if not np.all(np.isfinite(val)):
                raise ValueError(
                    f"Variable '{v.name}' initial_solution contains non-finite value(s)"
                )

            below = val < lb_flat - tol_bounds
            above = val > ub_flat + tol_bounds
            if np.any(below) or np.any(above):
                n_viol = int(np.sum(below) + np.sum(above))
                warnings.warn(
                    f"Variable '{v.name}': {n_viol} value(s) outside bounds. Clamping to [lb, ub].",
                    stacklevel=3,
                )
                val = np.clip(val, lb_flat, ub_flat)

            # Integrality checking and rounding
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                frac = np.abs(val - np.round(val))
                if np.any(frac > tol_integrality):
                    n_frac = int(np.sum(frac > tol_integrality))
                    warnings.warn(
                        f"Variable '{v.name}': {n_frac} value(s) are not "
                        "integer-valued. Rounding to nearest integer.",
                        stacklevel=3,
                    )
                    val = np.round(val)
                    val = np.clip(val, lb_flat, ub_flat)

            x_flat[offset : offset + size] = val
            provided_mask[offset : offset + size] = True
        else:
            # Default: midpoint of bounds (clipped for unbounded vars)
            lb_flat = v.lb.flatten()
            ub_flat = v.ub.flatten()
            lb_clip = np.clip(lb_flat, -1e4, 1e4)
            ub_clip = np.clip(ub_flat, -1e4, 1e4)
            x_flat[offset : offset + size] = 0.5 * (lb_clip + ub_clip)

        offset += size

    n_provided = int(np.sum(provided_mask))
    n_total_vars = len(model._variables)
    if n_provided > 0 and n_provided < n_vars:
        logger.info(
            "Warm start: %d of %d variables provided, using midpoint for the rest",
            len(initial_solution),
            n_total_vars,
        )

    return x_flat


def check_feasibility(
    model: Model,
    x_flat: np.ndarray,
    *,
    tol: float = 1e-4,
) -> tuple[bool, list[str]]:
    """Check whether a flat solution vector is feasible for the model.

    Returns a tuple ``(is_feasible, violations)`` where *violations* is a
    list of human-readable strings describing any constraint or bound
    violations found.
    """
    violations: list[str] = []

    # Variable bounds
    offset = 0
    for v in model._variables:
        size = v.size
        vals = x_flat[offset : offset + size]
        lb_flat = v.lb.flatten()
        ub_flat = v.ub.flatten()

        below = vals < lb_flat - tol
        above = vals > ub_flat + tol
        if np.any(below):
            violations.append(
                f"Variable '{v.name}': {int(np.sum(below))} value(s) below lower bound"
            )
        if np.any(above):
            violations.append(
                f"Variable '{v.name}': {int(np.sum(above))} value(s) above upper bound"
            )
        offset += size

    # Constraint feasibility (requires evaluator)
    try:
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.modeling.core import Constraint

        evaluator = NLPEvaluator(model)
        if evaluator.n_constraints > 0:
            cons = evaluator.evaluate_constraints(x_flat)
            idx = 0
            for c in model._constraints:
                if not isinstance(c, Constraint):
                    continue
                val = cons[idx]
                if c.sense == "<=":
                    if val > tol:
                        name = c.name or f"constraint_{idx}"
                        violations.append(f"Constraint '{name}': value {val:.6g} > 0 (sense <=)")
                elif c.sense == "==":
                    if abs(val) > tol:
                        name = c.name or f"constraint_{idx}"
                        violations.append(
                            f"Constraint '{name}': |value| = {abs(val):.6g} != 0 (sense ==)"
                        )
                elif c.sense == ">=":
                    if val < -tol:
                        name = c.name or f"constraint_{idx}"
                        violations.append(f"Constraint '{name}': value {val:.6g} < 0 (sense >=)")
                idx += 1
    except Exception as e:
        logger.debug("Feasibility check skipped (evaluator error): %s", e)

    return len(violations) == 0, violations
