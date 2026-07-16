"""Small helpers shared across JAX-backed solver modules."""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import Model, VarType


def flat_variable_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened lower and upper bounds for all model variables."""
    lbs: list[float] = []
    ubs: list[float] = []
    for v in model._variables:
        lbs.extend(np.asarray(v.lb, dtype=np.float64).ravel().tolist())
        ubs.extend(np.asarray(v.ub, dtype=np.float64).ravel().tolist())
    return np.array(lbs, dtype=np.float64), np.array(ubs, dtype=np.float64)


def binary_flat_cols(model: Model) -> frozenset[int]:
    """Flat column indices of the model's **binary** variables — integer-typed with
    bounds ``[0, 1]``.

    A binary variable satisfies ``x**2 == x`` at every feasible (integer) point, so
    the moment-matrix diagonal ``X_ii`` equals ``x_i`` there: the PSD/moment-cut
    separator can use the original ``x_i`` column as ``X_ii`` for these variables
    even when no explicit lifted square column exists (the common case for pure
    products of distinct binaries, e.g. QAP). Only binary variables qualify — for a
    continuous ``x_i`` in ``[0,1]`` the relation ``X_ii == x_i`` is false and the
    substitution would be unsound.
    """
    lb, ub = flat_variable_bounds(model)
    is_int: list[np.ndarray] = []
    for v in model._variables:
        integral = v.var_type in (VarType.BINARY, VarType.INTEGER)
        is_int.append(np.full(int(v.size), integral, dtype=bool))
    int_mask = np.concatenate(is_int) if is_int else np.zeros(0, dtype=bool)
    binary = int_mask & np.isclose(lb, 0.0) & np.isclose(ub, 1.0)
    return frozenset(int(i) for i in np.flatnonzero(binary))
