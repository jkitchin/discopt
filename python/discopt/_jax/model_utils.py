"""Small helpers shared across JAX-backed solver modules."""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import Model


def flat_variable_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened lower and upper bounds for all model variables."""
    lbs: list[float] = []
    ubs: list[float] = []
    for v in model._variables:
        lbs.extend(np.asarray(v.lb, dtype=np.float64).ravel().tolist())
        ubs.extend(np.asarray(v.ub, dtype=np.float64).ravel().tolist())
    return np.array(lbs, dtype=np.float64), np.array(ubs, dtype=np.float64)
