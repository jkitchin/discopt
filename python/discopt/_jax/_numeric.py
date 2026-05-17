"""Numeric helper predicates shared by JAX solver components."""

from __future__ import annotations

import numpy as np

EFFECTIVE_INF = 1e19


def is_effectively_finite(value: float) -> bool:
    """Return True when a bound is finite in the solver sense."""
    return bool(np.isfinite(value) and abs(float(value)) < EFFECTIVE_INF)
