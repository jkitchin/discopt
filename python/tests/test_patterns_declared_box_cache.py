"""Per-model declared-box cache for the syntactic convexity recognizers.

``convexity.patterns._box_bounds`` / ``_total_scalar_variables`` are pure functions
of the model's DECLARED variable bounds, yet the recognizers call them once per
objective term (66k times on a 21k-term quadratic). Memoizing them on the model —
safe because declared ``Variable`` bounds are immutable ``np.broadcast_to`` views
for a model's life, with tightening flowing through node boxes, never mutation —
removes ~10s of pure recomputation. This test pins the cache's correctness and its
explicit invalidation.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt._jax.convexity.patterns import (
    _DECLARED_BOX_CACHE_ATTR,
    _box_bounds,
    _total_scalar_variables,
    clear_declared_box_cache,
)


@pytest.mark.unit
def test_box_bounds_values_correct_and_cached():
    m = Model()
    m.continuous("x", lb=-1.0, ub=2.0)
    m.continuous("y", lb=0.0, ub=5.0)

    lo, hi = _box_bounds(m)
    assert np.allclose(lo, [-1.0, 0.0])
    assert np.allclose(hi, [2.0, 5.0])
    assert _total_scalar_variables(m) == 2

    # Second call returns the SAME cached arrays (identity), proving no rebuild.
    lo2, hi2 = _box_bounds(m)
    assert lo2 is lo and hi2 is hi
    assert hasattr(m, _DECLARED_BOX_CACHE_ATTR)


@pytest.mark.unit
def test_clear_declared_box_cache_invalidates():
    m = Model()
    m.continuous("x", lb=0.0, ub=1.0)
    lo, _ = _box_bounds(m)
    assert hasattr(m, _DECLARED_BOX_CACHE_ATTR)

    clear_declared_box_cache(m)
    assert not hasattr(m, _DECLARED_BOX_CACHE_ATTR)

    # Recomputes fresh (equal values, new object) after invalidation.
    lo2, _ = _box_bounds(m)
    assert lo2 is not lo
    assert np.allclose(lo2, lo)

    # Idempotent / safe on an un-cached model.
    clear_declared_box_cache(m)
    clear_declared_box_cache(Model())


@pytest.mark.unit
def test_total_scalar_variables_matches_box_size_for_array_vars():
    """The cached scalar count must equal the flattened box length, including
    array variables (size > 1)."""
    m = Model()
    m.continuous("s", lb=0.0, ub=1.0)
    m.continuous("v", lb=np.zeros(4), ub=np.ones(4), shape=(4,))
    lo, hi = _box_bounds(m)
    assert lo.size == hi.size == 5
    assert _total_scalar_variables(m) == 5
