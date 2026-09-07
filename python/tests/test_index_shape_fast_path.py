"""Differential test for the pure-integer fast path in ``_index_result_shape``.

The fast path replaces a ``np.broadcast_to`` probe on the construction hot path
(``x[i]`` on every element of every indexed family). It is a pure constant-factor
change, so the gate is *exact* agreement with the numpy oracle it replaces --
including the "unknown" (``None``) answer for indices numpy rejects.

Per CLAUDE.md §6 the equivalence test counts its executed comparisons and fails
if the grid ever degrades to zero, so a refactor that stops exercising the fast
path cannot pass silently.
"""

import itertools

import numpy as np
import pytest
from discopt.modeling.core import _index_result_shape, _pure_integer_index


def _numpy_oracle(base_shape, index):
    """The original implementation, verbatim -- the behavior being preserved."""
    try:
        probe = np.broadcast_to(np.zeros((), dtype=np.int8), base_shape)
        return tuple(int(d) for d in np.shape(probe[index]))
    except Exception:
        return None


SHAPES = [(1,), (5,), (4, 3), (2, 3, 4), (3, 1, 2)]


def _index_grid(ndim):
    """Indices spanning the fast path, its boundaries, and the fallback."""
    scalars = [0, 1, 2, -1, -3, 4, 5, 99, -99, np.int64(1), np.int32(-1), np.uint8(2)]
    # bools must route to the numpy mask semantics, never the integer fast path
    masks = [True, False, np.bool_(True)]
    exotic = [slice(None), slice(0, 2), Ellipsis, None, np.array([0, 1]), "nope", 1.5]
    out = list(scalars) + masks + exotic
    for k in range(1, ndim + 2):  # includes an over-long tuple (arity > ndim)
        for combo in itertools.product([0, 1, -1, 7], repeat=k):
            out.append(combo)
    out.append((0, slice(None)))
    out.append((True, 0))
    out.append((0, 1.5))
    return out


def test_fast_path_matches_numpy_oracle_exactly():
    compared = 0
    mismatches = []
    for base_shape in SHAPES:
        for index in _index_grid(len(base_shape)):
            got = _index_result_shape(base_shape, index)
            want = _numpy_oracle(base_shape, index)
            compared += 1
            if got != want:
                mismatches.append((base_shape, index, got, want))
    assert not mismatches, f"fast path diverged from numpy on: {mismatches[:10]}"
    # §6: prove the probe fired, and fired on a grid worth trusting.
    assert compared > 500, f"equivalence grid degraded to {compared} comparisons"


def test_fast_path_actually_engages_on_the_hot_case():
    """The whole point: `x[i]` on a 1-D base must not touch numpy."""
    assert _pure_integer_index(3) == (3,)
    assert _pure_integer_index((1, 2)) == (1, 2)
    assert _index_result_shape((5,), 3) == ()
    assert _index_result_shape((4, 3), (1, 2)) == ()
    assert _index_result_shape((4, 3), 1) == (3,)
    assert _index_result_shape((2, 3, 4), (1,)) == (3, 4)


@pytest.mark.parametrize("flag", [True, False, np.bool_(True), np.bool_(False)])
def test_bool_is_a_mask_not_an_index(flag):
    """`isinstance(True, int)` is True; numpy still treats it as a mask."""
    assert _pure_integer_index(flag) is None
    assert _index_result_shape((5,), flag) == _numpy_oracle((5,), flag)


def test_out_of_range_reports_unknown_not_a_guessed_shape():
    """Where numpy raises, both paths must answer "unknown" (None)."""
    for shape, idx in [((5,), 5), ((5,), -6), ((4, 3), (0, 3)), ((4, 3), (4, 0))]:
        assert _index_result_shape(shape, idx) is None
        assert _numpy_oracle(shape, idx) is None


def test_overlong_integer_tuple_falls_back():
    """Arity > ndim is numpy's error, not a shape the fast path may invent."""
    assert _index_result_shape((5,), (0, 0)) is None
    assert _numpy_oracle((5,), (0, 0)) is None
