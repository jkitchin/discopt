"""Issue #941: a scalar variable reference must resolve to the *correct* flat slot.

A negative index (``v[-1]``) used to resolve to ``base_offset - 1`` — a valid slot
belonging to a different variable. Nothing downstream errors on a wrong-but-
in-range slot, so the McCormick relaxation was built for a bilinear pair that does
not exist in the model, cutting off the true optimum and reporting it as
``optimal`` with ``gap_certified=True``.

The oracle here is numpy's own indexing, not a reimplementation of it: build
``np.arange(size).reshape(shape)`` and read ``probe[idx]``. Whatever numpy says
that element's flat position is, that is the truth — negatives, mixed spellings
and all. (This allocation is fine in a test; it is exactly what the production
resolver must *not* do per-leaf, for the O(size·leaves) reason issue #654 fixed.)
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._flat_index import flat_index_in_shape, normalize_axis_index, resolve_scalar_slot
from discopt.modeling.core import IndexExpression


@pytest.fixture
def model_with_shapes():
    """A model whose variables sit at known, non-zero flat offsets.

    The leading scalar is what makes this test able to fail: with ``v`` at offset
    0 a dropped base offset is invisible, and with ``s`` at slot 0 a negative
    index that under-shoots lands on ``s`` — exactly the #941 aliasing.
    """
    m = dm.Model("t941")
    s = m.continuous("s", lb=0.0, ub=1.0)  # slot 0
    v = m.continuous("v", shape=(4,), lb=0.0, ub=1.0)  # slots 1-4
    w = m.continuous("w", shape=(2, 3), lb=0.0, ub=1.0)  # slots 5-10
    u = m.continuous("u", shape=(2, 3, 4), lb=0.0, ub=1.0)  # slots 11-34
    return m, {"s": s, "v": v, "w": w, "u": u}


def test_every_index_of_every_shaped_variable_matches_numpy(model_with_shapes):
    """Exhaustive over all in-range indices, positive AND negative, for each shape."""
    m, vs = model_with_shapes
    offsets = {"v": 1, "w": 5, "u": 11}

    checked = 0
    for name in ("v", "w", "u"):
        var = vs[name]
        shape = tuple(var.shape)
        off = offsets[name]
        assert m._flat_var_offset(var) == off, f"{name} moved; the fixture's map is stale"
        probe = np.arange(int(np.prod(shape))).reshape(shape)

        # Each axis contributes both its positive and its negative spelling, so
        # mixed forms like w[0, -1] and u[-2, 1, -3] are covered too.
        per_axis = [list(range(d)) + list(range(-d, 0)) for d in shape]
        for idx in itertools.product(*per_axis):
            truth = off + int(probe[idx])
            key = idx if len(idx) > 1 else idx[0]
            got = resolve_scalar_slot(IndexExpression(var, key), m)
            assert got == truth, f"{name}{list(idx)}: got slot {got}, numpy says {truth}"
            checked += 1

    # 8 (v) + 24 (w) + 192 (u) = 224. Anti-vacuity per CLAUDE.md §6.
    assert checked == 224, f"expected 224 resolutions, made {checked}"


def test_the_exact_aliasing_from_the_issue(model_with_shapes):
    """`v[-1]` must not resolve to `s`'s slot. This is the reported failure."""
    m, vs = model_with_shapes
    assert resolve_scalar_slot(IndexExpression(vs["v"], -1), m) == 4  # not 0 (`s`)
    assert resolve_scalar_slot(IndexExpression(vs["v"], -4), m) == 1  # not -3
    assert resolve_scalar_slot(IndexExpression(vs["w"], (-1, -1)), m) == 10  # not 1
    assert resolve_scalar_slot(IndexExpression(vs["w"], (0, -1)), m) == 7  # not 4


@pytest.mark.parametrize(
    "label,index",
    [
        ("slice", slice(0, 2)),
        ("slice_all", slice(None)),
        ("partial_2d", (0,)),
        ("slice_in_tuple", (0, slice(None))),
        ("ellipsis", Ellipsis),
        ("float", 1.0),
        ("string", "a"),
        ("none", None),
        ("too_many", (0, 0, 0)),
    ],
)
def test_non_scalar_index_forms_return_none(model_with_shapes, label, index):
    """Unresolvable is `None` — the sound answer, never a guessed slot."""
    m, vs = model_with_shapes
    base = (
        vs["v"]
        if label in ("slice", "slice_all", "ellipsis", "float", "string", "none")
        else vs["w"]
    )
    assert resolve_scalar_slot(IndexExpression(base, index), m) is None


@pytest.mark.parametrize("index", [4, 7, -5, -100])
def test_out_of_range_is_refused_not_clamped(model_with_shapes, index):
    """numpy raises where jnp clamps; resolving here would pick a side."""
    m, vs = model_with_shapes
    assert resolve_scalar_slot(IndexExpression(vs["v"], index), m) is None


def test_bool_is_not_an_integer_index(model_with_shapes):
    """`bool` subclasses `int`, but numpy reads `x[True]` as a mask, not `x[1]`."""
    m, vs = model_with_shapes
    assert resolve_scalar_slot(IndexExpression(vs["v"], True), m) is None
    assert resolve_scalar_slot(IndexExpression(vs["v"], False), m) is None
    assert resolve_scalar_slot(IndexExpression(vs["w"], (True, 0)), m) is None


def test_numpy_integers_are_accepted(model_with_shapes):
    """`isinstance(np.int64(2), int)` is False, so an `int`-only check drops these."""
    m, vs = model_with_shapes
    assert resolve_scalar_slot(IndexExpression(vs["v"], np.int64(2)), m) == 3
    assert resolve_scalar_slot(IndexExpression(vs["v"], np.int32(-1)), m) == 4
    assert resolve_scalar_slot(IndexExpression(vs["w"], (np.int64(-1), np.int8(2))), m) == 10


def test_scalar_variable_and_non_variable_base(model_with_shapes):
    m, vs = model_with_shapes
    assert resolve_scalar_slot(vs["s"], m) == 0
    assert resolve_scalar_slot(vs["v"], m) is None  # array, not one slot
    # Base is an expression, not a Variable: no static slot exists.
    assert resolve_scalar_slot(IndexExpression(vs["v"] + 1.0, 0), m) is None


def test_flat_index_in_shape_is_c_order():
    """Row-major, matching dag_compiler's `x_flat[off:off+size].reshape(shape)`."""
    checked = 0
    for shape in ((4,), (2, 3), (2, 3, 4)):
        probe = np.arange(int(np.prod(shape))).reshape(shape)
        for idx in itertools.product(*[range(-d, d) for d in shape]):
            key = idx if len(idx) > 1 else idx[0]
            assert flat_index_in_shape(key, shape) == int(probe[idx])
            checked += 1
    assert checked == 8 + 24 + 192


def test_normalize_axis_index():
    assert normalize_axis_index(0, 4) == 0
    assert normalize_axis_index(3, 4) == 3
    assert normalize_axis_index(-1, 4) == 3
    assert normalize_axis_index(-4, 4) == 0
    assert normalize_axis_index(4, 4) is None
    assert normalize_axis_index(-5, 4) is None
    assert normalize_axis_index(True, 4) is None
    assert normalize_axis_index(1.0, 4) is None
