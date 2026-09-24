"""Follow-ups from the review of PR #1302 (adversary batch 2, #1288-#1292).

Five defects found in the batch itself. Each test here fails on 60dfa403 and
passes after the fix; the two `_sum_along` cases are the ones that mattered,
because they put the tape and the layer that decides `sum_is_full_reduction`
into disagreement about how many rows a node stands for (the #1160 shape).
"""

import json
import math

import discopt.modeling as dm
import discopt.serialize as S
import numpy as np
import pytest
from discopt import Model
from discopt._nl_expr_compiler import UnsupportedForTape, _sum_along
from discopt._relax.scalarize import sum_result_shape
from discopt.modeling.core import SumExpression
from discopt.modeling.gams_parser import _gams_round
from discopt.result_io import serialize_result


def _object_tape_array(n):
    """``n`` constant tape nodes as an object array, the shape `_sum_along` takes."""
    import pounce

    arr = np.empty((n,), dtype=object)
    for i in range(n):
        arr[i] = pounce.NlExpr.const_(float(i))
    return arr, pounce.NlExpr


# ── finding 1: `axis=()` and repeated axes ────────────────────────────────


def test_empty_axis_tuple_reduces_nothing():
    """``np.sum(a, axis=())`` is ``a``; the tape must not fold it to a scalar.

    With ``k = 0``, ``shape[:-0]`` is empty and ``shape[-0:]`` is the whole
    shape, so the no-op compiled to one full-reduction node.
    """
    arr, E = _object_tape_array(3)
    assert _sum_along(arr, (), E).shape == (3,)


def test_empty_axis_tuple_agrees_with_sum_result_shape():
    """The tape and `scalarize` must report the same shape for the same node."""
    m = Model("t")
    x = m.continuous("x", shape=3, lb=0.0, ub=1.0)
    node = SumExpression(x, axis=())
    arr, E = _object_tape_array(3)

    assert sum_result_shape(node) == (3,)
    assert _sum_along(arr, (), E).shape == sum_result_shape(node)


@pytest.mark.parametrize("axis", [(0, 0), (0, -1)])
def test_repeated_axis_is_refused_by_both_layers(axis):
    """numpy refuses a repeated axis, so neither layer may invent a shape for it.

    `_sum_along` raised `ValueError` from `np.moveaxis`, which `try_compile` does
    not catch, and `sum_result_shape` folded the duplicate away through its set
    and returned the single-reduction shape.
    """
    m = Model("t")
    x = m.continuous("x", shape=3, lb=0.0, ub=1.0)
    arr, E = _object_tape_array(3)

    with pytest.raises(UnsupportedForTape, match="more than once"):
        _sum_along(arr, axis, E)
    assert sum_result_shape(SumExpression(x, axis=axis)) is None


def test_out_of_range_axis_is_refused_in_the_tapes_own_currency():
    arr, E = _object_tape_array(3)
    with pytest.raises(UnsupportedForTape, match="out of range"):
        _sum_along(arr, (5,), E)


def test_ordinary_axis_reductions_are_unchanged():
    """The fix must not move a reduction that already worked."""
    arr, E = _object_tape_array(3)
    assert _sum_along(arr, 0, E).shape == ()
    assert _sum_along(arr, None, E).shape == ()

    grid = np.empty((2, 3), dtype=object)
    flat, _ = _object_tape_array(6)
    grid.reshape(-1)[:] = flat
    assert _sum_along(grid, (0, 1), E).shape == ()
    assert _sum_along(grid, 1, E).shape == (2,)


# ── findings 2 and 5: the model schema minor, and the legacy solution ─────


def _solved_model():
    m = Model("t")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    return m, m.solve()


def test_schema_minor_is_bumped_for_the_non_additive_change():
    """1.1 promised additive minors; the bool index kind and list axis are not.

    1.2 was that bump. 1.3 adds the registered-atom tag (#1248 A), which *is*
    additive -- a reader that ignores the key gets the lowering and relaxes it
    term by term, which is sound -- but the minor still moves, because absence of
    the key has two meanings and only the minor tells them apart: "this model has
    no registered atoms" (1.3) versus "the writer could not record them" (≤1.2).
    A 1.2 reader that reloads and re-saves a 1.3 document drops the tags
    permanently, and the minor is what lets that be detected rather than guessed.
    """
    assert S.SCHEMA == "discopt.model/1.3"


def test_legacy_11_document_restores_a_tagged_float_in_the_trace():
    """A 1.1 `.dopt` holds a trace `nan` as the STRING "nan", and must decode it.

    Through 1.1 the solution subtree went through a blanket `_enc_tree`, which is
    what let a non-finite float in `mip_nlp_trace` -- a field `serialize_result`
    left raw -- past `allow_nan=False` at all. `deserialize_result` has no legacy
    path for that field, so without the minor gate the trace reloaded holding the
    string, and a re-save made it permanent.
    """
    m, r = _solved_model()
    raw_trace = {"iters": [{"obj": float("nan")}]}
    r.mip_nlp_trace = raw_trace

    solution = serialize_result(r)
    solution["schema_version"] = 2  # the version that wrote the trace raw
    solution["mip_nlp_trace"] = raw_trace
    legacy = json.loads(S.dumps(m, result=r))
    legacy["schema"] = "discopt.model/1.1"
    legacy["solution"] = S._enc_tree(solution)

    assert legacy["solution"]["mip_nlp_trace"] == {"iters": [{"obj": "nan"}]}

    back = S.loads(json.dumps(legacy))
    value = back.saved_result.mip_nlp_trace["iters"][0]["obj"]
    assert isinstance(value, float) and math.isnan(value)


def test_current_document_keeps_a_genuine_nan_string_a_string():
    """The #1292 fix itself: 1.2 has no blanket pass, so no ambiguity to resolve."""
    m, r = _solved_model()
    r.mip_nlp_trace = {"iters": [{"note": "nan"}]}

    back = S.loads(S.dumps(m, result=r))
    assert back.saved_result.mip_nlp_trace["iters"][0]["note"] == "nan"


def test_non_numeric_minor_is_refused_rather_than_guessed():
    """The minor picks a decoder, so an unreadable one cannot be shrugged off."""
    m, r = _solved_model()
    doc = json.loads(S.dumps(m, result=r))
    doc["schema"] = "discopt.model/1.x"
    with pytest.raises(S.SerializationError, match="non-numeric minor"):
        S.loads(json.dumps(doc))


# ── finding 3: `round` over a non-finite GAMS constant ────────────────────


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_gams_round_passes_non_finite_through(value):
    """`math.floor` raises on both; `round`, which it replaced, returned them."""
    out = _gams_round(value)
    assert math.isnan(out) if math.isnan(value) else out == value


def test_gams_round_survives_an_overflowing_scale():
    assert _gams_round(1e300, 30) == 1e300


def test_gams_round_still_rounds_half_away_from_zero():
    """The #1288 behaviour the fix must not disturb."""
    assert _gams_round(2.5) == 3.0
    assert _gams_round(-2.5) == -3.0
    assert _gams_round(1.2345, 2) == 1.23


# ── finding 4: the unknown-shape refusal names a spelling that works ──────


def test_custom_call_refusal_names_a_working_spelling():
    """A `CustomCall` is opaque by construction, so the generic advice misses it.

    The refusal is right -- an arbitrary callable has no static shape -- but it
    told the author to use shaped variables, which is not a workaround for one.
    """
    m = Model("t")
    x = m.continuous("x", lb=0.0, ub=1.0)
    call = dm.custom(lambda v: v * 2.0)(x)
    arr = np.array([x, x], dtype=object)

    with pytest.raises(TypeError, match="dm.custom"):
        arr * call

    # and the spelling the message names has to actually build a model
    out = np.array([element * call for element in arr], dtype=object)
    assert out.shape == (2,)
    assert all(isinstance(e, dm.Expression) for e in out)


def test_a_trailing_version_component_does_not_block_a_read():
    """This reader must not be what stops a later "1.3.1" from loading."""
    m, r = _solved_model()
    doc = json.loads(S.dumps(m, result=r))
    doc["schema"] = "discopt.model/1.3.1"
    assert S.loads(json.dumps(doc)) is not None
