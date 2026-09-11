"""Differential tests for the two #1215 construction fast paths.

Both are pure constant-factor changes on the model-construction hot path, so the
gate is *exact* agreement with the general route they bypass -- same node, same
cached shape, same exception type and message.

1. ``Variable.__getitem__`` answers a plain ``int`` into a 1-D variable directly
   (range test + ``IndexExpression(..., shape_hint=())``) instead of routing
   through ``Expression.__getitem__`` -> ``_known_shape`` -> ``_integer_index_out_of_range``
   -> ``IndexExpression`` -> ``_known_shape`` -> ``_index_result_shape``.
2. ``IndexedVar.__getitem__`` memoises ``key -> node`` instead of re-running
   ``Set.ordinal`` on every access.

Per CLAUDE.md §6 each grid counts its executed comparisons and fails if the count
ever falls to zero, so a refactor that stops exercising a fast path cannot pass
silently.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Expression, IndexExpression

# ── 1. Variable.__getitem__: plain int into a 1-D variable ──

SIZES = [1, 2, 5, 17]
# in-range, negative-in-range, both boundaries, and out of range on both ends
OFFSETS = [0, 1, 2, 4, 16, 17, 18, 99, -1, -2, -5, -17, -18, -99]


def _fresh(size, ndim=1):
    """A variable with an empty element cache, so the fast path really runs."""
    m = dm.Model()
    shape = (size,) if ndim == 1 else (size, 3)
    return m.continuous("x", shape=shape, lb=0.0, ub=1.0)


def _outcome(node_or_exc):
    """Comparable summary of one indexing attempt."""
    if isinstance(node_or_exc, BaseException):
        return ("raised", type(node_or_exc), str(node_or_exc))
    return (
        "node",
        type(node_or_exc),
        node_or_exc.base,
        node_or_exc.index,
        type(node_or_exc.index),
        node_or_exc._shape,
    )


def _try(fn):
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - the outcome under test
        return exc


def test_int_index_fast_path_matches_the_general_route():
    """``var[i]`` must be indistinguishable from ``Expression.__getitem__(var, i)``."""
    compared = 0
    for size in SIZES:
        for off in OFFSETS:
            for idx in (off, -off if off else 0):
                fast_var = _fresh(size)
                slow_var = _fresh(size)
                fast = _try(lambda v=fast_var, i=idx: v[i])
                slow = _try(lambda v=slow_var, i=idx: Expression.__getitem__(v, i))
                f, s = _outcome(fast), _outcome(slow)
                # `base` differs by construction (two distinct variables), so
                # compare it structurally and everything else exactly.
                if f[0] == "node":
                    assert f[2] is fast_var and s[2] is slow_var, (size, idx)
                    assert f[:2] + f[3:] == s[:2] + s[3:], (size, idx, f, s)
                else:
                    assert f == s, (size, idx, f, s)
                compared += 1
    assert compared > 0, "the equivalence grid degraded to zero comparisons"
    print(f"executed comparisons: {compared}")


def test_in_range_int_index_gets_the_scalar_shape():
    checked = 0
    for size in SIZES:
        var = _fresh(size)
        for i in list(range(size)) + list(range(-size, 0)):
            node = var[i]
            assert isinstance(node, IndexExpression)
            assert node._shape == (), (size, i)
            assert node.shape == (), (size, i)
            checked += 1
    assert checked > 0
    print(f"executed shape assertions: {checked}")


def test_out_of_range_int_index_raises_the_same_message():
    checked = 0
    for size in SIZES:
        for i in (size, size + 1, 999, -size - 1, -999):
            var = _fresh(size)
            with pytest.raises(IndexError) as fast:
                var[i]
            var2 = _fresh(size)
            with pytest.raises(IndexError) as slow:
                Expression.__getitem__(var2, i)
            assert str(fast.value) == str(slow.value), (size, i)
            assert str(fast.value) == (f"index {i} is out of bounds for axis 0 with size {size}")
            checked += 1
    assert checked > 0
    print(f"executed message comparisons: {checked}")


def test_int_index_is_canonicalised():
    var = _fresh(5)
    assert var[0] is var[0]
    assert var[-1] is var[-1]
    assert var[0] is not var[1]


@pytest.mark.parametrize(
    "idx",
    [
        (0, 1),  # tuple -> general route
        np.int64(0),  # not an exact int -> uncached general route
        True,  # bool -> numpy mask semantics, never the int fast path
        slice(None),
        Ellipsis,
        None,
        [0, 1],
    ],
)
def test_non_fast_path_indices_are_untouched(idx):
    """Everything the fast path does not claim must behave exactly as before."""
    fast_var = _fresh(5, ndim=2)
    slow_var = _fresh(5, ndim=2)
    fast = _outcome(_try(lambda: fast_var[idx]))
    slow = _outcome(_try(lambda: Expression.__getitem__(slow_var, idx)))
    if fast[0] == "node":
        assert fast[2] is fast_var and slow[2] is slow_var
        assert fast[:2] + fast[3:] == slow[:2] + slow[3:]
    else:
        assert fast == slow


def test_one_d_tuple_index_still_routes_through_the_general_path():
    """A 1-element tuple is not the fast path; its node must keep the tuple index."""
    var = _fresh(5)
    node = var[(0,)]
    assert node.index == (0,) and type(node.index) is tuple
    assert node._shape == ()


def test_scalar_variable_is_not_claimed_by_the_fast_path():
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    assert x._shape == () or x._shape is None
    fast = _outcome(_try(lambda: x[0]))
    m2 = dm.Model()
    y = m2.continuous("x", lb=0.0, ub=1.0)
    slow = _outcome(_try(lambda: Expression.__getitem__(y, 0)))
    if fast[0] == "node":
        assert fast[:2] + fast[3:] == slow[:2] + slow[3:]
    else:
        assert fast == slow


def test_shape_hint_is_the_only_constructor_change():
    """Without the hint the constructor must still infer, exactly as before."""
    var = _fresh(5)
    inferred = IndexExpression(var, 2)
    hinted = IndexExpression(var, 2, shape_hint=())
    assert inferred._shape == hinted._shape == ()
    # and the hint is honoured verbatim, including the "unknown" answer
    assert IndexExpression(var, 2, shape_hint=None)._shape is None
    assert IndexExpression(var, 2, shape_hint=(3,))._shape == (3,)


# ── 2. IndexedVar.__getitem__: the key cache ──


def _indexed(members):
    m = dm.Model()
    s = m.set("S", members)
    return m.continuous("y", over=s), s


def test_key_cache_returns_the_same_node_as_the_ordinal_route():
    checked = 0
    for members in (
        list(range(6)),
        ["a", "b", "c"],
        [("a", 1), ("b", 2), ("c", 3)],
    ):
        iv, s = _indexed(members)
        for key in members:
            expected = iv.flat[s.ordinal(key)]
            assert iv[key] is expected, (members, key)
            assert iv[key] is iv[key], (members, key)
            checked += 1
    assert checked > 0
    print(f"executed key comparisons: {checked}")


def test_key_cache_accepts_the_same_key_spellings_as_ordinal():
    iv, s = _indexed([("a", 1), ("b", 2)])
    canonical = iv[("a", 1)]
    assert iv["a", 1] is canonical  # flat
    assert iv[["a", 1]] is canonical  # list, coerced by _normalize_member
    iv1, _ = _indexed(["a", "b"])
    assert iv1[("a",)] is iv1["a"]  # 1-tuple unwrapped


def test_unknown_key_still_raises_ordinal_s_keyerror():
    iv, _ = _indexed(["a", "b"])
    for bad in ("z", ("z", 1), ["z", 1]):
        with pytest.raises(KeyError, match="is not a member of set 'S'"):
            iv[bad]
    # and a miss must not poison the cache
    assert iv["a"] is iv["a"]


def test_cache_miss_after_a_hit_is_independent():
    iv, s = _indexed(list(range(4)))
    first = iv[2]
    assert iv[3] is iv.flat[s.ordinal(3)]
    assert iv[2] is first
