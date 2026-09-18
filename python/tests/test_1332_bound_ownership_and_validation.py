"""#1332: a variable's box is owned and validated, not borrowed and trusted.

#1321 made every installed box read-only. Round 4 found what that left open:

1. the box *aliased the caller's array* -- ``_readonly_bound`` installed a
   read-only view over the caller's still-writable buffer, so writing to the
   array after declaring the variable silently moved the model's bounds;
2. ``copy.deepcopy`` and ``pickle`` rebuilt ``__dict__`` directly and handed
   back writable boxes (and a writable ``_bound_stack``, which holds the
   declared domain ``fix()`` validates against);
3. a BINARY with ``ub = 5`` was accepted by ``loads()`` and by the setter, and
   solved to ``b = 5.0``, ``optimal``, ``gap_certified=True``;
4. ``loads()`` never validated a ``bound_stack`` frame -- an inverted domain, a
   triple, a single, a wrong shape all loaded silently or raised a bare numpy
   error;
5. the setter validated no shape and no dtype -- ``x.lb = 5.0`` on a shape-(3,)
   variable stored a 0-d array and failed much later with an unrelated
   ``IndexError``; a complex bound was truncated to its real part;
6. the box was a view over a *writable* base, so the read-only flag could be
   flipped straight back on; and ``m.fixed({x: 1}, x=2)`` silently kept one of
   the two values.

The fix does the validating, the copy and the freeze in one place --
``_readonly_bound`` -- and restores the invariant after copying via
``Variable.__setstate__``.
"""

import copy
import json
import pickle

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.serialize import SerializationError, dumps, loads

# ── 1. the declared box must not alias the caller's array ───────────────────


@pytest.mark.correctness
def test_declared_box_does_not_alias_the_callers_array():
    """The issue's headline: reusing one array for two variables."""
    lo = np.zeros(2)
    lo[:] = 1.0
    m = dm.Model("i1332_alias")
    x = m.continuous("x", shape=(2,), lb=lo, ub=10)
    lo[:] = 5.0
    y = m.continuous("y", shape=(2,), lb=lo, ub=10)
    m.minimize(x[0] + x[1] + y[0] + y[1])

    assert np.allclose(x.lb, 1.0), "writing to the caller's array moved x's declared bound"
    assert np.allclose(y.lb, 5.0)
    r = m.solve()
    assert r.objective == pytest.approx(12.0, abs=1e-6), (
        f"objective {r.objective} -- the declared box followed the caller's array"
    )


@pytest.mark.unit
@pytest.mark.parametrize("factory", ["continuous", "integer"])
def test_constructor_copies_for_every_variable_kind(factory):
    m = dm.Model(f"i1332_ctor_{factory}")
    lo = np.zeros(3) + 1.0
    v = getattr(m, factory)("v", shape=(3,), lb=lo, ub=10)
    lo[:] = 7.0
    assert np.allclose(v.lb, 1.0)


@pytest.mark.unit
def test_setter_copies():
    m = dm.Model("i1332_setter")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    a = np.zeros(3) + 2.0
    x.lb = a
    a[:] = 6.0
    assert np.allclose(x.lb, 2.0), "x.lb = arr kept a live view of arr"

    b = np.zeros(3) + 8.0
    x.ub = b
    b[:] = 1.0
    assert np.allclose(x.ub, 8.0)


# ── 2. deepcopy and pickle must not launder the read-only flag ──────────────


@pytest.mark.correctness
@pytest.mark.parametrize("clone", ["deepcopy", "pickle"], ids=["deepcopy", "pickle"])
def test_copying_a_model_keeps_its_boxes_read_only(clone):
    m = dm.Model("i1332_copy")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.minimize(x[0] + x[1])

    m2 = copy.deepcopy(m) if clone == "deepcopy" else pickle.loads(pickle.dumps(m))
    v = m2._variables[0]
    assert not v.lb.flags.writeable, f"{clone} handed back a writable lb"
    assert not v.ub.flags.writeable, f"{clone} handed back a writable ub"
    with pytest.raises(ValueError):
        v.lb[0] = -5.0


@pytest.mark.correctness
def test_copying_a_model_keeps_its_fix_stack_read_only():
    """``_bound_stack[0]`` is the declared domain ``unfix()`` restores."""
    m = dm.Model("i1332_copy_stack")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.minimize(x[0] + x[1])
    x.fix(np.array([1.0, 1.0]))

    for clone in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        v = clone._variables[0]
        assert v._bound_stack, "the fix stack must survive the copy"
        for frame in v._bound_stack:
            for arr in frame:
                assert not arr.flags.writeable, "a copied fix frame came back writable"


# ── 3. a binary's box must stay inside [0, 1] ───────────────────────────────


@pytest.mark.correctness
def test_loads_refuses_a_binary_with_ub_above_one():
    m = dm.Model("i1332_bin")
    b = m.binary("b")
    m.maximize(b)

    doc = json.loads(dumps(m))
    doc["variables"][0]["ub"] = 5.0
    with pytest.raises((ValueError, SerializationError)):
        loads(json.dumps(doc))


@pytest.mark.correctness
def test_setter_refuses_a_binary_outside_the_unit_box():
    m = dm.Model("i1332_bin2")
    b = m.binary("b")
    m.maximize(b)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        b.ub = 5.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        b.lb = -2.0
    assert float(b.ub) == 1.0 and float(b.lb) == 0.0, "a refused write must change nothing"


@pytest.mark.unit
def test_a_binary_box_inside_the_unit_box_is_still_accepted():
    """Refusing ub>1 must not refuse the ordinary node-loop fixings."""
    m = dm.Model("i1332_bin3")
    b = m.binary("b", shape=(2,))
    b.lb = np.array([1.0, 0.0])
    b.ub = np.array([1.0, 0.0])
    assert np.allclose(b.lb, [1.0, 0.0])
    b.lb = np.array([0.0, -1e-17])  # an ulp-scale excursion is snapped, not refused
    assert np.all(np.asarray(b.lb) >= 0.0)


# ── 4. loads() must validate every bound_stack frame ────────────────────────


def _model_with_stack(stack):
    m = dm.Model("i1332_stack")
    v = m.continuous("v", lb=0.0, ub=10.0)
    m.minimize(v)
    doc = json.loads(dumps(m))
    doc["variables"][0]["bound_stack"] = stack
    return json.dumps(doc)


@pytest.mark.correctness
@pytest.mark.parametrize(
    "stack,label",
    [
        ([[10.0, 0.0]], "inverted"),
        ([[0.0, 10.0, 99.0]], "triple"),
        ([[0.0]], "single"),
        ([[[0.0, 1.0], [2.0]]], "wrong-shape"),
        (["nonsense"], "not-a-pair"),
    ],
)
def test_loads_refuses_a_malformed_bound_stack_frame(stack, label):
    with pytest.raises(SerializationError):
        loads(_model_with_stack(stack))


@pytest.mark.correctness
def test_a_well_formed_bound_stack_still_round_trips():
    """The control: the refusals must not cost the feature they guard."""
    m = dm.Model("i1332_roundtrip")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    x.fix(3.0)
    back = loads(dumps(m))
    v = back._variables[0]
    assert v.fix_depth == 1
    assert float(v.lb) == 3.0 and float(v.ub) == 3.0
    v.unfix()
    assert float(v.lb) == 0.0 and float(v.ub) == 10.0
    assert not v.lb.flags.writeable


# ── 5. the setter must validate shape and dtype ─────────────────────────────


@pytest.mark.correctness
@pytest.mark.parametrize(
    "value,exc",
    [
        ([1.0, 2.0], ValueError),
        (None, ValueError),
        (np.nan, ValueError),
        (np.array([1 + 2j, 0.0, 0.0]), TypeError),
    ],
    ids=["wrong-length", "none", "nan", "complex"],
)
def test_setter_refuses_a_box_it_cannot_honour(value, exc):
    m = dm.Model("i1332_validate")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    with pytest.raises(exc):
        x.lb = value
    assert np.allclose(x.lb, 0.0), "a refused write must change nothing"


@pytest.mark.unit
def test_setter_reshapes_a_size_matching_box():
    """A size-matching array that does not *broadcast* is still the caller's box.

    ``u.lb = np.array([0.5])`` on a SCALAR variable is ordinary usage (the
    relaxation layer's own unit tests do it), and ``np.broadcast_to((1,), ())``
    raises. Only a genuine element-count mismatch is refused.
    """
    m = dm.Model("i1332_reshape")
    u = m.continuous("u", lb=0.0, ub=1.0)
    u.lb = np.array([0.5])
    assert u.lb.shape == () and float(u.lb) == 0.5

    g = m.continuous("g", shape=(2, 3), lb=0.0, ub=1.0)
    g.lb = np.arange(6, dtype=float) / 10.0
    assert g.lb.shape == (2, 3)
    assert np.allclose(g.lb, np.arange(6).reshape(2, 3) / 10.0)

    with pytest.raises(ValueError):
        g.lb = np.zeros(5)


@pytest.mark.unit
def test_setter_broadcasts_a_scalar_to_the_variable_shape():
    m = dm.Model("i1332_broadcast")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    x.lb = 5.0
    assert x.lb.shape == (3,), f"a scalar stored as {x.lb.shape}, not the variable's shape"
    assert np.allclose(x.lb, 5.0)


# ── 6. the box's base must be read-only too; fixed() must not silently drop ──


@pytest.mark.unit
def test_the_read_only_flag_cannot_be_flipped_back_on():
    m = dm.Model("i1332_flag")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=5.0)
    box = x.lb
    with pytest.raises(ValueError):
        box.flags.writeable = True
    assert box.base is not None and not box.base.flags.writeable, (
        "the box must be a view over a read-only base, or the flag can be reset"
    )
    with pytest.raises(ValueError):
        box.base[...] = 5.0


@pytest.mark.correctness
def test_model_fixed_refuses_the_same_variable_twice():
    m = dm.Model("i1332_dupfix")
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.minimize(x)
    with pytest.raises(ValueError, match="more than once"):
        with m.fixed({x: 1.0}, x=2.0):
            pass
    assert x.fix_depth == 0, "the refused scope must leave no frame behind"

    # The control: two DIFFERENT variables in one call still work.
    y = m.continuous("y", lb=0.0, ub=5.0)
    with m.fixed({x: 1.0}, y=2.0):
        assert float(x.lb) == 1.0 and float(y.lb) == 2.0
    assert x.fix_depth == 0 and y.fix_depth == 0


# ── the no-copy fast path the node loop depends on ──────────────────────────


@pytest.mark.unit
def test_reinstalling_one_of_our_own_boxes_does_not_copy():
    """``v.lb, v.ub = v._bound_stack.pop()`` must stay allocation-free."""
    m = dm.Model("i1332_nocopy")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=5.0)
    box = x.lb
    x.lb = box
    assert x.lb is box, "an already-frozen box of ours was copied on reinstall"

    y = m.continuous("y", shape=(4,), lb=0.0, ub=5.0)
    y.lb = box
    assert y.lb is box
