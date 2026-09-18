"""#1321: the read-only-box invariant, enforced at assignment rather than per call site.

``Model.saved_bounds(copy=False)`` snapshots boxes BY REFERENCE, once per
branch-and-bound node, and documents itself as safe because an in-place bound
write raises. #1311 made ``fix()`` freeze the box it pushes, which closed one
call site. This file pins the invariant on the paths that were still open --
``loads()``, the live box inside a ``fix()``, and the GAMS reader -- plus the
mirror-image scope hazard where ``saved_bounds`` would pop a ``fixed()`` scope
that is still open.

Each in-place write below is not merely untidy: on ``main`` before this change
every one of them changed the solve's answer (issue #1321 has the numbers).
"""

import json

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.gams_parser import parse_gams
from discopt.serialize import SerializationError, dumps, loads


def _fixed_then_saved(model):
    """`dumps` of *model* taken while its first variable is fixed."""
    return dumps(model)


# ── 1. a loaded model's restored fix stack ───────────────────────────────


@pytest.mark.smoke
def test_loaded_model_unfix_restores_a_read_only_box():
    """`loads()` rebuilt `_bound_stack` from `_dec_array`, which hands back
    writable arrays, so the first `unfix()` on a reloaded model installed a
    writable live box. An in-place write then reached through
    `saved_bounds(copy=False)`'s snapshot and changed the answer (0.0 -> 7.5
    for the bound, -10 -> -2.5 for the objective)."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.subject_to(x + y <= 8.0)
    m.minimize(x + 2 * y)
    with x.fixed(4.0):
        doc = _fixed_then_saved(m)

    mm = loads(doc)
    xx = mm._variables[0]
    assert xx.fix_depth == 1
    xx.unfix()
    assert not xx.lb.flags.writeable
    assert not xx.ub.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        with mm.saved_bounds(copy=False):
            xx.lb[()] = 7.5
    assert (float(xx.lb), float(xx.ub)) == (0.0, 10.0)
    assert mm.solve().objective == pytest.approx(-10.0, abs=1e-6)


@pytest.mark.smoke
def test_loaded_bound_stack_entries_are_read_only():
    """The declared domain lives at `_bound_stack[0]` and is what `fix()`
    validates against; a writable one there is the same corruption one level
    down."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    with x.fixed(4.0):
        mm = loads(dumps(m))
    xx = mm._variables[0]
    assert xx.fix_depth == 1
    for lo, hi in xx._bound_stack:
        assert not lo.flags.writeable
        assert not hi.flags.writeable


# ── 2. the live box installed by fix() ───────────────────────────────────


@pytest.mark.smoke
def test_the_box_installed_by_fix_is_read_only():
    """#1311 froze the box `fix()` *pushed*; the box it *installed* stayed
    writable, so an in-place write inside the fixed scope moved the pin (a
    (3,3) fix became (1,3) and the objective -7 became -9)."""
    m = dm.Model("t")
    a = m.continuous("a", lb=0.0, ub=10.0)
    b = m.continuous("b", lb=-5.0, ub=5.0)
    m.subject_to(a + b <= 8.0)
    m.minimize(a + 2 * b)
    with a.fixed(3.0):
        assert not a.lb.flags.writeable
        assert not a.ub.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            with m.saved_bounds(copy=False):
                a.lb[()] = 1.0
        assert (float(a.lb), float(a.ub)) == (3.0, 3.0)
        assert m.solve().objective == pytest.approx(-7.0, abs=1e-6)


@pytest.mark.smoke
def test_the_box_installed_by_a_partial_fix_is_read_only():
    """`fix(where=...)` builds its box with `np.where`, a second writable
    source that the per-call-site fix did not cover."""
    m = dm.Model("t")
    c = m.continuous("c", shape=(3,), lb=0.0, ub=10.0)
    m.minimize(c.sum())
    with c.fixed(2.0, where=[True, False, False]):
        assert not c.lb.flags.writeable
        assert not c.ub.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            c.ub[1] = 0.0


@pytest.mark.smoke
def test_every_assignment_route_installs_a_read_only_box():
    """The invariant is enforced at the property, so a box handed in from
    anywhere -- a plain list, a writable array a caller still holds -- is
    read-only once installed, and the caller's own array keeps its own flags."""
    m = dm.Model("t")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    mine = np.array([1.0, 2.0])
    x.lb = mine
    assert not x.lb.flags.writeable
    assert mine.flags.writeable, "the caller's array must not be frozen underneath it"
    x.ub = [8.0, 9.0]
    assert not x.ub.flags.writeable
    assert x.ub.dtype == np.float64
    assert list(x.lb) == [1.0, 2.0] and list(x.ub) == [8.0, 9.0]


# ── 3. the GAMS reader ───────────────────────────────────────────────────


@pytest.mark.smoke
def test_gams_scalar_bounds_are_read_only():
    """`x.lo = 1` went through `var.lb = np.array(...)` + an in-place write,
    leaving a writable declared box. This predates #1311; it is the same
    problem on a sibling path (the solve returned 5.0 instead of 1.0)."""
    gm = parse_gams(
        """
        Variables x, obj;
        Equations objdef;
        x.lo = 1;
        x.up = 10;
        objdef.. obj =e= x;
        Model mm /all/;
        Solve mm using nlp minimizing obj;
        """
    )
    gx = next(v for v in gm._variables if v.name == "x")
    assert (float(gx.lb), float(gx.ub)) == (1.0, 10.0)
    assert not gx.lb.flags.writeable
    assert not gx.ub.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        with gm.saved_bounds(copy=False):
            gx.lb[()] = 5.0


@pytest.mark.smoke
def test_gams_indexed_bounds_are_read_only_and_still_correct():
    """The per-element `.lo/.up/.fx` path edits a private copy and rebinds; the
    values it produces must be unchanged by that."""
    gm = parse_gams(
        """
        Set i /1*3/;
        Variables x(i), obj;
        Equations objdef;
        x.lo('1') = 1;
        x.up('2') = 4;
        x.fx('3') = 2;
        objdef.. obj =e= sum(i, x(i));
        Model mm /all/;
        Solve mm using nlp minimizing obj;
        """
    )
    gx = next(v for v in gm._variables if v.name == "x")
    assert not gx.lb.flags.writeable
    assert not gx.ub.flags.writeable
    assert float(gx.lb[0]) == 1.0
    assert float(gx.ub[1]) == 4.0
    assert (float(gx.lb[2]), float(gx.ub[2])) == (2.0, 2.0)


# ── 4. saved_bounds must not pop a fixed() scope that is still open ──────


@pytest.mark.smoke
def test_saved_bounds_refuses_to_discard_a_still_open_fixed_scope():
    """The mirror image of the #1311 hazard: a `fixed()` scope entered inside a
    `saved_bounds()` block and still open at its exit used to have its frame
    truncated away silently, leaving a context manager with nothing to restore
    and no error until (or unless) it ran."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    cm = x.fixed(2.0)
    with pytest.raises(RuntimeError, match="still open"):
        with m.saved_bounds():
            cm.__enter__()
    # The box is restored regardless -- that is saved_bounds' contract.
    assert (float(x.lb), float(x.ub)) == (0.0, 10.0)
    # The orphan's own exit also refuses, rather than pretending it restored
    # something: its frames are gone. Closed here so the generator is not
    # finalised (and its error swallowed) at some arbitrary later GC.
    with pytest.raises(RuntimeError, match="out of order"):
        cm.__exit__(None, None, None)


@pytest.mark.smoke
def test_saved_bounds_leaves_a_scope_opened_outside_it_alone():
    """No false positive: a `fixed()` scope that encloses the `saved_bounds()`
    block is untouched by it and still restores normally."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    with x.fixed(2.0):
        with m.saved_bounds():
            pass
        assert (float(x.lb), float(x.ub)) == (2.0, 2.0)
        assert x.fix_depth == 1
    assert (float(x.lb), float(x.ub)) == (0.0, 10.0)
    assert x.fix_depth == 0


@pytest.mark.smoke
def test_saved_bounds_still_unwinds_a_bare_leaked_fix():
    """A raw `fix()` leaked in the body carries no scope token, so it is still
    unwound silently -- the documented behaviour, unchanged."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    with m.saved_bounds():
        x.fix(2.0)
    assert (float(x.lb), float(x.ub)) == (0.0, 10.0)
    assert x.fix_depth == 0


@pytest.mark.smoke
def test_model_fixed_inside_saved_bounds_is_caught_too():
    """`Model.fixed` pushes the same tokens, so the guard covers it."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    cm = m.fixed(x=2.0)
    with pytest.raises(RuntimeError, match="still open"):
        with m.saved_bounds():
            cm.__enter__()
    assert (float(x.lb), float(x.ub)) == (0.0, 10.0)
    with pytest.raises(RuntimeError, match="out of order"):
        cm.__exit__(None, None, None)


# ── 5. documents the modeling API would refuse ───────────────────────────


def _doc_of(model):
    return json.loads(dumps(model))


def _tiny():
    m = dm.Model("t")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    return m, x


@pytest.mark.smoke
@pytest.mark.parametrize("bad_name", [1, None, ["a"], {"a": 1}])
def test_load_refuses_a_non_string_variable_name(bad_name):
    """`1` and `None` used to load silently; a list or dict failed much later
    with a bare `TypeError: unhashable type`, which does not say the document
    is at fault."""
    m, _ = _tiny()
    doc = _doc_of(m)
    doc["variables"][0]["name"] = bad_name
    with pytest.raises(SerializationError, match="name must be a string"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_load_refuses_duplicate_set_names():
    """`Model.set()` refuses a name already in use; `_dec_sets` did not, so two
    sets of one name reloaded and every lookup silently took the first."""
    m = dm.Model("t")
    m.set("i", [1, 2, 3])
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    doc = _doc_of(m)
    sets = doc["state"]["_sets"]
    assert len(sets) == 1, "probe fired against the wrong document shape"
    sets.append(dict(sets[0]))
    with pytest.raises(SerializationError, match="duplicate set name"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_load_refuses_an_initial_point_for_an_undeclared_variable():
    """`solve()` looks the point up by name, so an entry naming a variable the
    document does not declare dropped the user's warm start without a word."""
    m, _ = _tiny()
    doc = _doc_of(m)
    doc["initial_point"] = [["nope", 0, 1.0]]
    with pytest.raises(SerializationError, match="does not declare"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_load_refuses_an_out_of_range_initial_point_element():
    m, _ = _tiny()
    doc = _doc_of(m)
    doc["initial_point"] = [["x", 9, 1.0]]
    with pytest.raises(SerializationError, match="out of range"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_a_valid_initial_point_still_round_trips():
    """The new validation must not reject the documents it is guarding."""
    m = dm.Model("t")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    m.minimize(x.sum())
    m._initial_point = {("x", 0): 1.0, ("x", 2): 3.0}
    mm = loads(dumps(m))
    assert mm._initial_point == {("x", 0): 1.0, ("x", 2): 3.0}
