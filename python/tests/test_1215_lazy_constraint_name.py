"""``Constraint.name`` is materialized on first read (#1215, candidate 3).

A named indexed family used to format one ``"family[label]"`` string per row at
build time. The issue measures that at 49 B/instance retained and 0.497
us/instance of build wall, for metadata the solve path never reads.

These tests are the equivalence gate: every name the lazy path produces must be
byte-identical to the eager formatter it replaced, for every member spelling the
old code handled -- and the laziness must be real (nothing formatted until
someone asks). Per CLAUDE.md §6 the grids count their executed comparisons.
"""

import copy
import dataclasses

import discopt.modeling as dm
from discopt.modeling.core import Constraint
from discopt.modeling.indexed import key_label


def _eager_name(family, member):
    """The formatter this change replaced, verbatim."""
    label = key_label(member) if type(member) is tuple else str(member)
    return f"{family}[{label}]"


MEMBER_SETS = [
    list(range(5)),
    ["a", "bb", "c c"],
    [("a", 1), ("b", 2)],
    [(1, 2, 3), (4, 5, 6)],
    [0, 1],  # ints that also exercise the small-int cache
    [1.5, 2.5],
    [None, True],
]


def _family(members, name):
    """A *nonlinear* family, so rows stay real ``Constraint`` objects rather than
    being folded into the Rust builder by the fast-linear path."""
    m = dm.Model()
    s = m.set("S", members)
    x = m.continuous("x", over=s, lb=-10.0, ub=10.0)
    fam = m.constraint(s, lambda *k: x[k if len(k) > 1 else k[0]] * 2.0 <= 1.0, name=name)
    return m, s, fam


def test_lazy_name_matches_the_eager_formatter():
    compared = 0
    for members in MEMBER_SETS:
        _, s, fam = _family(members, "cap")
        for member in s:
            assert fam[member].name == _eager_name("cap", member), (members, member)
            compared += 1
    assert compared > 0, "the name grid degraded to zero comparisons"
    print(f"executed name comparisons: {compared}")


def test_nothing_is_formatted_until_the_name_is_read():
    """The point of the change: no string exists on an unread row."""
    _, s, fam = _family(list(range(5)), "cap")
    row = fam[3]
    assert row.__dict__["_name"] is None
    assert row.__dict__["_name_family"] == "cap"
    assert row.__dict__["_name_key"] == 3
    assert row.name == "cap[3]"
    assert row.__dict__["_name"] == "cap[3]"


def test_repeated_reads_format_once_and_agree():
    _, s, fam = _family(list(range(4)), "cap")
    row = fam[1]
    first = row.name
    assert row.name is first  # the same object, not a re-format
    assert first == "cap[1]"


def test_an_unnamed_family_leaves_no_pending_state():
    _, s, fam = _family(list(range(4)), None)
    for member in s:
        row = fam[member]
        assert row.name is None
        assert "_name_family" not in row.__dict__
        assert "_name_key" not in row.__dict__


def test_explicit_assignment_wins_over_a_pending_family_name():
    _, s, fam = _family(list(range(4)), "cap")
    unread, read_first = fam[0], fam[1]
    assert read_first.name == "cap[1]"  # materialize before overwriting
    for row, value in ((unread, "zero"), (read_first, "one")):
        row.name = value
        assert row.name == value
    # and clearing works from either state
    unread.name = None
    assert unread.name is None


def test_assignment_on_a_plain_constraint_is_an_ordinary_attribute():
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    c = x[0] * x[1] <= 1.0
    assert c.name is None
    assert "_name_family" not in c.__dict__
    c.name = "mine"
    assert c.name == "mine"
    assert Constraint(c.body, "<=", 0.0, "ctor").name == "ctor"


def test_dataclass_protocol_is_intact():
    _, s, fam = _family(list(range(3)), "cap")
    row = fam[2]
    assert [f.name for f in dataclasses.fields(Constraint)] == [
        "body",
        "sense",
        "rhs",
        "name",
    ]
    # replace() reads `name` back through the getter, so the copy is named
    assert dataclasses.replace(row).name == "cap[2]"
    assert dataclasses.replace(row, name="other").name == "other"


def test_a_shallow_copy_of_an_unread_row_still_names_itself():
    # Only `copy.copy`: a Constraint has never been deep-copyable or picklable
    # (its body reaches the Model, which holds a `PyModelBuilder`), so the
    # pending-name state has no deep-copy contract to preserve.
    _, s, fam = _family(list(range(3)), "cap")
    row = fam[1]
    assert row.__dict__["_name"] is None  # still unread
    assert copy.copy(row).name == "cap[1]"


def test_names_survive_export():
    """The consumer that actually reads names must see the old strings."""
    m = dm.Model()
    s = m.set("S", list(range(3)))
    x = m.continuous("x", over=s, lb=0.0, ub=1.0)
    # `fast=False` keeps the rows as Constraint objects (the fast-linear path
    # would fold them into one builder block named for the family instead).
    m.constraint(s, lambda i: x[i] <= 1.0, name="cap", fast=False)
    m.minimize(sum(x[i] for i in s))
    text = m.to_lp()
    missing = [i for i in range(3) if f"cap[{i}]" not in text]
    assert not missing, (missing, text[:2000])
