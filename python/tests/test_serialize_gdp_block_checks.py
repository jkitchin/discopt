"""Decode-time checks on Disjunct-block state (follow-up to #1801 / #1803).

A 1.7 document's block state -- ``block_indicator`` rows, ``nested_disjunction``
rows and the ``_gdp_blocks`` record -- refers into the node table. Before these
checks a reference of the wrong kind loaded: a ``parent`` on a continuous column
or an expression node failed only at solve (an unverified certificate, or a raw
``AttributeError``), and an *indicator* on a continuous column loaded and solved
a different model without a word. Each mutation below must be refused at
``loads`` with a message naming the field.

The record of each block disjunction must also agree with the rows it describes
(its indicators, parent and semantics), every parent must belong to a recorded
block disjunction, and the nesting must be acyclic. Its ``start`` must be the
position of its first row (for a block disjunction that tags none, no later than
its first descendant); ``transform_current_disjunctive_state`` renumbers it when
it drops earlier rows (``test_a_transformation_renumbers_block_starts``).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import discopt.modeling as dm
import pytest
from discopt.serialize import SerializationError, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures" / "v1.7"


def _doc(case):
    return json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))


def _var_node(doc, name):
    """Index of the node referring to variable *name*."""
    ref = next(i for i, v in enumerate(doc["variables"]) if v["name"] == name)
    return next(i for i, n in enumerate(doc["nodes"]) if n == {"op": "var", "ref": ref})


def _expr_node(doc):
    return next(i for i, n in enumerate(doc["nodes"]) if n["op"] == "binop")


def _members(doc):
    return doc["state"]["_gdp_blocks"]["members"]


def _disjuncts(doc):
    return doc["state"]["_gdp_blocks"]["disjuncts"]


def _row(doc, kind, k=0):
    return [r for r in doc["rows"] if r["kind"] == kind][k]


def _set(path, value):
    def mutate(doc):
        *head, last = path(doc)
        target = doc
        for key in head:
            target = target[key]
        target[last] = value(doc) if callable(value) else value

    return mutate


def _x(doc):
    return _var_node(doc, "x")


# (case, mutation, the field the refusal must name). Each mutation loaded before.
MUTATIONS = {
    "row parent on a continuous column": (
        "gdp_nested_blocks",
        lambda d: _row(d, "block_indicator").__setitem__("parent", _x(d)),
        "parent",
    ),
    "row parent on an expression node": (
        "gdp_nested_blocks",
        lambda d: _row(d, "block_indicator").__setitem__("parent", _expr_node(d)),
        "parent",
    ),
    "block row indicator on a continuous column": (
        "gdp_or_block",
        lambda d: _row(d, "block_indicator").__setitem__("indicator", _x(d)),
        "indicator",
    ),
    "nested disjunction parent on a continuous column": (
        "gdp_nested_blocks",
        lambda d: _row(d, "nested_disjunction").__setitem__("parent_indicator", _x(d)),
        "parent_indicator",
    ),
    "nested disjunction parent on an expression node": (
        "gdp_nested_blocks",
        lambda d: _row(d, "nested_disjunction").__setitem__("parent_indicator", _expr_node(d)),
        "parent_indicator",
    ),
    "member indicator on a continuous column": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1]["indicators"].__setitem__(0, _x(d)),
        "indicators",
    ),
    "member parent on a continuous column": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1].__setitem__("parent", _x(d)),
        "parent",
    ),
    "disjunct indicator on a continuous column": (
        "gdp_nested_blocks",
        lambda d: _disjuncts(d)[0].__setitem__("indicator", _x(d)),
        "indicator",
    ),
    "disjunct local on an expression node": (
        "gdp_nested_blocks",
        lambda d: _disjuncts(d)[1].__setitem__("local", [_expr_node(d)]),
        "local",
    ),
    "a block nested in itself": (
        "gdp_nested_blocks",
        lambda d: _disjuncts(d)[0]["nested"][0].__setitem__("block", [0]),
        "cycle",
    ),
    "two blocks nested in each other": (
        "gdp_nested_blocks",
        lambda d: _disjuncts(d)[1]["nested"].append(
            {"block": [0], "name": None, "semantics": "SELECT_ONE"}
        ),
        "cycle",
    ),
    "member semantics disagreeing with its rows": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1].__setitem__("semantics", "OR"),
        "semantics",
    ),
    "member parent disagreeing with its rows": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1].__setitem__("parent", None),
        "parent",
    ),
    "member parent on the wrong disjunct": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1].__setitem__("parent", _disjuncts(d)[3]["indicator"]),
        "parent",
    ),
    "member indicators swapped": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1]["indicators"].reverse(),
        "indicators",
    ),
    "member indicator dropped": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1]["indicators"].pop(),
        "indicators",
    ),
    "the nested record dropped": (
        "gdp_nested_blocks",
        lambda d: _members(d).pop(1),
        "no '_gdp_blocks' member record",
    ),
    "the enclosing record dropped": (
        "gdp_nested_blocks",
        lambda d: _members(d).pop(0),
        "no '_gdp_blocks' member record",
    ),
    "the start of a block disjunction with rows moved": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1].__setitem__("start", _members(d)[1]["start"] + 1),
        "'start'",
    ),
    "the start of a block disjunction with rows moved earlier": (
        "gdp_nested_blocks",
        lambda d: _members(d)[1].__setitem__("start", _members(d)[1]["start"] - 1),
        "'start'",
    ),
    "a disjunct listing another disjunct's row": (
        "gdp_nested_blocks",
        lambda d: _disjuncts(d)[0].__setitem__("rows", [1]),
        "rows",
    ),
    "an empty nested disjunction": (
        "gdp_nested_blocks",
        lambda d: _row(d, "nested_disjunction").__setitem__("disjuncts", []),
        "no disjuncts",
    ),
    "an empty top-level disjunction": (
        "nested_disjunction",
        lambda d: _row(d, "disjunction").__setitem__("disjuncts", []),
        "no disjuncts",
    ),
    "an empty disjunction as a disjunct member": (
        "nested_disjunction",
        lambda d: _row(d, "disjunction")["disjuncts"][0][0]["disjunction"].__setitem__(
            "disjuncts", []
        ),
        "no disjuncts",
    ),
}


def _refusal(label):
    """The refusal of mutation *label*, after checking the pristine document loads."""
    case, mutate, field = MUTATIONS[label]
    pristine = _doc(case)
    loads(json.dumps(pristine))
    doc = _doc(case)
    mutate(doc)
    assert doc != pristine, label  # the mutation took effect
    with pytest.raises(SerializationError, match=field) as info:
        loads(json.dumps(doc))
    return info.value


@pytest.mark.parametrize("label", list(MUTATIONS))
def test_a_mutated_block_reference_is_refused(label):
    _refusal(label)


def test_block_reference_probe_fired():
    """§6: count the refusals in one process, so the count cannot be vacuous."""
    refused = sum(isinstance(_refusal(label), SerializationError) for label in MUTATIONS)
    print(f"block-state mutations refused: {refused} of {len(MUTATIONS)}")
    assert refused == len(MUTATIONS) >= 20


def _rowless_after(first: bool):
    """A row-less block disjunction ``R`` between two others, optionally after
    ``first``, a 10-row block disjunction a transformation can decide away.

    ``R``'s blocks hold only nested block disjunctions, so ``R`` tags no row and is
    known only from its ``_BlockMembers`` record, placed by ``start``; its emission
    is 7 rows, fewer than the 10 that deciding ``first`` drops before it.
    """
    m = dm.Model("rowless")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    if first:
        f1 = m.make_disjunct("F1")
        f1.subject_to([x <= 9.0, x <= 9.1, x <= 9.2, x <= 9.3, x <= 9.4])
        f2 = m.make_disjunct("F2")
        f2.subject_to([y <= 9.0, y <= 9.1, y <= 9.2, y <= 9.3, y <= 9.4])
        m.add_disjunction([f1, f2], name="first")
    p, q = m.make_disjunct("P"), m.make_disjunct("Q")
    for outer, lo in ((p, 0.0), (q, 5.0)):
        inner = [m.make_disjunct(f"{outer.name}{k}") for k in (1, 2)]
        inner[0].subject_to(x >= lo)
        inner[1].subject_to(y >= lo)
        outer.add_disjunction(inner, name=f"in_{outer.name}")
    m.add_disjunction([p, q], name="R")
    m.either_or([[x <= 4.0], [y <= 4.0]], name="last")
    m.minimize(x + y)
    return m


def _root_order(model):
    from discopt.gdp.tree import gdp_tree

    # A block disjunction's node is named by its key, ``add_disjunction:<name>:<n>``;
    # the counter differs between the two builds, the name does not.
    return [
        r.name.rsplit(":", 1)[0] if r.kind == "add_disjunction" else r.name
        for r in gdp_tree(model).disjunctions
    ]


def test_a_transformation_renumbers_block_starts():
    """Deciding ``first`` drops its rows; the records after them move down too.

    Before the renumbering ``R`` kept its old start, 10 rows too late, sorted after
    ``last`` in the GDP tree, and its document was refused by the start check.
    """
    from discopt.gdp.state import transform_current_disjunctive_state
    from discopt.gdp.tree import gdp_tree

    m = _rowless_after(first=True)
    assert _root_order(m) == ["add_disjunction:first", "add_disjunction:R", "last"]
    first = next(n for n in gdp_tree(m).walk() if n.name.startswith("add_disjunction:first"))
    token = transform_current_disjunctive_state(m, [first.disjuncts[0]])
    fresh = _rowless_after(first=False)
    assert _root_order(m) == _root_order(fresh) == ["add_disjunction:R", "last"]
    # Each record starts at its first row; the row-less ``R`` at its first nested one.
    # (The positions are not a fresh build's: the decided selector row stays.)
    for model in (m, fresh):
        members = model._gdp_block_members
        tags = [getattr(c, "disjunction", None) for c in model._constraints]
        first_row = {
            k: min(i for i, t in enumerate(tags) if t and t[0] == k)
            for k in members
            if any(t and t[0] == k for t in tags)
        }
        assert {k: members[k].start for k in first_row} == first_row
        (r,) = (rec for rec in members.values() if rec.name == "R")
        assert r.start == min(first_row.values())
    token.reverse()
    assert _root_order(m) == ["add_disjunction:first", "add_disjunction:R", "last"]


def test_a_transformed_model_round_trips():
    """The transformed model's document passes the reader's start check."""
    from discopt.gdp.state import transform_current_disjunctive_state
    from discopt.gdp.tree import gdp_tree

    m = _rowless_after(first=True)
    first = next(n for n in gdp_tree(m).walk() if n.name.startswith("add_disjunction:first"))
    transform_current_disjunctive_state(m, [first.disjuncts[0]])
    text = dumps(m, provenance=False)
    assert dumps(loads(text), provenance=False) == text


@pytest.mark.parametrize("build", ["either_or", "disjunction"])
def test_an_empty_disjunction_is_refused_at_construction(build):
    m = dm.Model("empty")
    m.continuous("x", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="no disjuncts"):
        getattr(m, build)([], name="d")


# ── the OR fixture that solving can tell from select-one ─────────────────────


def gdp_or_block_both():
    """An OR block disjunction whose optimum needs both modes at once.

    The registry's ``gdp_or_block`` solves to 13 under OR and SELECT_ONE alike, so
    its round-trip solve checks no semantics; its v1.7 golden document is frozen,
    so the discriminating variant lives here. ``lo.active + hi.active >= 2`` forces
    both modes: OR gives x = 2, y = 3, objective 5; read as select-one it is
    infeasible.
    """
    m = dm.Model("gdp_or_block_both")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    lo = m.make_disjunct("lo")
    lo.subject_to(x <= 2.0)
    hi = m.make_disjunct("hi")
    hi.subject_to(y <= 3.0)
    m.add_disjunction([lo, hi], name="modes", semantics="or")
    m.subject_to(lo.indicator.variable + hi.indicator.variable >= 2)
    m.maximize(x + y)
    return m


def _solve(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.solve()


def test_the_or_round_trip_solves_to_the_or_optimum():
    loaded = loads(dumps(gdp_or_block_both(), provenance=False))
    got = _solve(loaded)
    assert got.status == "optimal"
    assert got.objective == pytest.approx(5.0, abs=1e-6)


def test_the_or_fixture_read_as_select_one_is_infeasible():
    """The fixture discriminates: the same document with OR stripped is infeasible."""
    doc = json.loads(dumps(gdp_or_block_both(), provenance=False))
    n = 0
    for row in doc["rows"]:
        if row["kind"] == "block_indicator":
            row["kind"] = "indicator"
            row.pop("semantics")
            n += 1
    assert n == 2
    # The selector row is at_least(1); select-one's is exactly(1).
    sel = next(r for r in doc["rows"] if r.get("name") == "_disj_modes_select")
    sel["sense"] = "=="
    got = _solve(loads(json.dumps(doc)))
    assert got.status == "infeasible"


def test_a_row_less_block_disjunction_starting_after_its_rows_is_refused():
    """``R`` tags no row, so its start is bounded by the first row nested in it."""
    doc = json.loads(dumps(_rowless_after(first=False), provenance=False))
    (r,) = (rec for rec in _members(doc) if rec["name"] == "R")
    assert r["start"] == 0
    r["start"] = 1
    with pytest.raises(SerializationError, match="'start' 1 is after row 0"):
        loads(json.dumps(doc))
    r["start"] = 0
    loads(json.dumps(doc))
