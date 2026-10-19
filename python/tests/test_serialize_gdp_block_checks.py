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
block disjunction, and the nesting must be acyclic. ``start`` is not cross-checked
against row positions: ``transform_current_disjunctive_state`` leaves it stale
when it drops earlier rows (see ``test_a_transformed_model_still_round_trips``),
so a position check would refuse documents a writer produces.
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


def test_a_transformed_model_still_round_trips():
    """A transformation leaves ``_BlockMembers.start`` stale; the reader accepts it.

    Dropping the decided ``first`` disjunction's rows moves the block rows down
    while their records keep the old start, so a check of start against row
    positions would refuse this writer's own document.
    """
    from discopt.gdp.state import transform_current_disjunctive_state
    from discopt.gdp.tree import gdp_tree

    m = dm.Model("transformed")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.either_or([[x <= 1.0], [x >= 9.0]], name="first")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3.0)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6.0)
    a2 = m.make_disjunct("A2")
    a2.subject_to(y >= 8.0)
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to(x >= 7.0)
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + 3.0 * y)
    first = next(n for n in gdp_tree(m).walk() if n.name == "first")
    transform_current_disjunctive_state(m, [first.disjuncts[1]])
    starts = [rec.start for rec in m._gdp_block_members.values()]
    text = dumps(m, provenance=False)
    first_rows = [
        i for i, r in enumerate(json.loads(text)["rows"]) if r.get("disjunction") is not None
    ]
    assert min(first_rows) < min(starts)  # the start really is stale
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
