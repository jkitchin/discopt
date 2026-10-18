"""Schema 1.5: propositional logic constraints and complementarity lowering rows (WP-3).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-3 (DoD D6, D7), §3.6.

Through 1.4, saving a model with a ``Model.logical`` constraint was refused ("this
format version carries algebraic rows, indicator, SOS and disjunctive relations"),
and a complementarity lowering's ``rows`` were dropped on save, so a reload read
them as untracked (``None``) and ``rows_in`` / the lowered-row residual lost them.

The logic tree is written as a flat table (children first, root last); the
lowering (``gdp_reformulate._logical_to_linear``) is a function of the tree alone,
so a round trip is bound-neutral -- checked here by solving before and after and
requiring the same status, objective and node count (CLAUDE.md §5).
"""

from __future__ import annotations

import copy
import json
import pickle

import discopt.modeling as dm
import pytest
from discopt.modeling.core import (
    BooleanVar,
    LogicalAnd,
    _LogicalConstraint,
    atleast,
    atmost,
    exactly,
    land,
    lnot,
    lor,
    xor,
)
from discopt.serialize import SCHEMA, SerializationError, dumps, loads

# ── the six entry-experiment models: each connective, nested, over array indices ──

FORMULAS = {
    "and_or_not": lambda Y, z: (Y[0] & ~Y[1]) | (Y[2] & z),
    "implies": lambda Y, z: Y[0].implies(Y[1] | ~Y[3]),
    "iff": lambda Y, z: Y[1].equivalent_to(~Y[2] & z),
    "atleast": lambda Y, z: atleast(2, [Y[0], Y[1], Y[2], z]),
    "atmost_lor": lambda Y, z: land(atmost(1, Y[0], Y[3]), lor(Y[1], Y[2])),
    "nested": lambda Y, z: lnot(Y[0].implies(lnot(Y[1].equivalent_to(Y[2] | (Y[3] & ~z))))),
    "xor_exactly": lambda Y, z: xor(Y[0], Y[1]) & exactly(1, [Y[2], Y[3]]),
}


def _logic_model(kind: str):
    m = dm.Model(kind)
    Y = m.boolean("Y", shape=(4,))
    z = m.boolean("z")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=10.0)
    m.minimize(dm.sum(x) + 3 * Y.variable[0] - 2 * Y.variable[3] + z.variable)
    for i in range(4):
        m.subject_to(x[i] >= 2 * Y.variable[i], name=f"link{i}")
    m.logical(FORMULAS[kind](Y, z), name="L")
    return m


def test_schema_minor_is_1_5():
    assert SCHEMA == "discopt.model/1.5"


@pytest.mark.parametrize("kind", sorted(FORMULAS))
def test_logical_constraints_round_trip(kind):
    m = _logic_model(kind)
    text = dumps(m, provenance=False)
    m2 = loads(text)
    assert dumps(m2, provenance=False) == text
    rows = [c for c in m2._constraints if isinstance(c, _LogicalConstraint)]
    assert len(rows) == 1 and rows[0].name == "L"
    # And through pickle, whose children-first walk now includes logical leaves.
    assert dumps(pickle.loads(pickle.dumps(m2)), provenance=False) == text
    assert dumps(copy.deepcopy(m2), provenance=False) == text


@pytest.mark.parametrize("kind", sorted(FORMULAS))
def test_logical_round_trip_is_bound_neutral(kind):
    """Same status, objective and node count -- exactly, not approximately."""
    m = _logic_model(kind)
    r1 = m.solve(time_limit=60)
    r2 = loads(dumps(m)).solve(time_limit=60)
    assert r1.status in ("optimal", "infeasible")
    assert (r1.status, r1.objective, r1.node_count) == (r2.status, r2.objective, r2.node_count)


def test_logical_round_trip_lowers_to_the_same_rows():
    from discopt._relax.gdp_reformulate import reformulate_gdp

    for kind in FORMULAS:
        m = _logic_model(kind)
        a = [repr(c) for c in reformulate_gdp(m)._constraints]
        b = [repr(c) for c in reformulate_gdp(loads(dumps(m)))._constraints]
        assert a == b, kind


def test_logical_constraint_over_boolean_array_index_and_scalar():
    """Leaves of both shapes Model.boolean produces: an element of a binary array
    and a size-1 binary variable (``shape=()`` and ``shape=(1,)``)."""
    m = dm.Model("leaves")
    Y = m.boolean("Y", shape=(3,))
    a = m.boolean("a")
    b = m.boolean("b", shape=1)
    m.minimize(Y.variable[0] + a.variable + b.variable)
    m.logical(atleast(2, [Y[0], Y[2], a, b]), name="card")
    doc = json.loads(dumps(m))
    logic = next(r for r in doc["rows"] if r["kind"] == "logical")["logic"]
    leaf_ops = {doc["nodes"][e["x"]]["op"] for e in logic if e["c"] == "var"}
    assert leaf_ops == {"index", "var"}
    m2 = loads(json.dumps(doc))
    assert dumps(m2) and m2.solve(time_limit=30).objective == m.solve(time_limit=30).objective


def test_a_shared_subformula_is_written_once():
    m = dm.Model("shared")
    Y = m.boolean("Y", shape=(3,))
    m.minimize(Y.variable[0])
    s = Y[1] | ~Y[2]
    m.logical(LogicalAnd(s, Y[0].implies(s)), name="s")
    logic = next(r for r in json.loads(dumps(m))["rows"] if r["kind"] == "logical")["logic"]
    assert sum(1 for e in logic if e["c"] == "or") == 1
    m2 = loads(dumps(m))
    root = next(c for c in m2._constraints if isinstance(c, _LogicalConstraint)).expression
    assert root.left is root.right.consequent  # still one object after the reload


def test_a_long_land_chain_round_trips():
    """``land`` over many arguments is a left-deep chain; nested JSON would nest
    that deep, the flat table does not."""
    m = dm.Model("chain")
    Y = m.boolean("Y", shape=(3000,))
    m.minimize(Y.variable[0])
    m.logical(land(*[Y[i] | Y[i + 1] for i in range(2999)]), name="chain")
    text = dumps(m, provenance=False)
    assert dumps(loads(text), provenance=False) == text


def test_logic_interleaved_with_other_rows_keeps_its_place():
    m = _logic_model("implies")
    Y = m.boolean("W", shape=(2,))
    m.logical(Y[0] | Y[1], name="second")
    m.subject_to(Y.variable[0] <= 1, name="after")
    kinds = [r["kind"] for r in json.loads(dumps(m))["rows"]]
    assert kinds == ["algebraic"] * 4 + ["logical", "logical", "algebraic"]
    assert [type(c).__name__ for c in loads(dumps(m))._constraints] == [
        type(c).__name__ for c in m._constraints
    ]


@pytest.mark.parametrize(
    "make_leaf, needle",
    [
        (lambda m: m.continuous("c", lb=0, ub=1), "not binary"),
        (lambda m: m.integer("i", lb=0, ub=1), "not binary"),
        (lambda m: m.binary("v", shape=(2,)), "has 2 elements"),
    ],
)
def test_a_boolean_over_a_non_binary_variable_is_refused_on_save(make_leaf, needle):
    """``Model.boolean`` never builds these; the writer refuses rather than write a
    document the reader would refuse."""
    m = dm.Model("bad_leaf")
    good = m.boolean("g")
    m.minimize(good.variable)
    m.logical(BooleanVar(make_leaf(m)) | good, name="L")
    with pytest.raises(SerializationError, match=needle):
        dumps(m)


# ── LoweringRecord.rows ───────────────────────────────────────────────────


def _lowered_model():
    m = dm.Model("mpcc")
    p = m.continuous("p", shape=(2,), lb=0.0, ub=10.0)
    q = m.continuous("q", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(dm.sum((p - 1) ** 2) + dm.sum((q - 1) ** 2))
    m.subject_to(p[0] + q[1] >= 0.5, name="mix")
    m.complementarity(p, q, name="pq")
    return m


def _record(m):
    (rel,) = m._complementarities
    return rel, m._lowered_complementarities[rel]


def test_lowering_record_rows_round_trip():
    m = _lowered_model()
    rel, record = _record(m)
    assert record.rows  # the declaration lowered it and tracked its rows
    positions = [m._constraints.index(r) for r in record.rows]
    m2 = loads(dumps(m))
    rel2, record2 = _record(m2)
    assert record2.method == record.method
    assert [m2._constraints.index(r) for r in record2.rows] == positions
    # The same objects as the reloaded model's rows, not copies.
    assert all(any(r is c for c in m2._constraints) for r in record2.rows)
    assert len(rel2.rows_in(m2)) == len(rel.rows_in(m))


def test_lowering_rows_none_stays_none_and_empty_stays_empty():
    m = _lowered_model()
    rel, record = _record(m)
    record.rows = None
    doc = json.loads(dumps(m))
    assert doc["complementarities"]["lowered"][0] == [0, record.method]  # 1.4-readable form
    assert _record(loads(json.dumps(doc)))[1].rows is None
    record.rows = []
    assert _record(loads(dumps(m)))[1].rows == []


def test_lowering_rows_survive_pickle_and_deepcopy_too():
    m = loads(dumps(_lowered_model()))
    for copy_ in (pickle.loads(pickle.dumps(m)), copy.deepcopy(m)):
        _, record = _record(copy_)
        assert record.rows and all(any(r is c for c in copy_._constraints) for r in record.rows)


def test_a_lowering_row_no_longer_in_the_model_is_refused_on_save():
    m = _lowered_model()
    _, record = _record(m)
    m._constraints.remove(record.rows[0])
    with pytest.raises(SerializationError, match="no longer in the model"):
        dumps(m)


def test_a_bound_neutral_solve_after_reload_of_a_lowered_mpcc():
    m = _lowered_model()
    r1 = m.solve(time_limit=60)
    r2 = loads(dumps(m)).solve(time_limit=60)
    assert (r1.status, r1.objective, r1.node_count) == (r2.status, r2.objective, r2.node_count)


# ── malformed 1.5 content is refused ───────────────────────────────────────


def _logic_doc():
    m = _logic_model("atmost_lor")
    return json.loads(dumps(m))


def _logic(doc):
    return next(r for r in doc["rows"] if r["kind"] == "logical")["logic"]


def _first_c(doc, c):
    return next(e for e in _logic(doc) if e["c"] == c)


def _mpcc_doc():
    return json.loads(dumps(_lowered_model()))


def _lowered(doc):
    return doc["complementarities"]["lowered"][0]


MALFORMED = {
    "unknown connective": (_logic_doc, lambda d: _first_c(d, "or").update(c="nand")),
    "k as string": (_logic_doc, lambda d: _first_c(d, "atmost").update(k="1")),
    "k as bool": (_logic_doc, lambda d: _first_c(d, "atmost").update(k=True)),
    "args not a list": (_logic_doc, lambda d: _first_c(d, "atmost").update(args=0)),
    "forward logic ref": (_logic_doc, lambda d: _first_c(d, "and").update(a=999)),
    "negative logic ref": (_logic_doc, lambda d: _first_c(d, "and").update(a=-1)),
    "empty logic table": (_logic_doc, lambda d: _logic(d).clear()),
    "leaf on a continuous variable": (
        _logic_doc,
        lambda d: _first_c(d, "var").update(
            x=next(i for i, n in enumerate(d["nodes"]) if n["op"] == "var" and n["ref"] == 2)
        ),
    ),
    "leaf on a non-variable node": (
        _logic_doc,
        lambda d: _first_c(d, "var").update(
            x=next(i for i, n in enumerate(d["nodes"]) if n["op"] == "binop")
        ),
    ),
    "unknown logic entry key": (_logic_doc, lambda d: _first_c(d, "or").update(weight=1)),
    "logical name not a string": (
        _logic_doc,
        lambda d: next(r for r in d["rows"] if r["kind"] == "logical").update(name=5),
    ),
    "lowering row past the end": (_mpcc_doc, lambda d: _lowered(d)[2].append(10_000)),
    "lowering row negative": (_mpcc_doc, lambda d: _lowered(d)[2].__setitem__(0, -1)),
    "lowering row twice": (_mpcc_doc, lambda d: _lowered(d)[2].append(_lowered(d)[2][0])),
    "lowering rows not a list": (_mpcc_doc, lambda d: _lowered(d).__setitem__(2, 3)),
    "lowering entry of four": (_mpcc_doc, lambda d: _lowered(d).append(None)),
    "relation lowered twice": (
        _mpcc_doc,
        lambda d: d["complementarities"]["lowered"].append(list(_lowered(d))),
    ),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_logic_and_lowering_rows_are_refused(case):
    make, corrupt = MALFORMED[case]
    doc = make()
    corrupt(doc)
    with pytest.raises(SerializationError):
        loads(json.dumps(doc))


def test_the_malformed_probe_fired():
    """§6: every case starts from a document that loads unmodified."""
    assert len(MALFORMED) >= 15
    for make in {make for make, _ in MALFORMED.values()}:
        assert isinstance(loads(json.dumps(make())), dm.Model)


def test_a_three_element_lowering_entry_is_refused_before_1_5():
    """Rows were never written before 1.5: a pre-1.5 document carrying them is
    malformed, not an old document to read leniently."""
    doc = _mpcc_doc()
    doc["schema"] = "discopt.model/1.4"
    with pytest.raises(SerializationError, match="lowering entry"):
        loads(json.dumps(doc))


# ── what older readers did with 1.5 documents (recorded by the WP-1 generator) ──


def test_older_readers_refuse_1_5_logic_and_tracked_rows():
    """Run at generation with each shipped reader (1.0-1.4, own builds): the
    1.5-only content is refused, never misread, and nothing else changed."""
    import pathlib

    manifest = json.loads(
        (
            pathlib.Path(__file__).parent / "data" / "serialize_fixtures" / "MANIFEST.json"
        ).read_text()
    )
    only_15 = {
        "logical_constraints.json",  # a "logical" row
        "complementarity_lowered.json",  # tracked lowering rows: a 3-element entry
        "complementarity_vector_lowered.json",
        "from_nl_mpcc.json",
    }
    checked = 0
    for reader in ("1.0", "1.1", "1.2", "1.3", "1.4"):
        cells = manifest["versions"][reader]["forward"]["1.5"]
        for fname, cell in cells.items():
            if fname in only_15:
                assert cell["outcome"] == "refused", (reader, fname, cell)
            else:
                assert cell["outcome"] in ("loads_identical", "loaded"), (reader, fname, cell)
            checked += 1
    assert (
        "unknown kind 'logical'"
        in manifest["versions"]["1.4"]["forward"]["1.5"]["logical_constraints.json"]["error"]
    )
    assert checked >= 5 * 15
