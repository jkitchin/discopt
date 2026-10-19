"""Schema 1.7: Disjunct-block state (plan WP-3b-2).

A nested block's ``parent``, an OR block's ``semantics`` on its indicator rows, a
nested ``Model.disjunction``'s ``parent_indicator`` and the ``_gdp_blocks`` state
record (block members, local variables) are read only from a document of minor
>= 7: no earlier writer produced them, so in an older document each is refused
by name rather than loaded. The round trips are also covered by the golden
fixtures (``v1.7/gdp_nested_blocks``, ``v1.7/gdp_or_block``) and
``test_serialize_builders.py``.
"""

from __future__ import annotations

import json
import pickle
import warnings

import pytest
import serialize_fixture_models as reg
from discopt.serialize import SerializationError, dumps, loads


def _doc(case):
    model, _kwargs = reg.CASES[case]()
    return json.loads(dumps(model, provenance=False))


def _older(doc, minor=6):
    doc["schema"] = f"discopt.model/1.{minor}"
    return json.dumps(doc)


def test_the_cases_carry_the_new_encodings():
    """§6: the gate tests below are vacuous unless the documents use them."""
    nested = _doc("gdp_nested_blocks")
    assert any("parent" in r for r in nested["rows"] if r["kind"] == "indicator")
    assert any("parent_indicator" in r for r in nested["rows"] if r["kind"] == "disjunction")
    blocks = nested["state"]["_gdp_blocks"]
    assert len(blocks["members"]) == 2
    assert any(d["local"] for d in blocks["disjuncts"])
    rows = _doc("gdp_or_block")["rows"]
    assert [r.get("semantics") for r in rows if r["kind"] == "indicator"] == ["OR", "OR"]


@pytest.mark.parametrize("case", ["gdp_nested_blocks", "gdp_or_block"])
def test_the_round_trip_matches_pickle(case):
    model, _kwargs = reg.CASES[case]()
    text = dumps(model, provenance=False)
    loaded = loads(text)
    assert dumps(loaded, provenance=False) == text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = pickle.loads(pickle.dumps(model)).solve()
        got = loaded.solve()
    assert (got.status, got.node_count) == (ref.status, ref.node_count)
    assert got.objective == pytest.approx(ref.objective, abs=1e-9)


def test_a_local_variable_still_refuses_an_escaped_use():
    model, _kwargs = reg.CASES["gdp_nested_blocks"]()
    loaded = loads(dumps(model, provenance=False))
    w = next(v for v in loaded._variables if v.name == "w")
    loaded.subject_to(w <= 5.0)
    with pytest.raises(ValueError, match="declared local to disjunct"):
        loaded.validate()


def test_a_parent_in_an_older_document_is_refused():
    doc = _doc("gdp_nested_blocks")
    doc["state"].pop("_gdp_blocks")
    for row in doc["rows"]:
        row.pop("parent_indicator", None)
    with pytest.raises(SerializationError, match=r"produces \(/rows\[2\]/parent,"):
        loads(_older(doc))


def test_a_semantics_in_an_older_document_is_refused():
    with pytest.raises(SerializationError, match=r"produces \(/rows\[0\]/semantics,"):
        loads(_older(_doc("gdp_or_block")))


def test_a_parent_indicator_in_an_older_document_is_refused():
    doc = _doc("gdp_nested_blocks")
    doc["state"].pop("_gdp_blocks")
    for row in doc["rows"]:
        row.pop("parent", None)
    with pytest.raises(SerializationError, match=r"produces \(/rows\[7\]/parent_indicator\)"):
        loads(_older(doc))


def test_block_state_in_an_older_document_is_refused():
    doc = _doc("gdp_nested_blocks")
    for row in doc["rows"]:
        row.pop("parent", None)
        row.pop("parent_indicator", None)
    with pytest.raises(SerializationError, match="_gdp_blocks"):
        loads(_older(doc))


def test_an_explicit_select_one_on_an_indicator_row_is_refused():
    """The writer omits the default, so an explicit one has no writer."""
    doc = _doc("gdp_or_block")
    for row in doc["rows"]:
        if row["kind"] == "indicator":
            row["semantics"] = "SELECT_ONE"
    with pytest.raises(SerializationError, match="row 0 writes the default semantics SELECT_ONE"):
        loads(json.dumps(doc))
