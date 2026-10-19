"""Schema 1.7: Disjunct-block state (plan WP-3b-2).

A ``block_indicator`` row (an indicator row carrying its nested block's
``parent`` and/or an OR block's ``semantics``), a ``nested_disjunction`` row (a
disjunction row carrying the ``parent_indicator`` of the block it is nested in)
and the ``_gdp_blocks`` state record (block members, local variables) are read
only from a document of minor >= 7: no earlier writer produced them, so in an
older document each is refused by name rather than loaded. They are row *kinds*,
not keys on the old kinds, so that the readers before #1780 -- which skip an
unknown nested key -- refuse them too (recorded in the forward matrix; see
``test_every_older_reader_refuses_block_state``).
"""

from __future__ import annotations

import json
import pathlib
import pickle
import warnings

import pytest
import serialize_fixture_models as reg
from discopt.serialize import SerializationError, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())
CASES = ("gdp_nested_blocks", "gdp_or_block")


def _doc(case):
    return json.loads((FIXTURES / "v1.7" / f"{case}.json").read_text(encoding="utf-8"))


def _older(doc, minor=6):
    doc["schema"] = f"discopt.model/1.{minor}"
    return json.dumps(doc)


def _kinds(doc):
    return [r["kind"] for r in doc["rows"]]


def test_the_fixtures_carry_the_new_encodings():
    """§6: the gate tests below are vacuous unless the documents use them."""
    nested = _doc("gdp_nested_blocks")
    assert _kinds(nested).count("block_indicator") == 4
    assert _kinds(nested).count("nested_disjunction") == 1
    assert all("parent" in r for r in nested["rows"] if r["kind"] == "block_indicator")
    blocks = nested["state"]["_gdp_blocks"]
    assert len(blocks["members"]) == 2
    assert any(d["local"] for d in blocks["disjuncts"])
    rows = _doc("gdp_or_block")["rows"]
    assert [r.get("semantics") for r in rows if r["kind"] == "block_indicator"] == ["OR", "OR"]
    # A row outside any nested or OR block is written as at 1.6.
    assert "indicator" in _kinds(nested)


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_reloads_to_the_model_that_wrote_it(case):
    model, _kwargs = reg.CASES[case]()
    text = (FIXTURES / "v1.7" / f"{case}.json").read_text(encoding="utf-8")
    assert dumps(loads(text), provenance=False) == dumps(model, provenance=False)


@pytest.mark.parametrize("case", CASES)
def test_the_round_trip_solves_as_pickle_does(case):
    model, _kwargs = reg.CASES[case]()
    loaded = loads(dumps(model, provenance=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = pickle.loads(pickle.dumps(model)).solve()
        got = loaded.solve()
    assert (got.status, got.node_count) == (ref.status, ref.node_count)
    assert got.objective == pytest.approx(ref.objective, abs=1e-9)


def test_a_local_variable_still_refuses_an_escaped_use():
    loaded = loads(json.dumps(_doc("gdp_nested_blocks")))
    w = next(v for v in loaded._variables if v.name == "w")
    loaded.subject_to(w <= 5.0)
    with pytest.raises(ValueError, match="declared local to disjunct"):
        loaded.validate()


def test_every_older_reader_refuses_block_state():
    """The shipped readers, run at generation: none loads a 1.7 block document.

    The point of the new row kinds. A ``loaded`` cell would be allowed by the
    matrix test (nothing to compare: the case is unbuildable there) but would
    mean an OR block read as select-one, so it is pinned out here.
    """
    seen = 0
    for reader, entry in MANIFEST["versions"].items():
        if reader == "1.7":
            continue
        for case in CASES:
            assert entry["forward"]["1.7"][f"{case}.json"]["outcome"] == "refused", reader
            seen += 1
    assert seen == 2 * (len(MANIFEST["versions"]) - 1)


@pytest.mark.parametrize("case", CASES)
def test_a_block_indicator_row_in_an_older_document_is_refused(case):
    doc = _doc(case)
    doc["state"].pop("_gdp_blocks", None)
    doc["rows"] = [r for r in doc["rows"] if r["kind"] != "nested_disjunction"]
    with pytest.raises(SerializationError, match="1.6 document carries a 'block_indicator'"):
        loads(_older(doc))


def test_a_nested_disjunction_row_in_an_older_document_is_refused():
    doc = _doc("gdp_nested_blocks")
    doc["state"].pop("_gdp_blocks")
    for row in doc["rows"]:
        if row["kind"] == "block_indicator":
            row["kind"] = "indicator"
            row.pop("parent")
    with pytest.raises(SerializationError, match="1.6 document carries a 'nested_disjunction'"):
        loads(_older(doc))


def test_block_state_in_an_older_document_is_refused():
    doc = _doc("gdp_nested_blocks")
    doc["rows"] = [r for r in doc["rows"] if r["kind"] != "nested_disjunction"]
    for row in doc["rows"]:
        if row["kind"] == "block_indicator":
            row["kind"] = "indicator"
            row.pop("parent")
    # The specific refusal, not the generic unknown-key one, which names the key too.
    with pytest.raises(SerializationError, match="1.6 document carries a '_gdp_blocks' state"):
        loads(_older(doc))


def test_a_block_key_on_a_plain_indicator_row_is_refused():
    doc = _doc("gdp_or_block")
    for row in doc["rows"]:
        if row["kind"] == "block_indicator":
            row["kind"] = "indicator"
    with pytest.raises(SerializationError, match="/rows\\[0\\]/semantics"):
        loads(json.dumps(doc))


def test_a_parent_indicator_on_a_plain_disjunction_row_is_refused():
    doc = _doc("gdp_nested_blocks")
    for row in doc["rows"]:
        if row["kind"] == "nested_disjunction":
            row["kind"] = "disjunction"
    with pytest.raises(SerializationError, match="carries 'parent_indicator'"):
        loads(json.dumps(doc))


def test_a_block_indicator_without_block_state_is_refused():
    doc = _doc("gdp_or_block")
    doc["rows"][0].pop("semantics")
    with pytest.raises(SerializationError, match="has neither 'parent' nor 'semantics'"):
        loads(json.dumps(doc))


def test_a_nested_disjunction_without_its_parent_is_refused():
    doc = _doc("gdp_nested_blocks")
    for row in doc["rows"]:
        if row["kind"] == "nested_disjunction":
            row.pop("parent_indicator")
    with pytest.raises(SerializationError, match="without 'parent_indicator'"):
        loads(json.dumps(doc))


def test_an_explicit_select_one_on_a_block_row_is_refused():
    """The writer omits the default, so an explicit one has no writer."""
    doc = _doc("gdp_or_block")
    doc["rows"][0]["semantics"] = "SELECT_ONE"
    with pytest.raises(SerializationError, match="row 0 writes the default semantics SELECT_ONE"):
        loads(json.dumps(doc))
