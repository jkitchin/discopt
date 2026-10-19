"""Schema 1.6: ``None`` / ``Ellipsis`` index kinds and a disjunction as a disjunct
member (plan WP-3b-1).

Each is read only from a document of minor >= 6: in an older document no writer
produced it, so it is refused by name rather than loaded. The round trips
themselves are covered by the golden fixtures (``v1.6/index_none_ellipsis``,
``v1.6/nested_disjunction``) and ``test_serialize_builders.py``.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import serialize_fixture_models as reg
from discopt.serialize import SerializationError, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures" / "v1.6"


def _doc(case):
    return json.loads((FIXTURES / f"{case}.json").read_text(encoding="utf-8"))


def _indices(doc):
    out, stack = [], [n["index"] for n in doc["nodes"] if n["op"] == "index"]
    while stack:
        ix = stack.pop()
        out.append(ix["k"])
        if ix["k"] == "tuple":
            stack.extend(ix["v"])
    return out


def test_the_fixtures_carry_the_new_encodings():
    """§6: the gate tests below are vacuous unless the documents use them."""
    kinds = _indices(_doc("index_none_ellipsis"))
    assert "none" in kinds and "ellipsis" in kinds
    member = _doc("nested_disjunction")["rows"][0]["disjuncts"][0][0]
    assert set(member) == {"disjunction"}


@pytest.mark.parametrize("case", ["index_none_ellipsis", "nested_disjunction"])
def test_the_fixture_reloads_to_the_model_that_wrote_it(case):
    model, _kwargs = reg.CASES[case]()
    text = (FIXTURES / f"{case}.json").read_text(encoding="utf-8")
    assert dumps(loads(text), provenance=False) == dumps(model, provenance=False)


@pytest.mark.parametrize(
    "case, match",
    [
        ("index_none_ellipsis", "1.5 document carries a 'none' index"),
        ("nested_disjunction", "1.5 document nests a disjunction as a disjunct member"),
    ],
)
def test_a_1_6_encoding_in_an_older_document_is_refused(case, match):
    doc = _doc(case)
    doc["schema"] = "discopt.model/1.5"
    with pytest.raises(SerializationError, match=match):
        loads(json.dumps(doc))


def test_an_ellipsis_index_in_an_older_document_is_refused():
    doc = _doc("index_none_ellipsis")
    for node in doc["nodes"]:
        if node["op"] == "index":
            node["index"] = {"k": "ellipsis"}
            break
    doc["schema"] = "discopt.model/1.4"
    with pytest.raises(SerializationError, match="'ellipsis' index"):
        loads(json.dumps(doc))


def test_a_new_index_kind_carries_no_other_key():
    doc = _doc("index_none_ellipsis")
    node = next(
        n for n in doc["nodes"] if n["op"] == "index" and "none" in _indices({"nodes": [n]})
    )
    stack = [node["index"]]
    while stack:
        ix = stack.pop()
        if ix["k"] == "none":
            ix["v"] = 0
            break
        if ix["k"] == "tuple":
            stack.extend(ix["v"])
    with pytest.raises(SerializationError, match="unknown key"):
        loads(json.dumps(doc))
