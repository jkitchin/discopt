"""Serialization without the whole node table in memory as dicts (#1812).

``dumps`` built the document as one ``dict`` per expression node and then
``json.dumps``-ed it; ``loads`` ``json.loads``-ed the whole text into the same
shape before decoding. On 7-9 M-node models each table cost 2.8-3.7 GB above the
model (measured, #1812). The writer now streams each node record as text, and the
reader keeps the ``nodes`` array as offsets into the text, decoding one record at
a time. The bytes written and the model read back are unchanged; these tests pin
that, and that a malformed document still fails exactly as it did.
"""

from __future__ import annotations

import gzip
import json
import pathlib

import discopt.modeling as dm
import pytest
import serialize_fixture_models as reg
from discopt import serialize as ser
from discopt.serialize import SerializationError, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
LATEST = sorted(
    (p for p in FIXTURES.iterdir() if p.is_dir()),
    key=lambda p: tuple(map(int, p.name[1:].split("."))),
)[-1]
DOCS = sorted(LATEST.glob("*.json"))
NL = pathlib.Path(__file__).parent / "data" / "minlplib_nl" / "hda.nl"


def _built_cases():
    out = []
    for name, build in reg.CASES.items():
        try:
            model, _ = build()
        except Exception:  # noqa: BLE001 - a case this build cannot make is not this test's
            continue
        out.append((name, model))
    return out


CASES = _built_cases()


# ── writer ───────────────────────────────────────────────────────────────────


def test_dumps_streams_the_node_table(monkeypatch):
    """No node dicts accumulate: the table ``dumps`` uses keeps text, not dicts."""
    tables = []
    init = ser._NodeTable.__init__

    def spy(self, *args, **kwargs):
        init(self, *args, **kwargs)
        tables.append(self)

    monkeypatch.setattr(ser._NodeTable, "__init__", spy)
    model = dm.from_nl(str(NL))
    dumps(model, provenance=False)
    main = [t for t in tables if len(t) > 1000]
    assert len(main) == 1, [len(t) for t in tables]
    assert main[0]._stream and main[0].nodes == []


@pytest.mark.parametrize("name, model", CASES, ids=[n for n, _ in CASES])
def test_dumps_is_one_json_dumps_of_its_document(name, model):
    """The streamed bytes equal ``json.dumps`` of the same document in one call."""
    text = dumps(model, provenance=False)
    assert text == json.dumps(json.loads(text), allow_nan=False)
    indented = dumps(model, provenance=False, indent=2)
    assert indented == json.dumps(json.loads(text), allow_nan=False, indent=2)


def test_save_writes_exactly_dumps(tmp_path):
    model = dm.from_nl(str(NL))
    text = dumps(model, provenance=False)
    ser.save(model, tmp_path / "m.json", provenance=False)
    ser.save(model, tmp_path / "m.json.gz", provenance=False)
    assert (tmp_path / "m.json").read_text(encoding="utf-8") == text
    assert gzip.decompress((tmp_path / "m.json.gz").read_bytes()).decode("utf-8") == text
    assert dumps(ser.load(tmp_path / "m.json.gz"), provenance=False) == text


# ── reader ───────────────────────────────────────────────────────────────────


def test_loads_keeps_the_node_table_as_text():
    text = dumps(dm.from_nl(str(NL)), provenance=False)
    doc = ser._parse_document(text)
    assert isinstance(doc["nodes"], ser._NodeSpans)
    assert list(doc["nodes"]) == json.loads(text)["nodes"]
    assert doc["nodes"][17] == json.loads(text)["nodes"][17]


def _outcome(text):
    try:
        return ("model", dumps(loads(text), provenance=False))
    except SerializationError as exc:
        return ("refused", str(exc))


def _mutations(text: str):
    """Valid re-layouts (must load the same) and malformed variants (must fail the same)."""
    nodes_at = text.index('"nodes": [') + len('"nodes": [')
    first_node_end = text.index("}", nodes_at) + 1
    head, tail = text[:nodes_at], text[nodes_at:]
    after_first = text[first_node_end:].split("]", 1)[1]
    return {
        "as written": text,
        "pretty": json.dumps(json.loads(text), indent=1),
        "spaced": " \n " + text.replace(", ", " ,\r\n\t").replace(": ", " :  ") + " \n",
        "bytes utf-8": text.encode("utf-8"),
        "bytes utf-16": text.encode("utf-16"),
        "truncated in nodes": text[: nodes_at + 40],
        "truncated at end": text[:-1],
        "extra data": text + " {}",
        "bom": "\ufeff" + text,
        "trailing comma in nodes": text[:first_node_end] + ", ]" + after_first,
        "NaN in a node": head + '{"op": "const", "v": NaN}, ' + tail,
        "repeated key in a node": head + '{"op": "const", "op": "const", "v": 1.0}, ' + tail,
        "repeated top-level key": text[:-1] + ', "name": "again"}',
        "nodes not an array": text.replace('"nodes": [', '"nodes": "x", "zz": [', 1),
        "nodes not closed": head + tail.replace("]", "", 1),
        "not an object": "[" + text + "]",
        "empty object": "{}",
        "number": "17",
        "key not a string": "{1: 2}",
    }


@pytest.mark.parametrize("doc", DOCS[:8], ids=[p.stem for p in DOCS[:8]])
def test_fast_path_fails_and_loads_exactly_like_json_loads(doc, monkeypatch):
    """Differential: every variant gets the outcome the one-call ``json.loads``
    path gives -- the same model, or the same refusal, message included."""
    text = doc.read_text(encoding="utf-8")
    variants = _mutations(text)
    # The valid layouts take the fast path itself, not the fallback.
    for name in ("as written", "pretty", "spaced"):
        assert isinstance(ser._parse_fast(variants[name])["nodes"], ser._NodeSpans), name
    fast = {name: _outcome(v) for name, v in variants.items()}

    def miss(_text):
        raise ser._FastPathMiss

    monkeypatch.setattr(ser, "_parse_fast", miss)
    slow = {name: _outcome(v) for name, v in variants.items()}
    compared = 0
    for name in variants:
        assert fast[name] == slow[name], name
        compared += 1
    assert compared == len(variants) >= 15
    # The variants are not all one outcome: both arms of the comparison fired.
    kinds = {fast[n][0] for n in variants}
    assert kinds == {"model", "refused"}, kinds
    assert fast["pretty"] == fast["as written"] == fast["spaced"] == fast["bytes utf-16"]


def test_a_document_with_atom_tags_retags_on_the_fast_path():
    """``has_atom`` stands in for a second decode of the table in ``_retag_atoms``.

    Built here, not at import: another test may re-register the atom's name in
    between, and a stale registration is (rightly) not written as a tag.
    """
    try:
        model, _ = reg.CASES["registered_atom"]()
    except Exception as exc:  # noqa: BLE001 - this build cannot make the case
        pytest.skip(f"no registered-atom case in this build: {exc}")
    text = dumps(model, provenance=False)
    assert '"atom"' in text
    assert ser._parse_document(text)["nodes"].has_atom
    assert dumps(loads(text), provenance=False) == text
