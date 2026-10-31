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
    out, failed = [], {}
    for name, build in reg.CASES.items():
        try:
            model, _ = build()
        except Exception as exc:  # noqa: BLE001 - recorded, and asserted below
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        out.append((name, model))
    return out, failed


CASES, UNBUILT = _built_cases()

#: Chunk sizes that make every fixture write many chunks, one record per chunk
#: included; the default (4096) leaves all but the largest in one.
CHUNKS = (1, 2, 5, ser._NodeTable._CHUNK)


def test_every_fixture_case_builds():
    """A case that fails to build drops out of the byte-identity tests below; it
    must not do so silently. A case retired by design (``reg.RETIRED``) must fail
    for exactly its recorded reason."""
    import re

    assert set(UNBUILT) == set(reg.RETIRED), UNBUILT
    for name, why in UNBUILT.items():
        assert re.search(reg.RETIRED[name], why), (name, why)
    assert len(CASES) + len(UNBUILT) == len(reg.CASES) >= 20


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


def _one_call(model) -> str:
    """``json.dumps`` of the document in one call: an ``indent`` takes the
    non-streamed path, so this is independent of the chunking under test."""
    return json.dumps(json.loads(dumps(model, provenance=False, indent=2)), allow_nan=False)


@pytest.mark.parametrize("chunk", CHUNKS)
@pytest.mark.parametrize("name, model", CASES, ids=[n for n, _ in CASES])
def test_dumps_is_one_json_dumps_of_its_document(name, model, chunk, monkeypatch):
    """The streamed bytes equal ``json.dumps`` of the same document in one call,
    however many chunks the node table is written in."""
    monkeypatch.setattr(ser._NodeTable, "_CHUNK", chunk)
    text = dumps(model, provenance=False)
    assert text == _one_call(model)
    assert text == json.dumps(json.loads(text), allow_nan=False)
    indented = dumps(model, provenance=False, indent=2)
    assert indented == json.dumps(json.loads(text), allow_nan=False, indent=2)


@pytest.mark.parametrize("chunk", (1, 64, 1000, ser._NodeTable._CHUNK))
def test_a_corpus_model_is_byte_identical_in_any_number_of_chunks(chunk, monkeypatch):
    """A real model (hda, 4069 nodes) over chunk sizes that split it 1-4069 ways."""
    model = dm.from_nl(str(NL))
    expected = _one_call(model)
    monkeypatch.setattr(ser._NodeTable, "_CHUNK", chunk)
    chunks = []
    flush = ser._NodeTable._flush

    def spy(self):
        flush(self)
        if len(self) > 1000:
            chunks.append(len(self._chunks))

    monkeypatch.setattr(ser._NodeTable, "_flush", spy)
    text = dumps(model, provenance=False)
    assert text == expected
    assert max(chunks) == -(-len(json.loads(text)["nodes"]) // chunk)


def _ignores_arg_model(terms: int):
    """Atoms whose lowering ignores the argument: ``_tag_atom`` adds the argument's
    nodes while the atom's own record is still open, so a flush in between must
    stop before it (or the tag, added last, is dropped from the text)."""
    from discopt.modeling.core import Constant
    from discopt.operators import register_function

    fn = register_function("mem1812_ignores_arg", lambda t: dm.exp(Constant(1.0)), replace=True)
    m = dm.Model(f"open_record_{terms}")
    x = m.continuous("x", shape=(3,), lb=-1.0, ub=1.0)
    arg = x[0]
    for i in range(terms):
        arg = arg + (i + 0.5) * x[(i + 1) % 3]
    m.minimize(fn(arg) + fn(fn(arg * 2.0)) + x[2])
    return m


@pytest.mark.parametrize(
    "chunk, terms",
    [(1, 10), (2, 10), (5, 10), (5, 300), (ser._NodeTable._CHUNK, 1400)],
)
def test_an_open_atom_record_is_not_flushed_before_its_tag(chunk, terms, monkeypatch):
    """#1248's class: a dropped atom tag is sound but silently looser. With 1400
    terms the argument is over 4096 nodes, so the default chunk fills inside the
    open record too."""
    monkeypatch.setattr(ser._NodeTable, "_CHUNK", chunk)
    model = _ignores_arg_model(terms)
    text = dumps(model, provenance=False)
    nodes = json.loads(text)["nodes"]
    if terms == 1400:
        assert len(nodes) > ser._NodeTable._CHUNK
    assert text == _one_call(model)
    assert sum("atom" in nd for nd in nodes) == 3
    assert dumps(loads(text), provenance=False) == text


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


@pytest.mark.parametrize("bad", [memoryview(b"{}"), None, 17, ["{}"]])
def test_a_non_text_document_gets_json_loads_type_error(bad):
    with pytest.raises(TypeError) as exc:
        json.loads(bad)
    with pytest.raises(TypeError) as ours:
        loads(bad)
    assert str(ours.value) == str(exc.value)
    assert str(ours.value).startswith("the JSON object must be str, bytes or bytearray, not ")


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
