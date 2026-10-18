"""Fuzzing ``loads``: structure-aware mutations of valid documents, and raw bytes (WP-4, D9).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-4. ``loads`` reads files a
user was handed, so whatever it is given it must either return a ``Model`` or raise
``SerializationError`` -- never another exception, never a hang (a per-example
deadline makes a hang a failure). And a change to the model's content must never be
read back as the original model: a *value* mutation inside ``variables`` / ``nodes``
/ ``rows`` that loads must load into something else.

The base documents are the golden fixtures (every schema version, every case) --
real writer output, not a hand-made template.
"""

from __future__ import annotations

import copy
import json
import pathlib
import warnings

import discopt.modeling as dm
import pytest
import serialize_fixture_models as reg
from discopt.serialize import SerializationError, dumps, loads
from hypothesis import given, settings
from hypothesis import strategies as st

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"

# Register the fixtures' atoms first, as a program does at import before it loads
# anything (and as test_serialize_compat_matrix does): a registered-atom tag is
# re-derived on load only for a name this process has registered, and without the
# registration a tag whose ``arg`` was moved to another node is dropped in BOTH the
# original and the mutant -- identical reloads that are not the reader's fault.
reg.build("registered_atom")
BASES = sorted(FIXTURES.glob("v*/*.json"))
TEXTS = {p: p.read_text(encoding="utf-8") for p in BASES}

#: A hang is a failure: no document should take this long to accept or refuse.
FUZZ = settings(deadline=2000)


def _outcome(text):
    """``Model`` or ``SerializationError``; anything else propagates and fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return loads(text)
        except SerializationError as exc:
            return exc


def _paths(obj, prefix=()):
    """Every (path, value) in a JSON tree, containers included."""
    out = [(prefix, obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _paths(v, (*prefix, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _paths(v, (*prefix, i))
    return out


def _get_parent(doc, path):
    parent = doc
    for step in path[:-1]:
        parent = parent[step]
    return parent


def _retyped(value, draw):
    choices = [None, True, "x", 1.5, -1, [], {}, [value], "nan", 10**12]
    return draw(st.sampled_from([c for c in choices if c != value or type(c) is not type(value)]))


@st.composite
def mutations(draw):
    """``(mutated_text, kind, path, base)``: one mutation of one golden fixture."""
    base = draw(st.sampled_from(BASES))
    doc = json.loads(TEXTS[base])
    paths = [p for p, _ in _paths(doc) if p]
    kind = draw(
        st.sampled_from(
            ["delete", "retype", "int_shift", "swap_op", "truncate_nodes", "dup_name", "nan"]
        )
    )
    path = None
    if kind == "delete":
        path = draw(st.sampled_from(paths))
        parent = _get_parent(doc, path)
        del parent[path[-1]]
    elif kind == "retype":
        path = draw(st.sampled_from(paths))
        parent = _get_parent(doc, path)
        parent[path[-1]] = _retyped(parent[path[-1]], draw)
    elif kind == "int_shift":
        ints = [p for p, v in _paths(doc) if p and type(v) is int]
        if not ints:
            return TEXTS[base], "none", None, base
        path = draw(st.sampled_from(ints))
        parent = _get_parent(doc, path)
        old = parent[path[-1]]
        new = [old + 1, old - 1, -1, -old - 1, True, 10**9]
        parent[path[-1]] = draw(st.sampled_from([v for v in new if v != old or v is True]))
    elif kind == "swap_op":
        nodes = doc["nodes"]
        if not nodes:
            return TEXTS[base], "none", None, base
        i = draw(st.integers(0, len(nodes) - 1))
        path = ("nodes", i, "op")
        ops = ["const", "var", "param", "binop", "call", "sum", "index", "bogus"]
        nodes[i]["op"] = draw(st.sampled_from([o for o in ops if o != nodes[i]["op"]]))
    elif kind == "truncate_nodes":
        if not doc["nodes"]:
            return TEXTS[base], "none", None, base
        k = draw(st.integers(0, len(doc["nodes"]) - 1))
        path = ("nodes", k)
        del doc["nodes"][k:]
    elif kind == "dup_name":
        if len(doc["variables"]) < 1:
            return TEXTS[base], "none", None, base
        doc["variables"].append(copy.deepcopy(doc["variables"][0]))
        path = ("variables", len(doc["variables"]) - 1)
    else:  # a bare NaN token, which is not JSON
        text = json.dumps(doc)
        return text.replace("0.0", "NaN", 1), kind, None, base
    return json.dumps(doc), kind, path, base


@FUZZ
@given(mutations())
def test_mutated_document_never_escapes_serialization_error(case):
    text = case[0]
    outcome = _outcome(text)
    assert isinstance(outcome, (dm.Model, SerializationError)), type(outcome)


#: Mutations that change a *value*: if the document still loads, it must load into a
#: different model. (A deleted key can legitimately mean the default, e.g. a
#: missing optional ``bound_stack``; that is not a silent acceptance.)
VALUE_MUTATIONS = {"retype", "int_shift", "swap_op"}
CONTENT = {"variables", "nodes", "rows"}


@FUZZ
@given(mutations())
def test_mutation_of_content_is_never_silently_accepted(case):
    text, kind, path, base = case
    if kind not in VALUE_MUTATIONS or not path or path[0] not in CONTENT:
        return
    outcome = _outcome(text)
    if isinstance(outcome, SerializationError):
        return
    original = _outcome(TEXTS[base])
    assert dumps(outcome, provenance=False) != dumps(original, provenance=False), (kind, path)


@FUZZ
@given(st.binary(max_size=4096))
def test_random_bytes_are_refused(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    outcome = _outcome(text)
    # Random bytes are never a discopt document.
    assert isinstance(outcome, SerializationError)


@FUZZ
@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda c: st.lists(c) | st.dictionaries(st.text(), c),
        max_leaves=30,
    )
)
def test_arbitrary_json_is_refused_or_loads(value):
    """Any JSON value at all: never an exception other than SerializationError."""
    outcome = _outcome(json.dumps(value))
    assert isinstance(outcome, (dm.Model, SerializationError))


def test_fuzz_probe_fired():
    """§6: the bases load unmodified, so a refusal above is the mutation's doing."""
    assert len(BASES) >= 60
    for text in TEXTS.values():
        assert isinstance(_outcome(text), dm.Model)


@pytest.mark.parametrize("kind", sorted(VALUE_MUTATIONS))
def test_value_mutations_are_drawn(kind):
    """The silent-acceptance property is vacuous unless its mutations occur."""
    seen = []

    @settings(max_examples=60, database=None, derandomize=True, deadline=None)
    @given(mutations())
    def collect(case):
        seen.append(case[1])

    collect()
    assert kind in seen
