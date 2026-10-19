"""What a reader does with a document's version and with keys it does not know (WP-2).

Plan: ``docs/dev/serialization-completion-plan.md`` §3.2 (DoD D3), owner decision O3.

Before WP-2 every unknown top-level key was ignored -- the 1.1 trick that let a 1.0
reader open a document carrying ``provenance``. That is a silent drop the moment a
key can carry model content, so now:

* at a minor this reader knows, an unknown key is refused (no writer produced it);
* at a newer minor, an unknown key, node kind or row kind is refused with a message
  that says to upgrade, and a document of known keys loads with a
  ``SchemaVersionWarning``;
* a top-level ``x-`` key is an annotation: never read, never refused.

Each test below fails on the pre-WP-2 reader, which loaded every one of these
documents without a word.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import serialize
from discopt.modeling.core import SolveResult
from discopt.result_io import (
    _RESULT_KEYS,
    SCHEMA_VERSION,
    deserialize_result,
    serialize_result,
)
from discopt.serialize import (
    SCHEMA,
    SCHEMA_MINOR,
    SchemaVersionWarning,
    SerializationError,
    dumps,
    loads,
)

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
FUTURE = f"discopt.model/{serialize.SCHEMA_MAJOR}.{SCHEMA_MINOR + 1}"


def _model():
    m = dm.Model("versioning")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=4.0)
    z = m.binary("z")
    m.minimize(dm.exp(x[0]) + x[1] ** 2 - z)
    m.subject_to(x[0] + x[1] >= 1.0, name="cover")
    m.set_initial_point({x: [1.0, 1.0]})
    return m


def _doc(schema: str = SCHEMA) -> dict:
    doc = json.loads(dumps(_model()))
    doc["schema"] = schema
    return doc


def _loads_quietly(doc: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SchemaVersionWarning)
        return loads(json.dumps(doc))


# ── the key tables are complete ────────────────────────────────────────────


def test_schema_minor_is_the_one_schema_names():
    assert SCHEMA == f"discopt.model/{serialize.SCHEMA_MAJOR}.{SCHEMA_MINOR}"


def test_every_key_the_writer_emits_is_listed():
    """A key added to ``dumps`` without its table entry would make every document
    this writer produces unreadable by this reader."""
    m = _model()
    result = m.solve(time_limit=30)
    doc = json.loads(dumps(m, result=result))
    assert {"provenance", "solution"} <= set(doc)
    assert set(doc) <= set(serialize._TOP_LEVEL_KEYS)
    assert set(doc["state"]) <= set(serialize._STATE_KEYS)
    # Every carried "state" attribute, including the ones written only when present.
    carried = (
        set(serialize._STATE_PLAIN)
        | set(serialize._STATE_AS_SORTED_LIST)
        | set(serialize._STATE_BESPOKE)
        | set(serialize._LAZY_PLAIN)
        | {"nl_source", "_gams_initial_values"}
        # 1.7: `_gdp_block_members` and the attached disjuncts' local variables.
        | {"_gdp_blocks"}
    )
    assert carried == set(serialize._STATE_KEYS)


def test_every_golden_fixture_key_is_listed_at_or_before_its_minor():
    """The tables' minors are what WP-8's docs are generated from; check them against
    what each historical writer actually wrote."""
    checked = 0
    for vdir in sorted(FIXTURES.iterdir()):
        if not vdir.is_dir():
            continue
        minor = int(vdir.name.split(".")[1])
        for path in vdir.glob("*.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for key in doc:
                assert serialize._TOP_LEVEL_KEYS[key] <= minor, (path, key)
            for key in doc.get("state") or {}:
                assert serialize._STATE_KEYS[key] <= minor, (path, key)
            checked += 1
    assert checked >= 60


def test_nothing_is_deprecated_yet():
    assert serialize._DEPRECATED_KEYS == {}


# ── a minor this reader knows ──────────────────────────────────────────────


@pytest.mark.parametrize("schema", [SCHEMA, "discopt.model/1.2", "discopt.model/1"])
def test_unknown_key_at_known_minor_is_refused(schema):
    doc = _doc(schema)
    doc["constraints"] = [{"body": "x <= 3"}]  # plausible, and dropped before WP-2
    with pytest.raises(SerializationError, match=r"\['constraints'\].*no writer.*'x-'"):
        loads(json.dumps(doc))


def test_unknown_state_key_at_known_minor_is_refused():
    doc = _doc()
    doc["state"]["_bounds_override"] = {"x": [0, 1]}
    with pytest.raises(SerializationError, match=r"'state' key\(s\) \['_bounds_override'\]"):
        loads(json.dumps(doc))


@pytest.mark.parametrize("schema", [SCHEMA, FUTURE])
def test_an_x_prefixed_key_is_an_annotation(schema):
    """O3's escape hatch: ignored at any minor, and not carried by a re-save."""
    doc = _doc(schema)
    doc["x-reviewed-by"] = {"who": "a tool", "ok": True}
    model = _loads_quietly(doc)
    assert dumps(model, provenance=False) == dumps(_model(), provenance=False)
    assert "x-reviewed-by" not in json.loads(dumps(model))


def test_the_x_prefix_is_top_level_only():
    doc = _doc()
    doc["state"]["x-note"] = "hi"
    with pytest.raises(SerializationError, match="x-note"):
        loads(json.dumps(doc))


# ── a newer minor ──────────────────────────────────────────────────────────


def test_future_minor_with_known_keys_warns_and_loads():
    doc = _doc(FUTURE)
    with pytest.warns(SchemaVersionWarning, match=r"newer format.*upgrade discopt") as rec:
        model = loads(json.dumps(doc))
    assert dumps(model, provenance=False) == dumps(_model(), provenance=False)
    # Pointed at the caller's line, not at discopt's internals.
    assert rec[0].filename == __file__


def test_unknown_key_at_future_minor_is_refused_with_upgrade_hint():
    doc = _doc(FUTURE)
    doc["logic"] = []
    with pytest.raises(SerializationError, match=r"newer than this reader.*\['logic'\].*Upgrade"):
        loads(json.dumps(doc))


def test_unknown_state_key_at_future_minor_is_refused_with_upgrade_hint():
    doc = _doc(FUTURE)
    doc["state"]["_new_mark"] = 1
    with pytest.raises(SerializationError, match=r"newer than this reader.*_new_mark.*Upgrade"):
        loads(json.dumps(doc))


def test_unknown_node_kind_at_future_minor_names_the_upgrade():
    """The decode-level refusal already existed; a newer document adds the remedy."""
    doc = _doc(FUTURE)
    doc["nodes"][0] = {"op": "lambertw", "a": [1]}
    with pytest.raises(SerializationError, match=r"newer than this reader.*upgrade discopt"):
        loads(json.dumps(doc))
    # At a known minor the same node is malformed, and says nothing about upgrading.
    doc["schema"] = SCHEMA
    with pytest.raises(SerializationError) as info:
        loads(json.dumps(doc))
    assert "upgrade" not in str(info.value)


def test_unknown_row_kind_at_future_minor_names_the_upgrade():
    doc = _doc(FUTURE)
    doc["rows"].append({"kind": "quantified", "name": "l0"})
    with pytest.raises(SerializationError, match=r"newer than this reader.*upgrade discopt"):
        loads(json.dumps(doc))


def test_a_newer_document_can_be_refused_outright_by_the_caller():
    doc = _doc(FUTURE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SchemaVersionWarning)
        with pytest.raises(SchemaVersionWarning):
            loads(json.dumps(doc))


# ── the embedded result has the same rule ──────────────────────────────────


def _result_dict() -> dict:
    r = SolveResult(status="optimal", objective=1.0, bound=1.0, x={"x": np.array([1.0, 2.0])})
    return serialize_result(r)


def test_result_keys_cover_what_serialize_result_writes():
    d = _result_dict()
    assert set(d) <= _RESULT_KEYS
    m = _model()
    full = serialize_result(m.solve(time_limit=30), provenance=True, options={"time_limit": 30})
    assert set(full) <= _RESULT_KEYS
    assert {"provenance", "solve_options"} <= set(full)


def test_result_unknown_key_is_a_named_refusal():
    d = _result_dict()
    d["certificate_v2"] = {"ok": True}
    with pytest.raises(SerializationError, match=r"\['certificate_v2'\].*refusing"):
        deserialize_result(d)
    d["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(SerializationError, match=r"newer than this reader.*upgrade"):
        deserialize_result(d)


def test_result_unknown_key_inside_a_document_is_refused():
    m = _model()
    doc = json.loads(dumps(m, result=m.solve(time_limit=30)))
    doc["solution"]["dual_bound_proof"] = []
    with pytest.raises(SerializationError, match="dual_bound_proof"):
        loads(json.dumps(doc))


def test_versioning_probe_fired():
    """§6: the parametrized refusals above actually had documents to refuse."""
    doc = _doc()
    assert isinstance(loads(json.dumps(doc)), dm.Model)
    doc["bogus"] = 1
    with pytest.raises(SerializationError):
        loads(json.dumps(doc))
