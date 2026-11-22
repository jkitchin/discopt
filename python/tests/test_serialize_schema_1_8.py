"""Schema 1.8: ``int_default_sides`` on a variable record (#2281).

``Model.integer`` fills an unwritten side with the fallback ``0`` / ``1e6`` and
records which sides it filled; ``Model._withhold_default_box_certificate`` reads
that record so a point on an invented side is never certified. A document that
dropped it would reload the fallback as a written bound and certify that point
again, so the record travels with the variable from 1.8 on. It is read only from a
document of minor >= 8; in an older document it is refused by name.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import warnings

import discopt.modeling as dm
import pytest
import serialize_fixture_models as reg
from discopt.serialize import SerializationError, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())
CASE = "integer_default_box"


def _doc():
    return json.loads((FIXTURES / "v1.8" / f"{CASE}.json").read_text(encoding="utf-8"))


def _by_name(doc):
    return {v["name"]: v for v in doc["variables"]}


def _unbounded():
    m = dm.Model("u1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = m.integer("x", lb=0)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(-x - y)
    m.subject_to(x - 2 * y >= -3)
    return m


def test_the_fixture_carries_the_new_encoding():
    """§6: the gate tests below are vacuous unless the document uses the key."""
    variables = _by_name(_doc())
    assert variables["n"]["int_default_sides"] == ["ub"]
    assert variables["k"]["int_default_sides"] == ["lb", "ub"]
    # A written ``ub=1e6`` is not a default, and its record is what 1.7 wrote.
    assert "int_default_sides" not in variables["e"]
    assert variables["e"]["ub"] == variables["n"]["ub"]


def test_the_fixture_reloads_to_the_model_that_wrote_it():
    model, _kwargs = reg.CASES[CASE]()
    text = (FIXTURES / "v1.8" / f"{CASE}.json").read_text(encoding="utf-8")
    assert dumps(loads(text), provenance=False) == dumps(model, provenance=False)
    sides = {v.name: v._int_default_sides for v in loads(text)._variables}
    assert sides == {"n": {"ub"}, "k": {"lb", "ub"}, "e": frozenset()}


@pytest.mark.parametrize("copy", ["serialize", "pickle"])
def test_a_reloaded_model_still_withholds_the_certificate(copy):
    """The #2281 model, unbounded below: ``feasible`` with no bound after a reload,
    exactly as before it (the C-6 fallback ``ub=1e6`` is not a written bound)."""
    m = _unbounded()
    loaded = loads(dumps(m)) if copy == "serialize" else pickle.loads(pickle.dumps(m))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = m.solve(time_limit=20)
        got = loaded.solve(time_limit=20)
    for r in (ref, got):
        assert r.status == "feasible" and r.gap_certified is False and r.bound is None


def test_every_older_reader_refuses_or_reads_the_fallback_as_its_own_writer_does():
    """The shipped readers, run at generation, on the 1.8 document.

    1.5-1.7 refuse the unknown key. 1.0-1.4 skip an unknown nested key, so they
    load the fallback box as a written bound: the model each of them writes for
    this case itself (``loads_identical``), since none of them records the defaults.
    """
    seen = 0
    for reader, entry in MANIFEST["versions"].items():
        if reader == "1.8":
            continue
        cell = entry["forward"]["1.8"][f"{CASE}.json"]
        minor = int(reader.split(".")[1])
        assert cell["outcome"] == ("refused" if minor >= 5 else "loads_identical"), reader
        if minor >= 5:
            assert "int_default_sides" in cell["error"], (reader, cell)
        seen += 1
    assert seen == len(MANIFEST["versions"]) - 1


def test_the_record_in_an_older_document_is_refused():
    doc = _doc()
    doc["schema"] = "discopt.model/1.7"
    with pytest.raises(SerializationError, match="1.7 document carries an 'int_default_sides'"):
        loads(json.dumps(doc))


def test_an_older_document_reads_the_fallback_as_written():
    """A <= 1.7 document never recorded the defaults; its bounds are all as written."""
    text = (FIXTURES / "v1.7" / f"{CASE}.json").read_text(encoding="utf-8")
    assert all(not v._int_default_sides for v in loads(text)._variables)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda v: v.update(int_default_sides=[]), "non-empty list"),
        (lambda v: v.update(int_default_sides=["ub", "ub"]), "distinct"),
        (lambda v: v.update(int_default_sides=["upper"]), "'lb' / 'ub'"),
        (lambda v: v.update(int_default_sides="ub"), "int_default_sides of 'n'"),
        (lambda v: v.update(type="continuous"), "only an integer variable"),
    ],
)
def test_a_malformed_record_is_refused(mutate, match):
    doc = _doc()
    mutate(_by_name(doc)["n"])
    with pytest.raises(SerializationError, match=match):
        loads(json.dumps(doc))
