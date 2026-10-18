"""Golden fixtures for every shipped schema version, and the compatibility matrix (WP-1).

Plan: ``docs/dev/serialization-completion-plan.md`` §3.3 / WP-1 (DoD D1, D2).

``python/tests/data/serialize_fixtures/v<version>/`` holds documents written by the
discopt that last wrote each schema version -- a historical worktree built with its
own Rust extension, with the imported code proven to come from that tree
(``scripts/serialize_fixture_gen.py``) -- for the fixed cases in
``serialize_fixture_models.py``. ``MANIFEST.json`` records, per version, the
generating commit, the asserted ``SCHEMA`` string, a SHA-256 per document, cases
that version could not build, and the *forward* cells: what each older reader did
with every newer version's documents.

* **Backward** (old document -> this reader) is tested live: every fixture loads,
  and re-dumped by today's writer it equals the same case built today, except for
  content its version did not yet write (``_WRITTEN_FROM``) -- an older document
  carries less, never something else.
* **Forward** (new document -> old reader) was recorded at generation by running
  the old reader; the test asserts the record is complete and contains no silent
  misread (``loads_different``).

Shipped fixtures are immutable (plan §0.3.11): ``test_fixture_files_are_unchanged``
fails if one is edited. A new schema version adds its own ``v1.N/`` and manifest
entry in the PR that bumps ``SCHEMA``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pathlib

import discopt.modeling as dm
import pytest
import serialize_fixture_models as reg
from discopt.serialize import SCHEMA, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())
VERSIONS = list(MANIFEST["versions"])


def _key(label: str) -> tuple:
    return tuple(int(p) for p in label.split("."))


#: Content a version did not write yet, keyed by the first version that writes it.
#: Comparing an older document with today's build drops exactly these, nothing else.
#: 1.3 added the registered-atom tag on nodes (#1248 A); 1.4 added ``state.nl_source``
#: (a ``from_nl`` model's parse marker and digest-checked source path).
_WRITTEN_FROM = {
    (1, 3): "node atom tags",
    (1, 4): "state.nl_source",
}


def _normalize(doc: dict, written_by: str) -> dict:
    doc = json.loads(json.dumps(doc))
    doc.pop("solution", None)  # the embedded result is checked on its own
    source = doc.get("state", {}).get("nl_source")
    if source and "path" in source:
        # Where the .nl was read from is a fact about the machine that wrote the
        # document (CI checks the tree out elsewhere); the digest identifies the
        # content and stays compared, as does the file name.
        source["path"] = pathlib.PurePath(source["path"]).name
    if _key(written_by) < (1, 4):
        doc.get("state", {}).pop("nl_source", None)
    if _key(written_by) < (1, 3):
        for node in doc["nodes"]:
            node.pop("atom", None)
    return doc


@functools.lru_cache(maxsize=None)
def _built_today(case: str) -> str:
    model, _, fixed = reg.build(case)
    if fixed is None:
        return dumps(model, provenance=False)
    var, value = fixed
    with var.fixed(value):
        return dumps(model, provenance=False)


def _fixtures():
    for version in VERSIONS:
        for case, entry in MANIFEST["versions"][version]["cases"].items():
            if "file" in entry:
                yield pytest.param(version, case, entry, id=f"v{version}-{case}")


FIXTURE_PARAMS = list(_fixtures())


def _read(version: str, entry: dict) -> str:
    return (FIXTURES / f"v{version}" / entry["file"]).read_text(encoding="utf-8")


# ── the manifest and the files agree ──────────────────────────────────────


def test_manifest_lists_every_version_directory():
    dirs = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())
    assert dirs == [f"v{v}" for v in VERSIONS]
    assert VERSIONS == sorted(VERSIONS, key=_key)


def test_manifest_markers_match_versions():
    """Each version's fixtures were written by a reader that reported that schema.

    1.0 spelled its identifier with a bare major (``discopt.model/1``).
    """
    for version, entry in MANIFEST["versions"].items():
        expected = "discopt.model/1" if version == "1.0" else f"discopt.model/{version}"
        assert entry["schema"] == expected, version
        assert len(entry["commit"]) == 40, version


def test_current_schema_has_fixtures():
    """A ``SCHEMA`` bump must ship its own fixtures in the same PR (plan §0.3.10)."""
    current = SCHEMA.split("/", 1)[1]
    assert current in VERSIONS, f"no golden fixtures for the current schema {SCHEMA}"


@pytest.mark.parametrize("version, case, entry", FIXTURE_PARAMS)
def test_fixture_files_are_unchanged(version, case, entry):
    text = _read(version, entry)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["sha256"], (
        "a shipped golden fixture was edited; fixtures are immutable"
    )
    assert json.loads(text)["schema"] == MANIFEST["versions"][version]["schema"]


def test_corpus_inputs_are_unchanged():
    """The ``from_nl`` cases read today's data files; they must be the files the
    fixtures were written from, or the comparison below means nothing."""
    checked = 0
    for version in VERSIONS:
        for case, entry in MANIFEST["versions"][version]["cases"].items():
            if "input_sha256" in entry:
                data = pathlib.Path(reg.NL_FILES[case]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["input_sha256"], (version, case)
                checked += 1
    assert checked >= 3 * len(VERSIONS)


def test_unbuildable_cases_are_recorded_not_skipped():
    """Every registry case is either a fixture or an explicit ``unbuildable`` entry."""
    for version, entry in MANIFEST["versions"].items():
        assert set(entry["cases"]) == set(reg.CASES), version
        for case, record in entry["cases"].items():
            assert ("file" in record) != ("unbuildable" in record), (version, case)
    # register_function arrived in 1.1, so 1.0 cannot build the atom case.
    assert (
        "No module named 'discopt.operators'"
        in (MANIFEST["versions"]["1.0"]["cases"]["registered_atom"]["unbuildable"])
    )


# ── backward: every old document loads into the model it describes ────────


@pytest.mark.parametrize("version, case, entry", FIXTURE_PARAMS)
def test_every_golden_fixture_loads(version, case, entry):
    # Built FIRST: a registered atom's tag is re-derived on load only for a name the
    # loading process has registered, and building the case is what registers it --
    # as a program registers its atoms at import, before it loads anything.
    expected = _normalize(json.loads(_built_today(case)), written_by=version)
    model = loads(_read(version, entry))
    got = _normalize(json.loads(dumps(model, provenance=False)), written_by="99.0")
    assert got == expected


@pytest.mark.parametrize("version, case, entry", FIXTURE_PARAMS)
def test_golden_fixture_reload_is_stable(version, case, entry):
    once = dumps(loads(_read(version, entry)), provenance=False)
    assert dumps(loads(once), provenance=False) == once


def test_fixtures_with_a_result_carry_it():
    checked = 0
    for version in VERSIONS:
        entry = MANIFEST["versions"][version]["cases"]["with_result"]
        model = loads(_read(version, entry))
        result = model.saved_result
        assert result is not None and result.status == "optimal", version
        assert set(result.x) == {v.name for v in model._variables}, version
        checked += 1
    assert checked == len(VERSIONS)


def test_fixture_probe_fired():
    """§6: the parametrized tests above must have had something to load."""
    assert len(FIXTURE_PARAMS) >= 10 * len(VERSIONS)
    assert isinstance(loads(_read(*FIXTURE_PARAMS[0].values[::2])), dm.Model)


# ── forward: what older readers do with newer documents ───────────────────


def test_forward_matrix_is_complete():
    """Every older reader has a recorded cell for every newer version's documents."""
    cells = 0
    for i, reader in enumerate(VERSIONS):
        forward = MANIFEST["versions"][reader].get("forward", {})
        for writer in VERSIONS[i + 1 :]:
            cases = MANIFEST["versions"][writer]["cases"].values()
            written = {e["file"] for e in cases if "file" in e}
            assert set(forward.get(writer, {})) == written, (reader, writer)
            cells += len(written)
    assert cells > 0


def test_forward_matrix_has_no_silent_misread():
    """No older reader turned a newer document into a different model.

    ``loads_identical``: the older reader decoded it into exactly what that version
    writes for the case itself. ``refused``: a loud error -- acceptable. ``loaded``:
    the older version could not build the case, so there is nothing to compare;
    allowed only then. ``loads_different`` is the one forbidden outcome.
    """
    seen = 0
    for reader in VERSIONS:
        own = MANIFEST["versions"][reader]["cases"]
        for writer, cells in MANIFEST["versions"][reader].get("forward", {}).items():
            for fname, cell in cells.items():
                seen += 1
                outcome = cell["outcome"]
                assert outcome != "loads_different", (reader, writer, fname, cell)
                assert outcome in {"loads_identical", "refused", "loaded"}, outcome
                if outcome == "loaded":
                    case = fname.removesuffix(".json")
                    assert "unbuildable" in own[case], (reader, writer, fname)
    assert seen > 0
