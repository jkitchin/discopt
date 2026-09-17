"""FAIR provenance metadata on saved models (:mod:`discopt.provenance`).

Before this, a saved document carried ``discopt.__version__`` and nothing else --
and that field was *written but never read*: ``loads`` validated only the schema
major, so reloading under a different discopt produced no signal, and the rebuilt
model could not say what wrote it. There was no timestamp at all.

Two properties matter more than the field list and are tested hardest here:

* the block is **descriptive only** -- no provenance content, however corrupt,
  may change one coefficient of the model that comes back
  (``test_a_corrupt_provenance_block_cannot_change_the_model``); and
* a version difference is **surfaced, not swallowed** -- the old field's failure
  mode was silence (``test_version_skew_warns_on_load``).
"""

from __future__ import annotations

import json
import warnings

import discopt
import discopt.modeling as dm
import pytest
from discopt.provenance import _git_head, capture, skew_warning
from discopt.serialize import ProvenanceSkewWarning, SerializationError, dumps, load, loads


def _model():
    m = dm.Model("kinetics")
    k = m.continuous("k", shape=(3,), lb=0.0, ub=10.0)
    n = m.integer("n", lb=0, ub=5)
    m.minimize(dm.sum((k - 1.5) ** 2) + n)
    m.subject_to(k[0] + k[1] <= 4.0)
    return m


def _doc(model=None, **kw):
    return json.loads(dumps(model if model is not None else _model(), **kw))


# ── what is recorded ───────────────────────────────────────────────────────


@pytest.mark.smoke
def test_saved_document_records_time_software_and_platform():
    prov = _doc()["provenance"]

    # Checked field by field with a counter, so a block that quietly lost a key
    # cannot pass this as "nothing to compare".
    checked = 0
    for key in ("created", "software", "source_fingerprint", "platform"):
        assert key in prov, f"provenance lost the {key!r} field"
        checked += 1
    for key in ("python", "implementation", "system", "machine"):
        assert prov["platform"][key], f"platform.{key} is empty"
        checked += 1
    assert checked == 8, f"expected 8 field assertions, ran {checked}"

    # `created` is a real UTC ISO-8601 stamp, not a free-form string.
    from datetime import datetime, timezone

    created = datetime.fromisoformat(prov["created"])
    assert created.tzinfo is not None, "created must be timezone-aware"
    assert created.utcoffset() == timezone.utc.utcoffset(None), "created must be UTC"

    assert prov["software"]["version"] == discopt.__version__
    assert prov["software"]["name"] == "discopt"
    assert prov["software"]["license"] == "EPL-2.0"


@pytest.mark.smoke
def test_the_rust_core_version_is_recorded_separately_from_the_python_version():
    """The Rust core is the easy one to lose: it is not what ``pip show`` reports.

    The Expression IR, the ``.nl`` parser and the LP layer live there and version
    independently, so recording only ``discopt.__version__`` would pin the wrong
    half of the solver.
    """
    from discopt import _rust

    prov = _doc()["provenance"]
    assert prov["software"]["rust_core"] == _rust.version()
    # Not an assertion about specific numbers -- just that the two fields are read
    # from different sources, which is the whole reason both are recorded.
    assert prov["software"]["rust_core"] is not None


@pytest.mark.smoke
def test_the_legacy_discopt_version_field_is_still_written():
    """A reader predating provenance looks for the top-level ``discopt`` key."""
    doc = _doc()
    assert doc["discopt"] == discopt.__version__
    assert doc["provenance"]["software"]["version"] == doc["discopt"]


@pytest.mark.smoke
def test_schema_minor_was_bumped_for_the_added_block():
    """Provenance arrived at minor 1, so a document must carry at least that.

    Not pinned to an exact minor: the minor has moved since (1.2 dropped the
    blanket tag over the solution subtree) and will move again. The tripwire for
    the current value lives with the change that sets it, in
    ``test_1302_review_followups.py``; what this test defends is that the block's
    arrival was versioned at all.
    """
    schema = _doc()["schema"]
    major, _, minor = schema.split("/", 1)[1].partition(".")
    assert major == "1"
    assert minor.isdigit() and int(minor) >= 1, schema


# ── it comes back on the model ─────────────────────────────────────────────


@pytest.mark.smoke
def test_provenance_round_trips_onto_the_reloaded_model():
    doc = _doc()
    assert loads(json.dumps(doc)).provenance == doc["provenance"]


@pytest.mark.smoke
def test_provenance_survives_a_file_save_and_load(tmp_path):
    m = _model()
    path = tmp_path / "m.dopt.gz"
    m.save(path)
    assert load(path).provenance["software"]["version"] == discopt.__version__


@pytest.mark.smoke
def test_a_model_built_in_memory_has_no_provenance():
    """Nothing has been written, so there is no document to describe."""
    assert _model().provenance is None


# ── version skew is surfaced, not swallowed ────────────────────────────────


@pytest.mark.smoke
def test_version_skew_warns_on_load():
    """The old field's failure mode was silence. This is the regression guard."""
    doc = _doc()
    doc["provenance"]["software"]["version"] = "0.0.1-ancient"
    with pytest.warns(ProvenanceSkewWarning, match="0.0.1-ancient"):
        model = loads(json.dumps(doc))
    # Warned, but still read as written.
    assert model.provenance["software"]["version"] == "0.0.1-ancient"


@pytest.mark.smoke
def test_rust_core_skew_warns_on_load():
    doc = _doc()
    doc["provenance"]["software"]["rust_core"] = "0.0.1-ancient"
    with pytest.warns(ProvenanceSkewWarning, match="Rust core"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_a_matching_version_does_not_warn():
    doc = _doc()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProvenanceSkewWarning)
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_skew_can_be_escalated_to_an_error():
    """Its own category, so an exact-version pipeline can make skew fatal."""
    doc = _doc()
    doc["provenance"]["software"]["version"] = "0.0.1-ancient"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProvenanceSkewWarning)
        with pytest.raises(ProvenanceSkewWarning):
            loads(json.dumps(doc))


@pytest.mark.smoke
def test_a_document_written_before_provenance_loads_silently():
    """A 1.0-era document is legitimately provenance-free; warning would be noise."""
    doc = _doc()
    del doc["provenance"]
    doc["schema"] = "discopt.model/1"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = loads(json.dumps(doc))
    assert model.provenance is None


@pytest.mark.smoke
def test_a_non_object_provenance_section_is_refused():
    """Consistent with the format's no-silent-skip rule for anything it cannot read."""
    doc = _doc()
    doc["provenance"] = "0.8.0"
    with pytest.raises(SerializationError, match="must be an object"):
        loads(json.dumps(doc))


# ── the safety property ────────────────────────────────────────────────────


@pytest.mark.smoke
def test_a_corrupt_provenance_block_cannot_change_the_model():
    """Provenance is metadata. No content of it may reach the model's mathematics."""
    m = _model()
    reference = dumps(m, provenance=False)

    doc = json.loads(dumps(m))
    doc["provenance"] = {
        "created": "not a date",
        "software": {"name": "evil", "version": None, "rust_core": None},
        "platform": {"python": ["nonsense"]},
        "objective": 12345,
        "variables": "clobber",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProvenanceSkewWarning)
        back = loads(json.dumps(doc))
    assert dumps(back, provenance=False) == reference


# ── authorship is never inferred ───────────────────────────────────────────


@pytest.mark.smoke
def test_author_is_absent_unless_given():
    """discopt's own CITATION.cff author must never be copied onto a user's model."""
    prov = _doc()["provenance"]
    assert "author" not in prov


@pytest.mark.smoke
def test_author_is_recorded_when_passed():
    assert _doc(author="A. Researcher")["provenance"]["author"] == "A. Researcher"


@pytest.mark.smoke
def test_author_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROVENANCE_AUTHOR", "Env Author")
    assert _doc()["provenance"]["author"] == "Env Author"


@pytest.mark.smoke
def test_an_explicit_author_beats_the_environment(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROVENANCE_AUTHOR", "Env Author")
    assert _doc(author="Explicit")["provenance"]["author"] == "Explicit"


# ── the derivation chain ───────────────────────────────────────────────────


@pytest.mark.smoke
def test_re_saving_a_loaded_model_records_what_it_came_from():
    """`created` describes the new document, so the old one would otherwise vanish."""
    first = _doc()
    reloaded = loads(json.dumps(first))
    second = json.loads(dumps(reloaded))

    assert second["provenance"]["derived_from"] == first["provenance"]
    assert "derived_from" not in first["provenance"], "the first save came from nothing"


@pytest.mark.smoke
def test_the_derivation_chain_is_capped_at_one_level():
    """Otherwise a thousand round trips carry a thousand-deep block."""
    doc = _doc()
    for _ in range(4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ProvenanceSkewWarning)
            doc = json.loads(dumps(loads(json.dumps(doc))))
    assert "derived_from" in doc["provenance"]
    assert "derived_from" not in doc["provenance"]["derived_from"]


# ── the opt-out ────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_provenance_off_is_byte_reproducible():
    """The timestamp makes the default non-reproducible; this is the escape hatch."""
    m = _model()
    assert dumps(m, provenance=False) == dumps(m, provenance=False)
    assert "provenance" not in json.loads(dumps(m, provenance=False))


@pytest.mark.smoke
def test_two_dumps_of_one_model_differ_only_in_the_provenance_block():
    """Nothing but the block moves between saves.

    Deliberately NOT named "is not byte reproducible": two saves inside the same
    second produce the same timestamp, so non-reproducibility is not something
    this can assert deterministically. What it does assert is the containment --
    that the non-reproducibility, when it happens, is confined to `provenance`.
    """
    m = _model()
    a, b = json.loads(dumps(m)), json.loads(dumps(m))
    a.pop("provenance")
    b.pop("provenance")
    assert a == b


@pytest.mark.smoke
def test_save_passes_the_opt_out_through(tmp_path):
    path = tmp_path / "m.dopt"
    _model().save(path, provenance=False)
    assert "provenance" not in json.loads(path.read_text())
    assert load(path).provenance is None


# ── the git reader (no subprocess) ─────────────────────────────────────────


@pytest.mark.smoke
def test_git_head_reads_a_loose_ref(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert _git_head(tmp_path) == "a" * 40


@pytest.mark.smoke
def test_git_head_reads_a_detached_head(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("b" * 40 + "\n")
    assert _git_head(tmp_path) == "b" * 40


@pytest.mark.smoke
def test_git_head_reads_a_packed_ref(tmp_path):
    """`git gc` moves loose refs into packed-refs; a loose-only reader returns None."""
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{'c' * 40} refs/heads/main\n"
        f"{'d' * 40} refs/tags/v1\n"
    )
    assert _git_head(tmp_path) == "c" * 40


@pytest.mark.smoke
def test_git_head_is_none_outside_a_checkout(tmp_path):
    assert _git_head(tmp_path) is None


@pytest.mark.smoke
def test_git_commit_is_omitted_rather_than_written_null(tmp_path, monkeypatch):
    """An absent field says 'not a checkout'; a null field reads as a failed lookup."""
    monkeypatch.setattr("discopt.provenance._git_head", lambda root: None)
    assert "git_commit" not in capture()


# ── helpers ────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_skew_warning_is_silent_without_a_block():
    assert skew_warning(None) is None
    assert skew_warning({}) is None
    assert skew_warning({"created": "2020-01-01T00:00:00+00:00"}) is None
