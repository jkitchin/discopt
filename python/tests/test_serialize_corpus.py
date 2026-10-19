"""Corpus-wide round trip of the native format (WP-5, D10).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-5. Every in-repo corpus
instance -- the two vendored MINLPLib ``.nl`` sets, the vendored QPLIB instances,
``binary_nl`` and the complementarity ``.nl`` -- goes through
``scripts/serialize_corpus_roundtrip.py``'s per-instance check: ``dumps`` stable
under reload, ``.nl`` byte-identical (or refused the same way), equal
``model_fingerprint`` after ``loads(dumps())`` and after ``pickle``, and the
``.nl`` source digest preserved. The same script runs the full MINLPLib / QPLIB
corpora where they are present (D11; local, recorded in the plan's §8).

The two corpus tests that lived in ``test_serialize_roundtrip.py`` moved here
unchanged; the bound-neutral panel below is the solve-level check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.serialize import dumps, loads

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "serialize_corpus_roundtrip", REPO / "scripts" / "serialize_corpus_roundtrip.py"
)
harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = harness  # dataclasses resolve their module by name
_spec.loader.exec_module(harness)

INSTANCES = harness.in_repo_instances()
NL_INSTANCES = [i for i in INSTANCES if i.kind == "nl"]
QPLIB_INSTANCES = [i for i in INSTANCES if i.kind == "qplib"]

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
CORPUS = sorted(NL_DIR.glob("*.nl"))


def _id(inst) -> str:
    return f"{inst.corpus}/{inst.name}"


@pytest.mark.slow
@pytest.mark.parametrize("inst", NL_INSTANCES, ids=_id)
def test_in_repo_corpus_round_trips(inst):
    row = harness.check_instance(inst)
    assert row["status"] == "ok", row


@pytest.mark.slow
@pytest.mark.parametrize("inst", QPLIB_INSTANCES, ids=_id)
def test_qplib_in_repo_round_trips(inst):
    row = harness.check_instance(inst)
    assert row["status"] == "ok", row


def test_corpus_probe_fired():
    """§6: the parametrized tests above measure nothing over an empty glob."""
    assert len(INSTANCES) >= 75, len(INSTANCES)
    assert len(QPLIB_INSTANCES) >= 9, len(QPLIB_INSTANCES)
    assert {i.corpus for i in INSTANCES} == {
        "minlplib_nl",
        "minlplib",
        "binary_nl",
        "mpcc",
        "qplib_in_repo",
    }
    assert all(i.path.is_file() for i in INSTANCES)


def test_the_source_digest_check_fires():
    """The digest check compares *presence* too, so it would pass vacuously if no
    instance carried one; a complementarity ``.nl`` deliberately carries none."""
    rows = [harness.check_instance(i) for i in NL_INSTANCES[:5]]
    assert all(r["source_digest"] for r in rows), rows
    mpcc = next(i for i in NL_INSTANCES if i.corpus == "mpcc")
    assert harness.check_instance(mpcc)["source_digest"] is False


def test_harness_refuses_a_run_that_executed_nothing(monkeypatch, capsys):
    monkeypatch.setattr(harness, "in_repo_instances", lambda: [])
    assert harness.main(["--in-repo-only"]) == 2
    assert "executed: 0" in capsys.readouterr().out


def test_harness_reports_a_failed_instance(monkeypatch, capsys):
    """A failing check must reach the exit code, not only the progress line."""
    inst = NL_INSTANCES[0]
    monkeypatch.setattr(harness, "in_repo_instances", lambda: [inst])
    monkeypatch.setattr(harness, "check_instance", lambda i: {"status": "planted failure"})
    assert harness.main(["--in-repo-only"]) == 1
    assert "FAIL planted failure" in capsys.readouterr().out


@pytest.mark.parametrize("inst", [NL_INSTANCES[0], QPLIB_INSTANCES[0]], ids=_id)
def test_harness_detects_a_lossy_reader(monkeypatch, inst):
    """§6: the checks must catch a reader that loses something. Plant one -- a
    ``loads`` that moves the first variable's upper bound -- and every model-level
    check has to report it. (Set, not shifted: ``inf + 1`` is ``inf``, and a first
    draft of this probe planted nothing on an unbounded variable.)"""
    import discopt.serialize as ser
    import numpy as np

    real = ser.loads

    def lossy(text, *a, **k):
        m = real(text, *a, **k)
        v = m._variables[0]
        v.ub = np.full(np.shape(v.ub), 12345.678)
        return m

    monkeypatch.setattr(ser, "loads", lossy)
    status = harness.check_instance(inst)["status"]
    assert "dumps not stable under reload" in status, status
    assert "fingerprint differs after loads(dumps())" in status, status
    assert ".nl differs after round trip" in status, status


# ── moved unchanged from test_serialize_roundtrip.py ───────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("nl_path", CORPUS, ids=lambda p: p.stem)
def test_corpus_round_trip_is_nl_byte_identical(nl_path):
    """Every in-repo MINLPLib instance survives .nl -> native -> .nl unchanged."""
    m = dm.from_nl(str(nl_path))
    once = dumps(m, provenance=False)
    reloaded = loads(once)
    assert dumps(reloaded, provenance=False) == once, (
        f"{nl_path.stem}: document not stable under reload"
    )
    assert reloaded.to_nl() == m.to_nl(), f"{nl_path.stem}: .nl differs after round trip"


def test_corpus_probe_actually_fired():
    """The corpus test is parametrized over a glob; an empty glob would pass silently."""
    assert len(CORPUS) >= 60, f"expected the in-repo MINLPLib corpus, found {len(CORPUS)} files"


# ── the solve is bound-neutral across the round trip (CLAUDE.md §5) ────────

#: Selection rule: the ``known_optima.toml`` entries with an in-repo ``.nl`` that
#: solve to ``optimal`` in under 20 s, the 12 largest by ``n_vars + n_rows`` (ties by
#: name). Measured at WP-5 entry (wall of one solve; each also gave the identical
#: ``(status, node_count, objective)`` on a fresh re-parse, so the comparison below
#: is not at the mercy of solver nondeterminism): clay0303hfsg 249 vars+rows 5.3 s,
#: st_testgr3 40 0.5 s, mathopt3 13 0.7 s, nvs09 10 9.5 s, nvs13 10 1.7 s,
#: nvs12 8 0.4 s, st_miqp2 7 0.1 s, nvs11 6 0.4 s, st_miqp1 6 0.1 s, st_test1 6
#: 0.1 s, nvs07 5 0.1 s, gear 4 0.2 s. Excluded by the 20 s rule: nvs17 (50 s, and
#: its node count drifts by 2 across fresh parses), nvs19 and nvs24 (time limit).
PANEL: list[str] = [
    "clay0303hfsg",
    "st_testgr3",
    "mathopt3",
    "nvs09",
    "nvs13",
    "nvs12",
    "st_miqp2",
    "nvs11",
    "st_miqp1",
    "st_test1",
    "nvs07",
    "gear",
]


@pytest.mark.slow
@pytest.mark.parametrize("name", PANEL)
def test_round_trip_panel_is_bound_neutral(name):
    """``status``, ``node_count`` and ``objective`` exactly unchanged -- not close."""
    path = next(
        p
        for p in (
            Path(__file__).parent / "data" / "minlplib_nl" / f"{name}.nl",
            Path(__file__).parent / "data" / "minlplib" / f"{name}.nl",
        )
        if p.exists()
    )
    original = dm.from_nl(str(path))
    reloaded = loads(dumps(original))
    r1 = original.solve(time_limit=60)
    r2 = reloaded.solve(time_limit=60)
    assert (r1.status, r1.node_count, r1.objective) == (r2.status, r2.node_count, r2.objective)
    # Equal is not enough: both could time out identically. Each panel member is a
    # known optimum solved well inside the limit, so demand the certified answer.
    from _optima import known_optimum

    assert r1.status == "optimal", r1.status
    assert r1.objective == pytest.approx(known_optimum(name), rel=1e-4, abs=1e-6)


def test_panel_is_twelve_known_optima():
    from _optima import optima_registry

    registry = optima_registry()
    assert len(PANEL) == 12 and len(set(PANEL)) == 12
    assert set(PANEL) <= set(registry), sorted(set(PANEL) - set(registry))
