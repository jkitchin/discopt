"""``discopt upgrade-model``: rewrite a saved model at the current schema (WP-2).

Plan: ``docs/dev/serialization-completion-plan.md`` §3.5 (DoD D5). Load, re-save at
the current ``SCHEMA`` with the embedded result and the document's own provenance
(as ``previous``), and refuse rather than write when the rewrite would lose
anything: a document from a NEWER format (rewriting downgrades it), or content this
process cannot carry (a registered-atom tag for an atom it has not registered).
Exit codes: 0 ok, 1 refused, 2 usage.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import shutil
import subprocess
import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.serialize import SCHEMA, SCHEMA_MINOR, dumps, loads

FIXTURES = pathlib.Path(__file__).parent / "data" / "serialize_fixtures"
V10 = FIXTURES / "v1.0"


def _cli(*args, cwd=None) -> subprocess.CompletedProcess:
    # A subprocess, as a user runs it: atom registration is per process, and that
    # is exactly what one of the refusals below is about.
    return subprocess.run(
        [sys.executable, "-m", "discopt.cli", "upgrade-model", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=120,
    )


def test_upgrade_model_rewrites_1_0_fixture_to_current(tmp_path):
    out = tmp_path / "x.json"
    proc = _cli(V10 / "scalar_nlp.json", out)
    assert proc.returncode == 0, proc.stderr
    assert f"discopt.model/1 -> {SCHEMA}" in proc.stdout
    doc = json.loads(out.read_text())
    assert doc["schema"] == SCHEMA
    # The same model: the 1.0 document and its upgrade load to identical models.
    old = loads((V10 / "scalar_nlp.json").read_text())
    assert dumps(dm.load(out), provenance=False) == dumps(old, provenance=False)


def test_upgrade_model_carries_the_embedded_result(tmp_path):
    out = tmp_path / "r.json"
    assert _cli(V10 / "with_result.json", out).returncode == 0
    old = loads((V10 / "with_result.json").read_text()).saved_result
    new = dm.load(out).saved_result
    assert new.status == old.status == "optimal"
    assert new.objective == old.objective
    assert new.x.keys() == old.x.keys()
    for name in old.x:
        assert np.array_equal(new.x[name], old.x[name]), name


def test_upgrade_model_check_writes_nothing(tmp_path):
    src = tmp_path / "in.json"
    shutil.copy(V10 / "fixed_scope.json", src)
    before = src.read_bytes()
    proc = _cli("--check", src)
    assert proc.returncode == 0, proc.stderr
    assert "content changes" in proc.stdout or "content unchanged" in proc.stdout
    assert src.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.json"]


def test_upgrade_model_carries_previous_provenance(tmp_path):
    m = dm.Model("prov")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    src = tmp_path / "p.json"
    m.save(src, author="first author")
    first = json.loads(src.read_text())["provenance"]
    assert _cli(src).returncode == 0  # in place
    block = json.loads(src.read_text())["provenance"]
    assert block["derived_from"]["created"] == first["created"]
    assert block["derived_from"]["author"] == block["author"] == "first author"


def test_upgrade_model_in_place_keeps_gzip(tmp_path):
    src = tmp_path / "m.dopt"  # no .gz suffix: compression is sniffed, and kept
    src.write_bytes(gzip.compress((V10 / "shared_dag.json").read_bytes()))
    assert _cli(src).returncode == 0
    raw = src.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert json.loads(gzip.decompress(raw))["schema"] == SCHEMA


def test_upgrade_model_refuses_a_newer_document(tmp_path):
    doc = json.loads((V10 / "scalar_nlp.json").read_text())
    doc["schema"] = f"discopt.model/1.{SCHEMA_MINOR + 1}"
    src = tmp_path / "newer.json"
    src.write_text(json.dumps(doc))
    before = src.read_bytes()
    proc = _cli(src)
    assert proc.returncode == 1
    assert "refused" in proc.stderr and "newer format" in proc.stderr
    assert src.read_bytes() == before


def test_upgrade_model_refuses_to_drop_an_unregistered_atom(tmp_path):
    """The atom tag is re-derived on load only for a registered name, so a bare
    process would silently strip it; with ``--import`` it is carried."""
    src = FIXTURES / "v1.3" / "registered_atom.json"
    out = tmp_path / "a.json"
    proc = _cli(src, out)
    assert proc.returncode == 1
    assert "/atom" in proc.stderr and "--import" in proc.stderr
    assert not out.exists()

    (tmp_path / "wp1_atoms.py").write_text(
        "from discopt.operators import register_function\n"
        "register_function('wp1_fixture_cubic', lambda t: t**3 - 2 * t, replace=True)\n"
    )
    proc = _cli("--import", "wp1_atoms", src, out, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    atoms = [n["atom"] for n in json.loads(out.read_text())["nodes"] if "atom" in n]
    assert len(atoms) == 2


@pytest.mark.parametrize(
    "content, needle",
    [("{not json", "not valid JSON"), ('{"schema": "discopt.model/2.0"}', "major version 2")],
)
def test_upgrade_model_refuses_a_bad_document(tmp_path, content, needle):
    src = tmp_path / "bad.json"
    src.write_text(content)
    proc = _cli(src)
    assert proc.returncode == 1
    assert needle in proc.stderr
    assert src.read_text() == content


def test_upgrade_model_usage_error_is_exit_2():
    assert _cli().returncode == 2


def test_every_golden_fixture_upgrades(tmp_path):
    """Every shipped document but the atom case (needs ``--import``) upgrades."""
    from types import SimpleNamespace

    from discopt.cli import _cmd_upgrade_model

    done = 0
    for path in sorted(FIXTURES.glob("v*/*.json")):
        if path.name == "registered_atom.json":
            continue
        out = tmp_path / f"{path.parent.name}-{path.name}"
        _cmd_upgrade_model(
            SimpleNamespace(input=str(path), output=str(out), check=False, imports=[])
        )
        old = loads(path.read_text())
        assert dumps(dm.load(out), provenance=False) == dumps(old, provenance=False), path
        done += 1
    assert done >= 60
