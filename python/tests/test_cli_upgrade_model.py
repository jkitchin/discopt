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
import os
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
# The tree under test: the subprocess must import discopt from here, not from
# whatever is installed or from another checkout (CLAUDE.md §8). Several calls
# run with ``cwd=tmp_path``, so a relative PYTHONPATH would not resolve.
REPO_PYTHON = pathlib.Path(__file__).resolve().parents[1]
_LOADED_MARK = "discopt-loaded-from:"
# Runs the CLI exactly as ``python -m discopt.cli`` does (cwd on sys.path, module
# run as __main__), after printing which discopt was imported.
_PRELUDE = (
    "import sys, runpy, discopt; "
    f"print({_LOADED_MARK!r} + discopt.__file__, file=sys.stderr, flush=True); "
    "sys.argv = ['discopt.cli'] + sys.argv[1:]; "
    "runpy.run_module('discopt.cli', run_name='__main__', alter_sys=True)"
)


def _cli(*args, cwd=None) -> subprocess.CompletedProcess:
    # A subprocess, as a user runs it: atom registration is per process, and that
    # is exactly what one of the refusals below is about.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_PYTHON)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE, "upgrade-model", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=120,
    )
    first, _, rest = proc.stderr.partition("\n")
    assert first.startswith(_LOADED_MARK), proc.stderr
    loaded = pathlib.Path(first[len(_LOADED_MARK) :]).resolve()
    assert loaded.is_relative_to(REPO_PYTHON / "discopt"), (
        f"subprocess imported discopt from {loaded}, not the tree under test {REPO_PYTHON}"
    )
    proc.stderr = rest
    return proc


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


def test_upgrade_model_carries_x_annotations(tmp_path):
    """An ``x-`` key is never refused, so the rewrite must keep it, not refuse it."""
    doc = json.loads((V10 / "scalar_nlp.json").read_text())
    doc["x-note"] = {"by": "a tool", "n": [1, 2]}
    doc["x-flag"] = True
    src = tmp_path / "ann.json"
    src.write_text(json.dumps(doc))
    assert _cli("--check", src).returncode == 0
    proc = _cli(src)  # in place
    assert proc.returncode == 0, proc.stderr
    new = json.loads(src.read_text())
    assert new["schema"] == SCHEMA
    assert new["x-note"] == {"by": "a tool", "n": [1, 2]} and new["x-flag"] is True
    assert dumps(dm.load(src), provenance=False) == dumps(
        loads((V10 / "scalar_nlp.json").read_text()), provenance=False
    )


def _args(src, out=None):
    from types import SimpleNamespace

    return SimpleNamespace(input=str(src), output=out and str(out), check=False, imports=[])


def test_upgrade_model_in_place_is_atomic(tmp_path):
    from discopt.cli import _cmd_upgrade_model

    src = tmp_path / "m.json"
    shutil.copy(V10 / "shared_dag.json", src)
    src.chmod(0o640)
    _cmd_upgrade_model(_args(src))
    assert json.loads(src.read_text())["schema"] == SCHEMA  # complete, parseable
    assert (src.stat().st_mode & 0o777) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.json"]  # no temp left


def test_upgrade_model_failed_write_leaves_the_original(tmp_path, monkeypatch):
    """A failure mid-write must not truncate the only copy of the model."""
    import discopt.cli as cli

    src = tmp_path / "m.json"
    shutil.copy(V10 / "shared_dag.json", src)
    before = src.read_bytes()
    writes = []

    def _half_then_fail(fh, payload):
        fh.write(payload[: len(payload) // 2])
        writes.append(1)
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(cli, "_write_upgraded", _half_then_fail)
    with pytest.raises(SystemExit) as exc:
        cli._cmd_upgrade_model(_args(src))
    assert writes == [1]  # the failure was injected, not skipped
    assert exc.value.code == 1
    assert src.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.json"]


def test_upgrade_model_import_failure_is_a_usage_error(tmp_path):
    src = tmp_path / "in.json"
    shutil.copy(V10 / "scalar_nlp.json", src)
    before = src.read_bytes()
    proc = _cli("--import", "no_such_module_wp2", src, cwd=tmp_path)
    assert proc.returncode == 2
    assert "Error: cannot import no_such_module_wp2:" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert src.read_bytes() == before

    (tmp_path / "broken_atoms.py").write_text("raise RuntimeError('boom at import')\n")
    proc = _cli("--import", "broken_atoms", src, cwd=tmp_path)
    assert proc.returncode == 2
    assert "Error: cannot import broken_atoms: RuntimeError: boom at import" in proc.stderr
    assert "Traceback" not in proc.stderr
