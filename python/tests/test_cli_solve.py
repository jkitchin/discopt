"""Tests for the ``discopt solve`` CLI command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.cli import _cmd_solve

pytestmark = pytest.mark.unit


def _make_nl(tmp_path: Path) -> Path:
    m = dm.Model("m")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.subject_to(x >= 1, name="c")
    nl = tmp_path / "m.nl"
    m.to_nl(str(nl))
    return nl


def _args(nl: Path, **over) -> argparse.Namespace:
    base = dict(
        file=str(nl),
        profile=None,
        time_limit=None,
        gap=None,
        threads=None,
        solver=None,
        partitions=None,
        branching_policy=None,
        rlt=None,
        nlp_bb=None,
        tuning=None,
        no_daemon=True,  # in-process: deterministic, no socket
        format="text",
        json=False,
        sol=False,
        out_dir=None,
        quiet=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_solve_in_process_prints_and_writes_no_files(tmp_path, capsys):
    nl = _make_nl(tmp_path)
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit) as ei:
        _cmd_solve(_args(nl))
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "status:" in out and "optimal" in out
    # The litter guard: a default solve writes NO files next to the .nl.
    assert set(tmp_path.iterdir()) == before


def test_solve_json_and_sol_outputs(tmp_path):
    nl = _make_nl(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit) as ei:
        _cmd_solve(_args(nl, json=True, sol=True, out_dir=str(out), quiet=True))
    assert ei.value.code == 0
    assert (out / "m.result.json").exists()
    assert (out / "m.sol").exists()
    import json

    assert json.loads((out / "m.result.json").read_text())["status"] == "optimal"
    assert (out / "m.sol").read_text().startswith("discopt optimal")


def test_solve_format_json_to_stdout(tmp_path, capsys):
    nl = _make_nl(tmp_path)
    with pytest.raises(SystemExit):
        _cmd_solve(_args(nl, format="json"))
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1 and payload["status"] == "optimal"


def test_solve_unknown_profile_errors(tmp_path):
    nl = _make_nl(tmp_path)
    with pytest.raises(SystemExit) as ei:
        _cmd_solve(_args(nl, profile="nope"))
    assert ei.value.code == 2


def test_solve_bad_extension_errors(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("x")
    with pytest.raises(SystemExit) as ei:
        _cmd_solve(_args(p))
    assert ei.value.code == 1


@pytest.mark.slow
def test_solve_end_to_end_via_daemon(tmp_path):
    """Real subprocess: spawn the daemon, solve, then a warm solve; clean up."""
    nl = _make_nl(tmp_path)
    env = dict(__import__("os").environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")

    def run(args):
        return subprocess.run(
            [sys.executable, "-m", "discopt.cli", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    try:
        r1 = run(["solve", str(nl), "--quiet"])  # spawns daemon
        assert r1.returncode == 0, r1.stderr[-500:]
        r2 = run(["solve", str(nl)])  # warm
        assert r2.returncode == 0 and "optimal" in r2.stdout
        status = run(["daemon", "status"])
        assert "running" in status.stdout
    finally:
        run(["daemon", "stop"])
