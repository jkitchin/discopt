"""External solvers must prove they work before a benchmark run starts.

The 2026-11 global50 run went out with ``[solvers.scip]`` pointing into a
miniforge env that had been deleted. Every launch raised, the runner's
``except Exception`` turned each into an ERROR row with infinite time, the log
printed ``TL``, and the summary said ``scip 0/50`` -- a column that read as a
solver result. These tests pin the class fix: a preflight probe with a known
optimum, a refusal before any instance runs, and no silent ERROR rows.
"""

from __future__ import annotations

import itertools
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from benchmarks.metrics import InstanceInfo, SolveResult, SolveStatus  # noqa: E402
from benchmarks.runner import (  # noqa: E402
    PREFLIGHT_PROBES,
    BenchmarkConfig,
    BenchmarkRunner,
    SolverConfig,
    SolverUnavailableError,
)

pytestmark = pytest.mark.unit

SCIP_OPTIMAL = (
    "SCIP Status        : problem is solved [optimal solution found]\n"
    "Primal Bound       : {obj}\n"
    "Dual Bound         : {obj}\n"
)


def _fake_solver(tmp_path: Path, name: str, stdout: str) -> str:
    """An executable that ignores its arguments and prints ``stdout``."""
    path = tmp_path / name
    body = stdout.replace("'", "'\\''")
    path.write_text(f"#!/bin/sh\nprintf '%s' '{body}'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _runner(*solvers: SolverConfig) -> BenchmarkRunner:
    return BenchmarkRunner(BenchmarkConfig(suite_name="unit", solvers=list(solvers)))


def _nl_header_sizes(path: Path) -> tuple[int, int]:
    n_vars, n_cons = path.read_text().splitlines()[1].split()[:2]
    return int(n_vars), int(n_cons)


# --- the probes themselves -------------------------------------------------


@pytest.mark.parametrize("key", sorted(PREFLIGHT_PROBES))
def test_probe_is_above_the_demo_license_cap(key):
    """Above 10 vars AND 10 cons, or a demo-licensed binary would pass it."""
    probe = PREFLIGHT_PROBES[key]
    assert probe.nl_path.is_file(), probe.nl_path
    n_vars, n_cons = _nl_header_sizes(probe.nl_path)
    assert n_vars > 10 and n_cons > 10, (key, n_vars, n_cons)


def test_milp_probe_optimum_is_brute_forced_and_its_side_row_binds():
    sys.path.insert(0, str(BENCH / "data" / "preflight"))
    import make_probe_milp as gen

    assert PREFLIGHT_PROBES["milp"].optimum == gen.brute_force_optimum()
    unconstrained = min(
        sum(gen.COST[i][p[i]] for i in range(gen.N)) for p in itertools.permutations(range(gen.N))
    )
    # If the side row did not bind, a solver that dropped it would still pass.
    assert unconstrained < gen.brute_force_optimum()


# --- refusals ----------------------------------------------------------------


def test_missing_binary_refuses_before_any_instance_runs(tmp_path, monkeypatch):
    missing = str(tmp_path / "no-such-dir" / "scip")
    r = _runner(SolverConfig(name="scip", command=missing, solver_type="external"))
    ran = []
    monkeypatch.setattr(r, "_run_single", lambda *a: ran.append(a))
    with pytest.raises(SolverUnavailableError, match="not found or not executable") as ei:
        r.preflight_external_solvers()
    assert missing in str(ei.value)
    assert ran == []


def test_wrong_answer_is_refused(tmp_path):
    cmd = _fake_solver(tmp_path, "scip", SCIP_OPTIMAL.format(obj="+1.0e+00"))
    r = _runner(SolverConfig(name="scip", command=cmd, solver_type="external"))
    with pytest.raises(SolverUnavailableError, match="known optimum"):
        r.preflight_external_solvers()


def test_demo_license_refusal_is_refused(tmp_path):
    cmd = _fake_solver(
        tmp_path,
        "baron",
        "Sorry, a demo license is limited to 10 variables and 10 constraints.\n",
    )
    r = _runner(SolverConfig(name="baron", command=cmd, solver_type="external"))
    with pytest.raises(SolverUnavailableError, match="baron"):
        r.preflight_external_solvers()


def test_every_failure_is_reported_not_just_the_first(tmp_path):
    a = SolverConfig(name="scip", command=str(tmp_path / "a"), solver_type="external")
    b = SolverConfig(name="couenne", command=str(tmp_path / "b"), solver_type="external")
    with pytest.raises(SolverUnavailableError) as ei:
        _runner(a, b).preflight_external_solvers()
    assert "scip" in str(ei.value) and "couenne" in str(ei.value)


def test_unknown_probe_key_is_refused(tmp_path):
    cmd = _fake_solver(tmp_path, "scip", "")
    cfg = SolverConfig(name="scip", command=cmd, solver_type="external", preflight_probe="nope")
    with pytest.raises(SolverUnavailableError, match="unknown preflight_probe"):
        _runner(cfg).preflight_external_solvers()


# --- acceptance --------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(PREFLIGHT_PROBES))
def test_a_correct_answer_passes(tmp_path, key):
    obj = f"{PREFLIGHT_PROBES[key].optimum:+.10e}"
    cmd = _fake_solver(tmp_path, "scip", SCIP_OPTIMAL.format(obj=obj))
    cfg = SolverConfig(name="scip", command=cmd, solver_type="external", preflight_probe=key)
    _runner(cfg).preflight_external_solvers()  # must not raise


def test_internal_solvers_are_not_probed(monkeypatch):
    r = _runner(SolverConfig(name="discopt", command="discopt", solver_type="internal"))
    monkeypatch.setattr(r, "_preflight_one", lambda s: pytest.fail("probed discopt"))
    r.preflight_external_solvers()


@pytest.mark.parametrize("key", sorted(PREFLIGHT_PROBES))
def test_real_scip_solves_both_probes(key):
    scip = shutil.which("scip")
    if scip is None:
        pytest.skip("no scip on PATH")
    cfg = SolverConfig(name="scip", command=scip, solver_type="external", preflight_probe=key)
    _runner(cfg).preflight_external_solvers()


# --- no silent ERROR rows ----------------------------------------------------


def test_launch_failure_during_a_run_is_printed(tmp_path, capsys):
    cfg = SolverConfig(name="scip", command=str(tmp_path / "gone"), solver_type="external")
    res = _runner(cfg)._run_external(cfg, "flay03m", 0)
    assert res.status is SolveStatus.ERROR
    out = capsys.readouterr().out
    assert "FileNotFoundError" in out and "recorded as ERROR" in out


def test_error_row_prints_err_not_tl(tmp_path, monkeypatch, capsys):
    (tmp_path / "i1.nl").write_text("g3 0 1 0\n")
    cfg = SolverConfig(name="a", command="a", solver_type="internal")
    r = BenchmarkRunner(
        BenchmarkConfig(suite_name="t", solvers=[cfg], num_runs=1, data_dir=tmp_path)
    )
    r.load_instances([InstanceInfo(name="i1", num_variables=1, num_constraints=1)])
    monkeypatch.setattr(
        r,
        "_run_single",
        lambda s, i, k: SolveResult(instance=i, solver=s.name, status=SolveStatus.ERROR),
    )
    r.run_all()
    line = next(ln for ln in capsys.readouterr().out.splitlines() if " i1 " in ln)
    assert "ERR" in line and "TL" not in line


# --- the CLI -----------------------------------------------------------------


def test_cli_refuses_a_solver_name_with_no_config_entry():
    """A typo used to run discopt under the typo's label."""
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH / "run_benchmarks.py"),
            "--suite",
            "smoke",
            "--solvers",
            "discopt,scpi",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "scpi" in proc.stderr and "no [solvers.*] entry" in proc.stderr


def test_cli_refuses_an_unavailable_external_solver(tmp_path):
    """The global50 incident end to end: bonmin's configured env is gone here."""
    import tomllib

    cfg = tomllib.loads((BENCH / "config" / "benchmarks.toml").read_text())
    if shutil.which(cfg["solvers"]["bonmin"]["command"]) is not None:
        pytest.skip("bonmin is installed on this machine; nothing to refuse")
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH / "run_benchmarks.py"),
            "--suite",
            "smoke",
            "--solvers",
            "discopt,bonmin",
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "preflight failed" in proc.stderr and "bonmin" in proc.stderr
    assert not (tmp_path / "out.json").exists()
