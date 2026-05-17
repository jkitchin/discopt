"""Tests for category benchmark runner process isolation."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import SolveResult, SolveStatus
from benchmarks.problems.base import TestProblem as BenchmarkProblem
from category_runner import CategoryBenchmarkRunner, _run_worker


def _problem(build_fn=None) -> BenchmarkProblem:
    return BenchmarkProblem(
        name="dummy",
        category="lp",
        level="smoke",
        build_fn=build_fn or (lambda: None),
        known_optimum=0.0,
        applicable_solvers=["ipm"],
    )


def _fake_popen_factory(
    worker_result=None,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    timeout_expired: bool = False,
    on_init=None,
):
    class FakePopen:
        pid = 12345

        def __init__(self, cmd, **kwargs):
            assert kwargs["stdout"] is subprocess.PIPE
            assert kwargs["stderr"] is subprocess.PIPE
            assert kwargs["text"] is True
            assert kwargs["start_new_session"] is True
            self.cmd = cmd
            self.returncode = returncode
            self._wrote_result = False
            if on_init is not None:
                on_init(self)

        def communicate(self, timeout=None):
            if timeout is not None and timeout_expired:
                raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout)
            if worker_result is not None and not timeout_expired and not self._wrote_result:
                result = worker_result() if callable(worker_result) else worker_result
                Path(self.cmd[-1]).write_text(
                    json.dumps(result.to_dict()),
                    encoding="utf-8",
                )
                self._wrote_result = True
            return stdout, stderr

    return FakePopen


def test_category_runner_hard_timeout_returns_time_limit(monkeypatch):
    """The parent runner should kill over-budget workers and record TL."""
    killed = []

    monkeypatch.setattr(
        "category_runner.subprocess.Popen",
        _fake_popen_factory(timeout_expired=True),
    )
    monkeypatch.setattr("category_runner.os.getpgid", lambda pid: 6789)
    monkeypatch.setattr(
        "category_runner.os.killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )

    runner = CategoryBenchmarkRunner(
        category="lp",
        level="smoke",
        time_limit=3.0,
        hard_timeout_grace=0.5,
    )
    result = runner._run_with_hard_timeout(_problem(), "ipm")

    assert result.status == SolveStatus.TIME_LIMIT
    assert result.wall_time == 3.0
    assert result.solver == "discopt_ipm"
    assert killed == [(6789, signal.SIGKILL)]


def test_category_runner_caps_over_limit_worker_results(monkeypatch):
    """A worker that returns during grace should not report over-limit time."""

    worker_result = SolveResult(
        instance="dummy",
        solver="discopt_ipm",
        status=SolveStatus.OPTIMAL,
        objective=1.0,
        wall_time=4.0,
    )

    monkeypatch.setattr(
        "category_runner.subprocess.Popen",
        _fake_popen_factory(worker_result),
    )

    runner = CategoryBenchmarkRunner(
        category="lp",
        level="smoke",
        time_limit=3.0,
        hard_timeout_grace=1.0,
    )
    result = runner._run_with_hard_timeout(_problem(), "ipm")

    assert result.status == SolveStatus.FEASIBLE
    assert result.objective == 1.0
    assert result.wall_time == 3.0


def test_category_runner_converts_late_certificates_to_time_limit(monkeypatch):
    """Late proof certificates should not be counted as in-budget results."""
    statuses = [SolveStatus.INFEASIBLE, SolveStatus.UNBOUNDED]

    def next_worker_result():
        worker_result = SolveResult(
            instance="dummy",
            solver="discopt_ipm",
            status=statuses.pop(0),
            wall_time=4.0,
        )
        return worker_result

    monkeypatch.setattr(
        "category_runner.subprocess.Popen",
        _fake_popen_factory(next_worker_result),
    )

    runner = CategoryBenchmarkRunner(
        category="lp",
        level="smoke",
        time_limit=3.0,
        hard_timeout_grace=1.0,
    )

    for _ in range(2):
        result = runner._run_with_hard_timeout(_problem(), "ipm")
        assert result.status == SolveStatus.TIME_LIMIT
        assert result.wall_time == 3.0


def test_category_runner_prints_error_worker_output_when_verbose(
    monkeypatch,
    capsys,
):
    """Verbose hard-timeout runs should surface worker diagnostics on ERROR."""

    worker_result = SolveResult(
        instance="dummy",
        solver="discopt_ipm",
        status=SolveStatus.ERROR,
        wall_time=0.2,
    )

    monkeypatch.setattr(
        "category_runner.subprocess.Popen",
        _fake_popen_factory(worker_result, stdout="worker stdout\n", stderr="worker stderr\n"),
    )
    monkeypatch.setattr("category_runner.sys.argv", ["run_category_benchmarks.py", "--verbose"])

    runner = CategoryBenchmarkRunner(
        category="lp",
        level="smoke",
        time_limit=3.0,
        hard_timeout_grace=1.0,
    )
    result = runner._run_with_hard_timeout(_problem(), "ipm")
    captured = capsys.readouterr()

    assert result.status == SolveStatus.ERROR
    assert "worker stdout" in captured.out
    assert "worker stderr" in captured.err


def test_category_runner_explicit_amp_filter_runs_amp(monkeypatch):
    """An explicit AMP filter should run AMP even when default solvers omit it."""
    calls = []

    def fake_run_discopt(self, problem, solver):
        del self, problem
        calls.append(solver)
        return SolveResult(
            instance="dummy",
            solver=f"discopt_{solver}",
            status=SolveStatus.TIME_LIMIT,
            wall_time=0.1,
        )

    monkeypatch.setattr("category_runner.get_problems", lambda category, level: [_problem()])
    monkeypatch.setattr(CategoryBenchmarkRunner, "_run_discopt", fake_run_discopt)

    runner = CategoryBenchmarkRunner(
        category="lp",
        level="smoke",
        time_limit=3.0,
        hard_timeout_grace=None,
        solver_filter=["amp"],
    )
    results = runner.run()

    assert calls == ["amp"]
    assert results.get_solvers() == ["discopt_amp"]


def test_category_runner_amp_solver_calls_model_solve_with_solver_amp():
    """The AMP benchmark backend should call Model.solve(solver='amp')."""
    captured = {}

    class FakeModel:
        def solve(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status="optimal",
                objective=1.0,
                bound=1.0,
                node_count=0,
                iterations=0,
            )

    runner = CategoryBenchmarkRunner(
        category="lp",
        level="smoke",
        time_limit=3.0,
        hard_timeout_grace=None,
    )
    result = runner._run_discopt(_problem(build_fn=FakeModel), "amp")

    assert captured["solver"] == "amp"
    assert captured["time_limit"] == 3.0
    assert captured["gap_tolerance"] == 1e-4
    assert "nlp_solver" not in captured
    assert "max_nodes" not in captured
    assert result.solver == "discopt_amp"
    assert result.status == SolveStatus.OPTIMAL


def test_category_worker_writes_amp_result(tmp_path, monkeypatch):
    """The hard-timeout worker entrypoint should run and serialize AMP results."""

    class FakeModel:
        def solve(self, **kwargs):
            assert kwargs["solver"] == "amp"
            return SimpleNamespace(
                status="optimal",
                objective=1.0,
                bound=1.0,
                node_count=0,
                iterations=0,
            )

    monkeypatch.setattr(
        "category_runner.get_problems",
        lambda category, level: [_problem(build_fn=FakeModel)],
    )

    result_path = tmp_path / "result.json"
    rc = _run_worker(["lp", "smoke", "dummy", "amp", "3.0", str(result_path)])

    assert rc == 0
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["solver"] == "discopt_amp"
    assert data["status"] == "optimal"
