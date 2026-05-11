"""Tests for category benchmark runner process isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import SolveResult, SolveStatus
from benchmarks.problems.base import TestProblem as BenchmarkProblem
from category_runner import CategoryBenchmarkRunner


def _problem() -> BenchmarkProblem:
    return BenchmarkProblem(
        name="dummy",
        category="lp",
        level="smoke",
        build_fn=lambda: None,
        known_optimum=0.0,
        applicable_solvers=["ipm"],
    )


def test_category_runner_hard_timeout_returns_time_limit(monkeypatch):
    """The parent runner should kill over-budget workers and record TL."""

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("category_runner.subprocess.run", fake_run)

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


def test_category_runner_caps_over_limit_worker_results(monkeypatch):
    """A worker that returns during grace should not report over-limit time."""

    def fake_run(cmd, **kwargs):
        del kwargs
        worker_result = SolveResult(
            instance="dummy",
            solver="discopt_ipm",
            status=SolveStatus.OPTIMAL,
            objective=1.0,
            wall_time=4.0,
        )
        Path(cmd[-1]).write_text(json.dumps(worker_result.to_dict()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("category_runner.subprocess.run", fake_run)

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
