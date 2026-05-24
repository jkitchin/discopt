"""Tests for the scaled MINLPLib validation pipeline (items 1-5).

Covers:
  * scripts.fetch_minlplib  — cache layout, manifest, sha256 verification
  * utils.minlplib_data     — CSV parsing, outcome scoring
  * benchmarks.scaled_runner — per-instance result files, resumability,
                               subprocess error handling (no real solve)
  * utils.reporting          — minlplib-style per-class table
  * utils.baseline           — pin / load / compare gate logic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make benchmarks/utils/scripts importable without depending on test conftest.
_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))


# ──────────────────────────────────────────────────────────────────
# fetch_minlplib
# ──────────────────────────────────────────────────────────────────


def test_fetch_cache_layout(tmp_path):
    from scripts.fetch_minlplib import (
        get_cache_dir,
        get_instancedata_path,
        get_manifest_path,
        get_nl_dir,
        get_version_dir,
    )

    cache = tmp_path / "cache"
    assert get_version_dir(cache, "v1") == cache / "v1"
    assert get_nl_dir(cache, "v1") == cache / "v1" / "nl"
    assert get_instancedata_path(cache, "v1") == cache / "v1" / "instancedata.csv"
    assert get_manifest_path(cache, "v1") == cache / "v1" / "manifest.json"


def test_fetch_env_override(tmp_path, monkeypatch):
    from scripts.fetch_minlplib import get_cache_dir

    monkeypatch.setenv("DISCOPT_MINLPLIB_CACHE", str(tmp_path / "envcache"))
    assert get_cache_dir() == tmp_path / "envcache"


def test_manifest_roundtrip(tmp_path):
    from scripts.fetch_minlplib import Manifest, load_manifest, save_manifest

    m = Manifest(
        version="v1",
        base_url="http://example",
        fetched_at="2026-05-23",
        instancedata_sha256="abc",
        nl_archive_sha256="def",
        nl_file_count=42,
        pinned=False,
    )
    save_manifest(m, tmp_path, "v1")
    loaded = load_manifest(tmp_path, "v1")
    assert loaded is not None
    assert loaded.version == "v1"
    assert loaded.nl_file_count == 42
    assert loaded.instancedata_sha256 == "abc"


def test_fetch_uses_pinned_sha_mismatch(tmp_path, monkeypatch):
    """If a pinned sha exists and the downloaded bytes mismatch, fetch raises."""
    from scripts import fetch_minlplib

    # Stub the pinned-versions loader to claim a specific sha
    monkeypatch.setattr(
        fetch_minlplib, "_load_pinned_versions",
        lambda: {"v1": {"instancedata_sha256": "0" * 64}},
    )

    fake_csv = b"name,probtype\nfoo,MINLP\n"

    def fake_download(url, dest=None):
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fake_csv)
        return fake_csv

    monkeypatch.setattr(fetch_minlplib, "_download", fake_download)

    with pytest.raises(RuntimeError, match="instancedata.csv sha256 mismatch"):
        fetch_minlplib.fetch(
            cache_dir=tmp_path, version="v1",
            instances=[],          # avoid touching nl archive
            skip_archive=True,
        )


# ──────────────────────────────────────────────────────────────────
# minlplib_data
# ──────────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    keys = list({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_load_instance_data_basic(tmp_path):
    from utils.minlplib_data import load_instance_data, known_optima_from_index

    csv_path = tmp_path / "instancedata.csv"
    _write_csv(csv_path, [
        {"name": "ex1221", "probtype": "MINLP", "nvars": "5", "ncons": "5",
         "objsense": "min", "primalbound": "7.66718007", "dualbound": "7.66718007",
         "provenoptimal": "True"},
        {"name": "big_open", "probtype": "MINLP", "nvars": "200", "ncons": "300",
         "objsense": "min", "primalbound": "10.5", "dualbound": "1.0",
         "provenoptimal": "False"},
    ])

    index = load_instance_data(csv_path)
    assert "ex1221" in index
    assert index["ex1221"].n_vars == 5
    assert index["ex1221"].proven_optimal is True
    assert index["ex1221"].known_optimum == pytest.approx(7.66718007)

    assert index["big_open"].proven_optimal is False
    assert index["big_open"].known_optimum is None  # unproven => no reference

    optima = known_optima_from_index(index)
    assert set(optima.keys()) == {"ex1221"}


def test_instancemeta_buckets():
    from utils.minlplib_data import InstanceMeta

    meta = InstanceMeta(name="x", probtype="MINLP", n_vars=8)
    assert meta.category_bucket == "MINLP"
    assert meta.size_bucket == "<=10"
    assert InstanceMeta(name="y", n_vars=50, probtype="MIQP").size_bucket == "<=100"
    assert InstanceMeta(name="z", n_vars=1000, probtype="LP").size_bucket == ">500"


def test_score_result_outcomes():
    from benchmarks.metrics import SolveResult, SolveStatus
    from utils.minlplib_data import (
        OUTCOME_ERROR,
        OUTCOME_FEASIBLE,
        OUTCOME_INCORRECT,
        OUTCOME_OPTIMAL,
        OUTCOME_TIMEOUT,
        InstanceMeta,
        score_result,
    )

    meta = InstanceMeta(name="ex1221", proven_optimal=True, primal_bound=10.0)

    # exact match -> optimal_proven
    r = SolveResult(instance="ex1221", solver="discopt",
                    status=SolveStatus.OPTIMAL, objective=10.0)
    assert score_result(r, meta) == OUTCOME_OPTIMAL

    # wildly wrong objective -> incorrect
    r = SolveResult(instance="ex1221", solver="discopt",
                    status=SolveStatus.OPTIMAL, objective=11.0)
    assert score_result(r, meta) == OUTCOME_INCORRECT

    # feasible only
    r = SolveResult(instance="ex1221", solver="discopt",
                    status=SolveStatus.FEASIBLE, objective=10.0)
    assert score_result(r, meta) == OUTCOME_FEASIBLE

    # timeout
    r = SolveResult(instance="ex1221", solver="discopt",
                    status=SolveStatus.TIME_LIMIT)
    assert score_result(r, meta) == OUTCOME_TIMEOUT

    # error
    r = SolveResult(instance="ex1221", solver="discopt",
                    status=SolveStatus.ERROR)
    assert score_result(r, meta) == OUTCOME_ERROR

    # No reference + claimed optimal -> still optimal (we trust solver)
    r = SolveResult(instance="open", solver="discopt",
                    status=SolveStatus.OPTIMAL, objective=42.0)
    assert score_result(r, None) == OUTCOME_OPTIMAL


# ──────────────────────────────────────────────────────────────────
# scaled_runner — without invoking discopt
# ──────────────────────────────────────────────────────────────────


def test_scaled_runner_handles_missing_nl(tmp_path):
    """A non-existent .nl file should produce an error payload, not crash the runner."""
    from benchmarks.scaled_runner import ScaledConfig, ScaledRunner

    cfg = ScaledConfig(
        suite_name="test",
        out_dir=tmp_path,
        time_limit=5.0,
        mem_limit_mb=0,
        grace_seconds=2.0,
        n_workers=1,
        solver_name="discopt",
    )
    runner = ScaledRunner(cfg)

    bogus = tmp_path / "nonexistent.nl"
    runner.run([("nonexistent", bogus)])

    result_file = tmp_path / "instances" / "nonexistent.json"
    assert result_file.exists()
    payload = json.loads(result_file.read_text())
    assert payload.get("status") in {"error", "time_limit"} or "_error" in payload


def test_scaled_runner_resumability(tmp_path):
    """Pre-existing result files are not re-run when skip_existing is True."""
    from benchmarks.scaled_runner import ScaledConfig, ScaledRunner

    cfg = ScaledConfig(
        suite_name="test", out_dir=tmp_path, time_limit=5.0,
        n_workers=1, skip_existing=True,
    )
    runner = ScaledRunner(cfg)

    # Pre-write a result for "already_done"
    existing = tmp_path / "instances" / "already_done.json"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps({
        "instance": "already_done", "solver": "discopt",
        "status": "optimal", "objective": 1.0, "bound": 1.0,
        "wall_time": 0.5, "node_count": 3,
    }))

    bogus = tmp_path / "nonexistent.nl"
    runner.run([("already_done", bogus)])  # should skip, not dispatch

    # File is unchanged
    payload = json.loads(existing.read_text())
    assert payload["status"] == "optimal"
    assert payload["objective"] == 1.0


def test_scaled_runner_collect(tmp_path):
    from benchmarks.metrics import SolveStatus
    from benchmarks.scaled_runner import ScaledConfig, ScaledRunner

    cfg = ScaledConfig(suite_name="test", out_dir=tmp_path, time_limit=5.0, n_workers=1)
    runner = ScaledRunner(cfg)

    (tmp_path / "instances").mkdir(exist_ok=True)
    for i, status in enumerate(["optimal", "time_limit", "error"]):
        (tmp_path / "instances" / f"inst{i}.json").write_text(json.dumps({
            "instance": f"inst{i}", "solver": "discopt", "status": status,
            "objective": 1.0 if status == "optimal" else None,
            "bound": 1.0 if status == "optimal" else None,
            "wall_time": 0.1 + i, "node_count": 0,
        }))

    bench = runner.collect()
    results = bench.get_results("discopt")
    assert len(results) == 3
    statuses = {r.instance: r.status for r in results}
    assert statuses["inst0"] == SolveStatus.OPTIMAL
    assert statuses["inst1"] == SolveStatus.TIME_LIMIT
    assert statuses["inst2"] == SolveStatus.ERROR


# ──────────────────────────────────────────────────────────────────
# reporting (minlplib-style)
# ──────────────────────────────────────────────────────────────────


def test_generate_minlplib_report(tmp_path):
    from benchmarks.metrics import BenchmarkResults, SolveResult, SolveStatus
    from utils.minlplib_data import InstanceMeta
    from utils.reporting import generate_minlplib_report

    bench = BenchmarkResults(suite="full", timestamp="2026-05-23T00:00:00")
    bench.add_result(SolveResult(instance="a", solver="discopt",
                                 status=SolveStatus.OPTIMAL, objective=10.0,
                                 bound=10.0, wall_time=0.5))
    bench.add_result(SolveResult(instance="b", solver="discopt",
                                 status=SolveStatus.OPTIMAL, objective=12.0,  # wrong
                                 bound=12.0, wall_time=1.5))
    bench.add_result(SolveResult(instance="c", solver="discopt",
                                 status=SolveStatus.TIME_LIMIT, wall_time=3600))

    index = {
        "a": InstanceMeta(name="a", probtype="MINLP", n_vars=5,
                          proven_optimal=True, primal_bound=10.0),
        "b": InstanceMeta(name="b", probtype="MIQP", n_vars=20,
                          proven_optimal=True, primal_bound=10.0),
        "c": InstanceMeta(name="c", probtype="MINLP", n_vars=200,
                          proven_optimal=False, primal_bound=99.0),
    }

    out = tmp_path / "report.md"
    text = generate_minlplib_report(bench, index, output_path=out)
    assert out.exists()
    # Spot-check the standard MINLPLib columns are present
    assert "#optimal" in text
    assert "#incorrect" in text
    assert "Incorrect results (must be zero for release)" in text
    assert "`b`" in text   # the wrong-answer instance is listed
    assert "TOTAL" in text


# ──────────────────────────────────────────────────────────────────
# baseline gate
# ──────────────────────────────────────────────────────────────────


def _make_bench(suite: str, entries: list[tuple[str, str, float]]):
    """entries = [(instance, status, objective_or_nan)]"""
    from benchmarks.metrics import BenchmarkResults, SolveResult, SolveStatus

    bench = BenchmarkResults(suite=suite, timestamp="2026-05-23")
    for inst, status, obj in entries:
        bench.add_result(SolveResult(
            instance=inst, solver="discopt",
            status=SolveStatus(status),
            objective=None if obj != obj else obj,  # nan check
            bound=None if obj != obj else obj,
            wall_time=1.0,
        ))
    return bench


def test_baseline_pin_load_roundtrip(tmp_path):
    from utils.baseline import load_baseline, save_baseline

    bench = _make_bench("full", [("a", "optimal", 1.0), ("b", "time_limit", float("nan"))])
    save_baseline(bench, "full", baselines_dir=tmp_path)

    loaded = load_baseline("full", baselines_dir=tmp_path)
    assert loaded is not None
    assert {r.instance for r in loaded.get_results("discopt")} == {"a", "b"}


def test_baseline_passes_when_unchanged(tmp_path):
    from utils.baseline import compare_to_baseline

    base = _make_bench("full", [("a", "optimal", 1.0), ("b", "optimal", 2.0)])
    cur = _make_bench("full", [("a", "optimal", 1.0), ("b", "optimal", 2.0)])
    report = compare_to_baseline(cur, base, "full", known_optima={"a": 1.0, "b": 2.0})
    assert report.passed
    assert report.failures == []
    assert report.solved_delta == 0


def test_baseline_fails_on_solved_count_drop(tmp_path):
    from utils.baseline import compare_to_baseline

    base = _make_bench("full", [("a", "optimal", 1.0), ("b", "optimal", 2.0)])
    cur = _make_bench("full", [("a", "optimal", 1.0), ("b", "time_limit", float("nan"))])
    report = compare_to_baseline(cur, base, "full", known_optima={"a": 1.0, "b": 2.0})
    assert not report.passed
    assert report.solved_delta == -1
    assert "lost" in " ".join(report.lost_instances) or "b" in report.lost_instances
    assert any("solved count dropped" in f for f in report.failures)


def test_baseline_fails_on_new_incorrect(tmp_path):
    from utils.baseline import compare_to_baseline

    base = _make_bench("full", [("a", "optimal", 1.0)])
    cur = _make_bench("full", [("a", "optimal", 999.0)])  # wrong answer
    report = compare_to_baseline(cur, base, "full", known_optima={"a": 1.0})
    assert not report.passed
    assert "a" in report.new_incorrect_instances
    assert any("incorrect" in f for f in report.failures)


def test_baseline_strict_time(tmp_path):
    from benchmarks.metrics import BenchmarkResults, SolveResult, SolveStatus
    from utils.baseline import compare_to_baseline

    base = BenchmarkResults(suite="full", timestamp="x")
    base.add_result(SolveResult(instance="a", solver="discopt",
                                status=SolveStatus.OPTIMAL,
                                objective=1.0, bound=1.0, wall_time=1.0))
    cur = BenchmarkResults(suite="full", timestamp="x")
    cur.add_result(SolveResult(instance="a", solver="discopt",
                               status=SolveStatus.OPTIMAL,
                               objective=1.0, bound=1.0, wall_time=5.0))

    # default: time regressions are reported but don't fail
    report = compare_to_baseline(cur, base, "full", known_optima={"a": 1.0})
    assert report.passed
    assert report.time_regressions and report.time_regressions[0]["ratio"] == 5.0

    # strict_time: fails
    report = compare_to_baseline(cur, base, "full", known_optima={"a": 1.0}, strict_time=True)
    assert not report.passed
    assert any("regressed" in f for f in report.failures)


def test_format_gate_report():
    from utils.baseline import GateReport, format_gate_report

    r = GateReport(suite="full", solver="discopt", passed=False,
                   baseline_solved=100, current_solved=98, solved_delta=-2,
                   new_incorrect_instances=["bad1"],
                   failures=["solved count dropped by 2"])
    text = format_gate_report(r)
    assert "FAIL" in text
    assert "bad1" in text
    assert "delta=-2" in text
