"""Subprocess-isolated, resumable, parallel MINLPLib benchmark runner.

Each instance runs in its own ``python -m benchmarks._subprocess_worker`` so
a segfault, hang, or OOM in one solve cannot poison the whole sweep. Results
are streamed to disk one file per instance, which makes the run resumable
(skip instances whose result JSON already exists).

This is the runner you want for the ``full`` MINLPLib suite (~1700 instances,
hours-to-days wall time). For small sweeps the in-process
:class:`BenchmarkRunner` is faster and gives nicer console output.

Usage:

    from discopt_benchmarks.benchmarks.scaled_runner import ScaledRunner, ScaledConfig

    cfg = ScaledConfig(
        suite_name="full",
        out_dir=Path("reports/full_run_2026-05"),
        time_limit=3600,
        mem_limit_mb=8192,
        n_workers=8,
        solver_name="discopt",
        solver_options={},
    )
    runner = ScaledRunner(cfg)
    runner.run(instance_specs)  # iterable of (name, nl_path)
    results = runner.collect()  # -> BenchmarkResults
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from benchmarks.metrics import (
    BenchmarkResults,
    InstanceInfo,
    SolveResult,
    SolveStatus,
)


@dataclass
class ScaledConfig:
    """Configuration for the scaled, subprocess-isolated runner."""

    suite_name: str
    out_dir: Path
    time_limit: float = 3600.0       # wall-clock cap per instance (seconds)
    mem_limit_mb: int = 8192          # address-space cap per instance (MB)
    grace_seconds: float = 30.0       # parent waits this much past time_limit before SIGKILL
    n_workers: int = 1                # parallel workers (1 = serial)
    solver_name: str = "discopt"
    solver_options: dict = field(default_factory=dict)
    python_executable: str = ""       # defaults to sys.executable
    skip_existing: bool = True        # resumability: don't re-run instances whose .json exists

    def __post_init__(self) -> None:
        if not self.python_executable:
            self.python_executable = sys.executable


# ── Worker dispatch ────────────────────────────────────────────────────────


def _result_path(out_dir: Path, instance: str) -> Path:
    return out_dir / "instances" / f"{instance}.json"


def _run_one_subprocess(
    python_exe: str,
    instance: str,
    nl_path: str,
    time_limit: float,
    mem_limit_mb: int,
    grace_seconds: float,
    solver_name: str,
    solver_options: dict,
    out_path: Path,
) -> dict:
    """Spawn one worker, wait, kill on overrun, return the JSON payload.

    This function is what each pool worker calls. It blocks until the child
    completes or the wall-clock budget expires. The result JSON is always
    written to ``out_path`` (success or failure).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    options_json = json.dumps({**solver_options, "_solver_name": solver_name})

    cmd = [
        python_exe,
        "-m",
        "benchmarks._subprocess_worker",
        "--instance", instance,
        "--nl-path", nl_path,
        "--time-limit", str(time_limit),
        "--mem-limit-mb", str(mem_limit_mb),
        "--options-json", options_json,
        "--out-json", str(out_path),
    ]

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent.parent),  # so benchmarks._subprocess_worker resolves
        )
    except OSError as e:
        elapsed = time.monotonic() - start
        payload = _make_error_payload(instance, solver_name, f"spawn failed: {e}", elapsed)
        out_path.write_text(json.dumps(payload))
        return payload

    deadline = time_limit + grace_seconds
    try:
        stdout, stderr = proc.communicate(timeout=deadline)
        elapsed = time.monotonic() - start
    except subprocess.TimeoutExpired:
        # Kill the runaway child, then collect whatever it managed to write.
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        elapsed = time.monotonic() - start
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text())
                # Override status to time_limit (worker didn't finish cleanly).
                payload["status"] = SolveStatus.TIME_LIMIT.value
                payload["wall_time"] = elapsed
            except json.JSONDecodeError:
                payload = _make_timeout_payload(instance, solver_name, elapsed)
        else:
            payload = _make_timeout_payload(instance, solver_name, elapsed)
        out_path.write_text(json.dumps(payload))
        return payload

    if proc.returncode != 0:
        # Subprocess died (segfault, OOM via SIGKILL, uncaught exception).
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text())
            except json.JSONDecodeError:
                payload = _make_error_payload(
                    instance, solver_name,
                    f"worker exit {proc.returncode}; stderr={stderr[-500:]}",
                    elapsed,
                )
        else:
            payload = _make_error_payload(
                instance, solver_name,
                f"worker exit {proc.returncode}; stderr={stderr[-500:]}",
                elapsed,
            )
        out_path.write_text(json.dumps(payload))
        return payload

    if not out_path.exists():
        payload = _make_error_payload(
            instance, solver_name,
            f"worker exited 0 but wrote no result; stderr={stderr[-500:]}",
            elapsed,
        )
        out_path.write_text(json.dumps(payload))
        return payload

    return json.loads(out_path.read_text())


def _make_error_payload(instance: str, solver: str, msg: str, elapsed: float) -> dict:
    return {
        "instance": instance,
        "solver": solver,
        "status": SolveStatus.ERROR.value,
        "objective": None,
        "bound": None,
        "wall_time": elapsed,
        "node_count": 0,
        "_error": msg,
    }


def _make_timeout_payload(instance: str, solver: str, elapsed: float) -> dict:
    return {
        "instance": instance,
        "solver": solver,
        "status": SolveStatus.TIME_LIMIT.value,
        "objective": None,
        "bound": None,
        "wall_time": elapsed,
        "node_count": 0,
        "_timeout": True,
    }


# ── ScaledRunner ───────────────────────────────────────────────────────────


class ScaledRunner:
    """Drives ``n_workers`` parallel subprocess solves with resumability."""

    def __init__(self, config: ScaledConfig):
        self.config = config
        # Resolve to absolute: the worker subprocess runs with cwd=discopt_benchmarks/
        # so a relative out_dir would land in the wrong place.
        self.out_dir = Path(config.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "instances").mkdir(parents=True, exist_ok=True)

    def run(
        self,
        instances: Iterable[tuple[str, Path]],
        instance_info: dict[str, InstanceInfo] | None = None,
    ) -> None:
        """Solve each ``(name, nl_path)`` and stream results to disk.

        Skips any instance whose result JSON already exists when
        ``config.skip_existing`` is True. Safe to call multiple times.
        """
        cfg = self.config
        instances = list(instances)

        # Filter out already-done instances if resuming.
        pending: list[tuple[str, Path]] = []
        done_count = 0
        for name, nl in instances:
            if cfg.skip_existing and _result_path(self.out_dir, name).exists():
                done_count += 1
                continue
            pending.append((name, nl))

        total = len(instances)
        print(
            f"\n[scaled-runner] suite={cfg.suite_name} "
            f"total={total} done={done_count} pending={len(pending)} "
            f"workers={cfg.n_workers} time_limit={cfg.time_limit}s "
            f"mem_limit={cfg.mem_limit_mb}MB"
        )

        # Write the suite manifest now so the run is identifiable even before
        # any instance finishes.
        self._write_run_manifest(total=total, pending=len(pending))

        if cfg.n_workers <= 1:
            self._run_serial(pending, total, done_count)
        else:
            self._run_parallel(pending, total, done_count)

        if instance_info is not None:
            self._save_instance_info(instance_info)

    def _run_serial(
        self,
        pending: list[tuple[str, Path]],
        total: int,
        done_offset: int,
    ) -> None:
        cfg = self.config
        for i, (name, nl) in enumerate(pending, 1):
            self._dispatch(name, nl)
            print(f"  [{done_offset + i}/{total}] {name}")

    def _run_parallel(
        self,
        pending: list[tuple[str, Path]],
        total: int,
        done_offset: int,
    ) -> None:
        cfg = self.config
        # Use a ProcessPoolExecutor purely as a job scheduler — each task
        # itself spawns a fresh subprocess via Popen for isolation. The pool
        # worker just blocks on Popen.communicate().
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
            futures = {
                pool.submit(
                    _run_one_subprocess,
                    cfg.python_executable,
                    name,
                    str(nl),
                    cfg.time_limit,
                    cfg.mem_limit_mb,
                    cfg.grace_seconds,
                    cfg.solver_name,
                    cfg.solver_options,
                    _result_path(self.out_dir, name),
                ): name
                for name, nl in pending
            }
            done_now = 0
            for fut in as_completed(futures):
                name = futures[fut]
                done_now += 1
                try:
                    payload = fut.result()
                    status = payload.get("status", "?")
                    wt = payload.get("wall_time", float("inf"))
                    print(f"  [{done_offset + done_now}/{total}] {name:40s} {status:>12s} {wt:>8.2f}s")
                except Exception as e:  # noqa: BLE001
                    print(f"  [{done_offset + done_now}/{total}] {name:40s} POOL-ERROR: {e}")

    def _dispatch(self, name: str, nl: Path) -> dict:
        cfg = self.config
        return _run_one_subprocess(
            cfg.python_executable,
            name,
            str(nl),
            cfg.time_limit,
            cfg.mem_limit_mb,
            cfg.grace_seconds,
            cfg.solver_name,
            cfg.solver_options,
            _result_path(self.out_dir, name),
        )

    def _write_run_manifest(self, total: int, pending: int) -> None:
        manifest = {
            "suite_name": self.config.suite_name,
            "started_at": datetime.now().isoformat(),
            "time_limit": self.config.time_limit,
            "mem_limit_mb": self.config.mem_limit_mb,
            "n_workers": self.config.n_workers,
            "solver_name": self.config.solver_name,
            "solver_options": self.config.solver_options,
            "total_instances": total,
            "pending_at_start": pending,
        }
        (self.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def _save_instance_info(self, info: dict[str, InstanceInfo]) -> None:
        from dataclasses import asdict
        payload = {name: asdict(meta) for name, meta in info.items()}
        (self.out_dir / "instance_info.json").write_text(json.dumps(payload, indent=2))

    # ── Result collection ──

    def collect(self) -> BenchmarkResults:
        """Aggregate per-instance result files into a BenchmarkResults."""
        results = BenchmarkResults(
            suite=self.config.suite_name,
            timestamp=datetime.now().isoformat(),
        )

        info_path = self.out_dir / "instance_info.json"
        if info_path.exists():
            for name, raw in json.loads(info_path.read_text()).items():
                results.instance_info[name] = InstanceInfo(**raw)

        for path in sorted((self.out_dir / "instances").glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            # Strip worker-private keys before reconstructing the dataclass.
            clean = {k: v for k, v in payload.items() if not k.startswith("_")}
            try:
                sr = SolveResult.from_dict(clean)
            except Exception:
                continue
            results.add_result(sr)

        return results

    def consolidated_path(self) -> Path:
        """Path where collect() will write the consolidated BenchmarkResults JSON."""
        return self.out_dir / "results.json"

    def save_consolidated(self) -> Path:
        results = self.collect()
        path = self.consolidated_path()
        results.save(path)
        return path

    def clean(self) -> None:
        """Delete all cached per-instance results (forces a full re-run)."""
        shutil.rmtree(self.out_dir / "instances", ignore_errors=True)
        (self.out_dir / "instances").mkdir(parents=True, exist_ok=True)
