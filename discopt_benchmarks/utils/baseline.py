"""Pinned-baseline comparison for the MINLPLib suites.

A baseline is a saved ``BenchmarkResults`` JSON checked into the repo under
``discopt_benchmarks/baselines/<suite>.json``. The gate fails if the current
run regresses any of:

  * **solved-count** drops by more than ``solved_count_tolerance`` instances,
  * **incorrect-count** rises above zero (new wrong-answer regressions),
  * **time** on any commonly-solved instance regresses by more than
    ``time_factor`` (default 1.5x, surfaced as a warning by default).

The strict checks are solved-count and incorrect-count. Time regressions are
informational unless ``--strict-time`` is passed.

Layout:

    discopt_benchmarks/
      baselines/
        full.json
        comparison.json
        nightly.json

Pinning a baseline:

    python run_benchmarks.py --suite full --pin-baseline
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.metrics import (
    BenchmarkResults,
    SolveResult,
    solved_count,
)


BASELINES_DIR = Path(__file__).parent.parent / "baselines"


def baseline_path(suite: str, baselines_dir: Path = BASELINES_DIR) -> Path:
    return baselines_dir / f"{suite}.json"


def save_baseline(results: BenchmarkResults, suite: str, baselines_dir: Path = BASELINES_DIR) -> Path:
    """Pin the given results as the baseline for ``suite``."""
    path = baseline_path(suite, baselines_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.save(path)
    return path


def load_baseline(suite: str, baselines_dir: Path = BASELINES_DIR) -> BenchmarkResults | None:
    path = baseline_path(suite, baselines_dir)
    if not path.exists():
        return None
    return BenchmarkResults.load(path)


@dataclass
class GateReport:
    """Outcome of comparing current run to baseline."""

    suite: str
    solver: str
    passed: bool
    baseline_solved: int = 0
    current_solved: int = 0
    solved_delta: int = 0
    baseline_incorrect: int = 0
    current_incorrect: int = 0
    incorrect_delta: int = 0
    new_incorrect_instances: list[str] = field(default_factory=list)
    lost_instances: list[str] = field(default_factory=list)        # were solved, now aren't
    gained_instances: list[str] = field(default_factory=list)      # weren't solved, now are
    time_regressions: list[dict] = field(default_factory=list)     # per-instance details
    failures: list[str] = field(default_factory=list)              # human-readable failure reasons

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _incorrect_instances(
    results: list[SolveResult],
    known_optima: dict[str, float],
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> list[str]:
    out = []
    for r in results:
        if not r.is_solved or r.objective is None:
            continue
        if r.instance not in known_optima:
            continue
        ref = known_optima[r.instance]
        if abs(r.objective - ref) > abs_tol + rel_tol * abs(ref):
            out.append(r.instance)
    return out


def compare_to_baseline(
    current: BenchmarkResults,
    baseline: BenchmarkResults,
    suite: str,
    solver: str = "discopt",
    known_optima: dict[str, float] | None = None,
    solved_count_tolerance: int = 0,
    time_factor: float = 1.5,
    strict_time: bool = False,
) -> GateReport:
    """Compare the current run to a pinned baseline and decide pass/fail.

    Pass requires both:

      * ``current_solved >= baseline_solved - solved_count_tolerance``
      * no new ``incorrect`` instances (current incorrect ⊆ baseline incorrect)

    When ``strict_time=True`` any instance > ``time_factor``× slower also fails.
    """
    known_optima = known_optima or {}

    cur_results = current.get_results(solver)
    base_results = baseline.get_results(solver)

    cur_solved_set = {r.instance for r in cur_results if r.is_solved}
    base_solved_set = {r.instance for r in base_results if r.is_solved}

    cur_solved = len(cur_solved_set)
    base_solved = len(base_solved_set)

    cur_incorrect = _incorrect_instances(cur_results, known_optima)
    base_incorrect = _incorrect_instances(base_results, known_optima)
    new_incorrect = sorted(set(cur_incorrect) - set(base_incorrect))

    lost = sorted(base_solved_set - cur_solved_set)
    gained = sorted(cur_solved_set - base_solved_set)

    # Per-instance time regressions on commonly-solved set
    cur_times = {r.instance: r.wall_time for r in cur_results if r.is_solved}
    base_times = {r.instance: r.wall_time for r in base_results if r.is_solved}
    common = set(cur_times) & set(base_times)
    time_regs = []
    for inst in sorted(common):
        b = base_times[inst]
        c = cur_times[inst]
        if b > 1e-3 and c / b > time_factor:
            time_regs.append({"instance": inst, "baseline_time": b, "current_time": c, "ratio": c / b})
    time_regs.sort(key=lambda x: x["ratio"], reverse=True)

    failures: list[str] = []
    if cur_solved < base_solved - solved_count_tolerance:
        failures.append(
            f"solved count dropped by {base_solved - cur_solved} "
            f"(baseline={base_solved}, current={cur_solved}, tolerance={solved_count_tolerance})"
        )
    if new_incorrect:
        failures.append(f"{len(new_incorrect)} new incorrect result(s): {new_incorrect[:10]}")
    if strict_time and time_regs:
        failures.append(
            f"{len(time_regs)} instance(s) regressed >{time_factor:g}x in time "
            f"(worst: {time_regs[0]['instance']} at {time_regs[0]['ratio']:.2f}x)"
        )

    return GateReport(
        suite=suite,
        solver=solver,
        passed=not failures,
        baseline_solved=base_solved,
        current_solved=cur_solved,
        solved_delta=cur_solved - base_solved,
        baseline_incorrect=len(base_incorrect),
        current_incorrect=len(cur_incorrect),
        incorrect_delta=len(cur_incorrect) - len(base_incorrect),
        new_incorrect_instances=new_incorrect,
        lost_instances=lost,
        gained_instances=gained,
        time_regressions=time_regs,
        failures=failures,
    )


def format_gate_report(report: GateReport, time_regression_limit: int = 10) -> str:
    """Render a GateReport as human-readable text."""
    lines = []
    status = "PASS" if report.passed else "FAIL"
    lines.append(f"=== Baseline gate: {report.suite} / {report.solver} — {status} ===")
    lines.append(f"  solved:    baseline={report.baseline_solved}  current={report.current_solved}  delta={report.solved_delta:+d}")
    lines.append(f"  incorrect: baseline={report.baseline_incorrect}  current={report.current_incorrect}  delta={report.incorrect_delta:+d}")
    if report.new_incorrect_instances:
        lines.append(f"  NEW INCORRECT ({len(report.new_incorrect_instances)}):")
        for inst in report.new_incorrect_instances[:20]:
            lines.append(f"    - {inst}")
    if report.lost_instances:
        lines.append(f"  Lost (previously solved, now not) [{len(report.lost_instances)}]:")
        for inst in report.lost_instances[:20]:
            lines.append(f"    - {inst}")
    if report.gained_instances:
        lines.append(f"  Gained (newly solved) [{len(report.gained_instances)}]:")
        for inst in report.gained_instances[:20]:
            lines.append(f"    + {inst}")
    if report.time_regressions:
        lines.append(f"  Time regressions ({len(report.time_regressions)} shown, top {time_regression_limit}):")
        for entry in report.time_regressions[:time_regression_limit]:
            lines.append(
                f"    {entry['instance']:30s}  "
                f"{entry['baseline_time']:.2f}s -> {entry['current_time']:.2f}s  ({entry['ratio']:.2f}x)"
            )
    if report.failures:
        lines.append("  FAILURES:")
        for f in report.failures:
            lines.append(f"    * {f}")
    return "\n".join(lines)


def save_gate_report(report: GateReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return path
