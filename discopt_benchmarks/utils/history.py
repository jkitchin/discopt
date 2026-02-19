"""
Historical performance tracking for benchmark runs.

Stores one JSON object per line (JSONL) for each benchmark run,
enabling trend analysis and regression detection across commits.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from benchmarks.metrics import (
    BenchmarkResults,
    final_gap_stats,
    shifted_geometric_mean,
    solved_count,
)


def _get_git_sha() -> str:
    """Get current git short SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def append_to_history(
    suite: str,
    results: BenchmarkResults,
    history_dir: Path = Path("reports/history"),
) -> Path:
    """Append key metrics from a benchmark run to the suite history file.

    Returns the path to the history file.
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{suite}_history.jsonl"

    discopt_results = results.get_results("discopt")
    times = [r.wall_time for r in discopt_results if r.is_solved]
    gap_stats = final_gap_stats(discopt_results)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "git_sha": _get_git_sha(),
        "suite": suite,
        "total_instances": len(results.get_instances()),
        "solved_count": solved_count(discopt_results),
        "mean_time": float(shifted_geometric_mean(times)),
        "mean_gap": gap_stats["mean"],
        "median_gap": gap_stats["median"],
    }

    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return history_file


def load_history(
    suite: str,
    history_dir: Path = Path("reports/history"),
) -> list[dict]:
    """Load all historical entries for a suite."""
    history_file = history_dir / f"{suite}_history.jsonl"
    if not history_file.exists():
        return []
    entries = []
    with open(history_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_trend(
    suite: str,
    n: int = 10,
    history_dir: Path = Path("reports/history"),
) -> None:
    """Print the last N runs showing solve count and time trends."""
    entries = load_history(suite, history_dir)
    if not entries:
        print(f"No history found for suite '{suite}'")
        return

    recent = entries[-n:]
    print(f"\nHistory for suite '{suite}' (last {len(recent)} runs):")
    print(f"{'Date':<22s} {'SHA':<10s} {'Solved':>8s} {'SGM(s)':>10s} "
          f"{'Mean Gap':>10s} {'Instances':>10s}")
    print("-" * 75)

    for e in recent:
        ts = e["timestamp"][:19]
        sha = e.get("git_sha", "?")[:8]
        solved = e.get("solved_count", 0)
        sgm = e.get("mean_time", float("nan"))
        gap = e.get("mean_gap", float("nan"))
        total = e.get("total_instances", 0)

        sgm_str = f"{sgm:.2f}" if sgm < 1e6 else "inf"
        gap_str = f"{gap:.2%}" if gap == gap else "N/A"

        print(f"{ts:<22s} {sha:<10s} {solved:>8d} {sgm_str:>10s} "
              f"{gap_str:>10s} {total:>10d}")
    print()
