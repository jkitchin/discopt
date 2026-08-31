"""Check Phase 1 differential bound-neutrality against cert-baseline.jsonl.

Re-solves the deterministic-certifying subset at the baseline budgets and checks
each row against the committed baseline with the differential criteria
(objective-to-tolerance, still-optimal, node_count one-directional). Prints any
violations and exits non-zero if there are any.

Two reporting-only aids, added by #1134 because their absence turned a
one-lookup question into a bisect. Neither changes a verdict:

  * the **reference's provenance** (generating commit and host, from
    ``cert-baseline-meta.json``) is printed up front, so "is this reference stale
    relative to the tree?" is answerable without archaeology; and
  * a **host-speed calibration**, measured on the instances whose ``node_count``
    reproduced *exactly* — identical node counts mean the two runs did the same
    work, so their wall-time ratio is a clean speed proxy. ``status`` violations
    are annotated with it because this panel's status verdicts are wall-clock
    verdicts: an instance that certifies in 20 s of a 60 s budget on the reference
    machine is ``time_limit`` on a box 3x slower, with nothing wrong in the tree.
    The violation still stands (never weaken a guard to make a panel read green,
    CLAUDE.md §1) — the annotation only stops it being misread.

Usage:
    python discopt_benchmarks/scripts/check_cert_neutrality.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402
from scripts.gen_cert_baseline import _instance_budgets  # noqa: E402
from utils.cert_neutrality import check_neutrality, load_baseline  # noqa: E402

_CERT_BASELINE = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline.jsonl"
_CERT_BASELINE_META = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline-meta.json"

# Minimum number of exactly-reproducing instances before a host-speed ratio is
# reported. Below this the median is noise, and a speed claim on a handful of
# sub-second solves is exactly the kind of unfounded timing statement CLAUDE.md §9
# exists to stop.
_MIN_CALIBRATION_SAMPLES = 5

# Documented performance-only regressions (soundness still enforced). T1.2's
# monomial coverage moves nvs17 from the cold path to the incremental path, which
# gives fewer nodes (205 -> 117) but a slower per-node cost from rejected warm
# starts (~45s of a 60s budget) — the T1.4 warm-start work resolves this. Tracked,
# not masked: its objective is still checked; only its wall/status is exempt.
_KNOWN_PERF_GATED = {
    "nvs17": "T1.2 monomial coverage -> incremental path; ~45s/60s wall pending T1.4 warm-starts",
}


def host_speed_ratio(
    new_rows: dict[str, dict], baseline: dict[str, dict]
) -> tuple[float | None, int]:
    """Median ``wall_new / wall_baseline`` over instances with an identical node_count.

    Returns ``(ratio, n_samples)``; ``ratio`` is None when fewer than
    ``_MIN_CALIBRATION_SAMPLES`` instances qualify. Equal node counts are the
    condition that makes this a *speed* measurement rather than a work measurement:
    the two runs explored the same tree, so the wall ratio is the machines'.
    Baseline walls at or below 0.05 s are excluded — at that scale the row is
    process noise, not throughput.
    """
    ratios = []
    for inst, base in baseline.items():
        new = new_rows.get(inst)
        if new is None or new.get("node_count") != base.get("node_count"):
            continue
        wb, wn = base.get("wall_time"), new.get("wall_time")
        if wb is None or wn is None or wb <= 0.05:
            continue
        ratios.append(wn / wb)
    if len(ratios) < _MIN_CALIBRATION_SAMPLES:
        return None, len(ratios)
    return statistics.median(ratios), len(ratios)


def _print_reference_provenance() -> None:
    """Print who generated the committed reference, so staleness is a lookup.

    Absent for a reference generated before #1134 added the record; that absence is
    itself the finding, and is reported rather than passed over.
    """
    if not _CERT_BASELINE_META.exists():
        print(
            f"  reference provenance: NONE ({_CERT_BASELINE_META.name} absent) — this "
            "reference predates the\n    #1134 provenance record, so the commit it was "
            "generated at is not recoverable from the tree."
        )
        return
    meta = json.loads(_CERT_BASELINE_META.read_text())
    host = meta.get("host") or {}
    print(
        f"  reference provenance: commit {meta.get('commit')} at {meta.get('generated_at')}, "
        f"budget {meta.get('time_limit')}s, host {host.get('platform')} "
        f"({host.get('cpu_count')} cpu)"
    )
    lost = meta.get("coverage_lost") or []
    if lost:
        print(f"  reference recorded {len(lost)} instance(s) of coverage loss: {', '.join(lost)}")


def main() -> int:
    baseline = load_baseline(_CERT_BASELINE)
    budgets = _instance_budgets(60.0)
    solver = SolverConfig(name="discopt", command="", solver_type="internal")
    print(f"Neutrality check: {len(baseline)} certifying instances vs {_CERT_BASELINE.name}")
    _print_reference_provenance()

    new_rows: dict[str, dict] = {}
    for i, name in enumerate(sorted(baseline), 1):
        cfg = BenchmarkConfig(
            suite_name="cert-neutral", time_limit=int(budgets.get(name, 60)), num_runs=1,
            solvers=[solver],
        )
        res = BenchmarkRunner(cfg)._run_discopt(solver, name, 0)
        new_rows[name] = res.to_dict()
        b = baseline[name]
        d_obj = (
            abs(res.objective - b["objective"])
            if res.objective is not None and b["objective"] is not None
            else float("nan")
        )
        print(
            f"  [{i}/{len(baseline)}] {name:20s} {res.status.value:10s} "
            f"nodes {b['node_count']}->{res.node_count}  |Δobj|={d_obj:.2e}",
            flush=True,
        )

    for inst, why in _KNOWN_PERF_GATED.items():
        if inst in baseline:
            print(f"  [perf-gated] {inst}: {why} (soundness still checked)")
    violations = check_neutrality(new_rows, baseline, known_perf_gated=_KNOWN_PERF_GATED)
    ratio, n_cal = host_speed_ratio(new_rows, baseline)
    print("\n─── neutrality result ───")
    if ratio is None:
        print(
            f"  host-speed calibration: unavailable ({n_cal} instance(s) reproduced their "
            f"node_count exactly, need {_MIN_CALIBRATION_SAMPLES})"
        )
    else:
        print(
            f"  host-speed calibration: this box is {ratio:.2f}x the reference machine's "
            f"wall on {n_cal} instance(s) that reproduced their node_count exactly"
        )
    if not violations:
        print("  NEUTRAL — all certifying instances pass (objective to tol, still "
              "optimal, node_count not materially worse).")
        return 0
    print(f"  {len(violations)} VIOLATION(S):")
    for v in violations:
        note = ""
        if v.kind == "status" and ratio is not None and ratio > 1.0:
            note = (
                f"  [wall-clock verdict; this box runs {ratio:.2f}x the reference's "
                "wall on equal-node work]"
            )
        print(f"    {v.instance:20s} [{v.kind}] {v.detail}{note}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
