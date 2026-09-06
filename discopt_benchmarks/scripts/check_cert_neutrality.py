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
  * a **host-speed calibration**, measured on the **unrouted** instances whose
    ``node_count`` reproduced *exactly* — same tree and no time spent outside it
    means the two runs did the same work, so their wall-time ratio is a clean speed
    proxy. (Equal node counts alone are not enough: an auto-routed algorithm that
    abstains at a budget checkpoint burns wall the tree does not account for. That
    is #1134's Cause 2, and it is why the routed rows are dropped.) ``status``
    violations are annotated with it because this panel's status verdicts are wall-clock
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
from utils.cert_neutrality import (  # noqa: E402
    check_neutrality,
    load_baseline,
    wall_limited_rows,
)

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
    """Median ``wall_new / wall_baseline`` over **unrouted** instances with an
    identical node_count.

    Returns ``(ratio, n_samples)``; ``ratio`` is None when fewer than
    ``_MIN_CALIBRATION_SAMPLES`` instances qualify. Equal node counts are *necessary*
    for this to be a speed measurement rather than a work measurement — the two runs
    explored the same tree, so the wall ratio is the machines'.

    They are not *sufficient*, which #1134's own Cause 2 is the proof of: an
    auto-routed algorithm that runs to a budget checkpoint and then abstains spends
    wall outside the counted tree, so `cvxnonsep_nsig30` (165 → 165 nodes,
    1.12 s → 49.6 s), `fac2` (39 → 39, 2.77 → 38.9) and `cvxnonsep_psig30` (89 → 89,
    0.41 → 8.5) all clear the equal-node filter carrying a 14-44x inflation that is
    not the box. Worse, the route's price is a *fraction of the wall-clock budget*
    (`_CONVEX_ROUTE_BUDGET_FRACTION`), i.e. the same number of seconds on a fast box
    and a slow one, so a row routed on **both** sides compresses the ratio toward 1
    instead of inflating it. Either way the row measures the router, not the
    machine, and is excluded — which is what `algorithm_route` (added by #1134 to
    `SolveResult`) is here for. Reference rows predate the field and read as
    unrouted, which is correct: they were generated before the route was reachable.

    Baseline walls at or below 0.05 s are excluded — at that scale the row is
    process noise, not throughput.
    """
    ratios = []
    for inst, base in baseline.items():
        new = new_rows.get(inst)
        if new is None or new.get("node_count") != base.get("node_count"):
            continue
        if new.get("algorithm_route") or base.get("algorithm_route"):
            continue
        wb, wn = base.get("wall_time"), new.get("wall_time")
        if wb is None or wn is None or wb <= 0.05:
            continue
        ratios.append(wn / wb)
    if len(ratios) < _MIN_CALIBRATION_SAMPLES:
        return None, len(ratios)
    return statistics.median(ratios), len(ratios)


def meta_describes_the_committed_reference(meta: dict) -> bool:
    """Whether ``meta`` is the provenance of the reference now on disk.

    ``gen_cert_baseline`` writes the meta on **every** run, including one whose
    write the shrink guard refused — deliberately, because a run that shrank
    coverage is the run whose evidence matters. So the meta on disk is the
    reference's provenance only when that run actually wrote the reference.
    ``baseline_written`` records it; for a meta written before that field existed
    it is re-derived from the guard's own condition (a run loses coverage and does
    not pass ``--allow-shrink`` ⇒ the write was refused).
    """
    written = meta.get("baseline_written")
    if written is not None:
        return bool(written)
    return not (meta.get("coverage_lost") or []) or bool(meta.get("allow_shrink"))


def provenance_lines(meta: dict | None) -> list[str]:
    """The provenance report for ``meta`` (None when the meta file is absent).

    Pure so the *stale/refused* case is testable without touching the committed
    ``docs/dev/data`` files. Reporting-only: nothing here decides a verdict.
    """
    if meta is None:
        return [
            f"  reference provenance: NONE ({_CERT_BASELINE_META.name} absent) — this "
            "reference predates the\n    #1134 provenance record, so the commit it was "
            "generated at is not recoverable from the tree."
        ]
    host = meta.get("host") or {}
    stamp = (
        f"commit {meta.get('commit')} at {meta.get('generated_at')}, "
        f"budget {meta.get('time_limit')}s (default; perf-panel instances run at their "
        f"own), host {host.get('platform')} ({host.get('cpu_count')} cpu)"
    )
    lost = meta.get("coverage_lost") or []
    if not meta_describes_the_committed_reference(meta):
        # The meta is a REFUSED run's. Attributing its commit and host to the
        # reference on disk would be a confidently wrong answer to exactly the
        # question #1134 exists to make answerable — worse than the missing answer
        # the NONE branch above gives. Say what it is instead.
        return [
            f"  reference provenance: UNKNOWN — {_CERT_BASELINE_META.name} records a "
            "REFUSED regeneration",
            f"    ({stamp}),",
            f"    which dropped {len(lost)} instance(s) the reference covered "
            f"({', '.join(lost)}) and so did NOT",
            "    overwrite cert-baseline.jsonl. The committed reference is OLDER than "
            "this record and its",
            "    own provenance is not recoverable from the tree.",
        ]
    out = [f"  reference provenance: {stamp}"]
    if lost:
        out.append(
            f"  reference was written under --allow-shrink, deliberately losing "
            f"{len(lost)} instance(s): {', '.join(lost)}"
        )
    return out


def _print_reference_provenance() -> None:
    """Print who generated the committed reference, so staleness is a lookup.

    Absent for a reference generated before #1134 added the record; that absence is
    itself the finding, and is reported rather than passed over.
    """
    meta = json.loads(_CERT_BASELINE_META.read_text()) if _CERT_BASELINE_META.exists() else None
    for line in provenance_lines(meta):
        print(line)


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
    # #1187: a row that ended on the wall clock in BOTH arms is not evidence either
    # way. ``deterministic=True`` neutralizes the role-2 budgets, but ``time_limit``
    # is role 1 and stays live by design, so on a wall-limited run the terminating
    # condition IS the clock and the two arms did different amounts of work. Reading
    # neutrality off such a row is reading noise (#1180 manufactured a reproducible
    # "0.516x regression" that way, on 13 of 66 rows). Excluded, and REPORTED as
    # unmeasured — the verdict below is only over the rows actually compared.
    unmeasured = wall_limited_rows(new_rows, baseline, budgets=budgets)
    violations = check_neutrality(
        new_rows, baseline, known_perf_gated=_KNOWN_PERF_GATED, exclude=unmeasured
    )
    ratio, n_cal = host_speed_ratio(new_rows, baseline)
    print("\n─── neutrality result ───")
    if ratio is None:
        print(
            f"  host-speed calibration: unavailable ({n_cal} instance(s) qualified — "
            f"node_count reproduced exactly, unrouted, baseline wall above noise; need "
            f"{_MIN_CALIBRATION_SAMPLES})"
        )
    else:
        print(
            f"  host-speed calibration: this box is {ratio:.2f}x the reference machine's "
            f"wall on {n_cal} unrouted instance(s) that reproduced their node_count exactly"
        )
    if unmeasured:
        print(f"  {len(unmeasured)} instance(s) UNMEASURED (#1187) — not compared:")
        for inst, why in sorted(unmeasured.items()):
            print(f"    {inst:20s} {why}")
    measured = len(baseline) - len(unmeasured)
    if not violations:
        print(f"  NEUTRAL over the {measured} instance(s) compared (objective to tol, "
              "still optimal, node_count not materially worse).")
        return 0
    print(f"  {len(violations)} VIOLATION(S):")
    for v in violations:
        note = ""
        if v.kind == "status" and ratio is not None and ratio > 1.0:
            note = (
                f"  [wall-clock verdict; this box runs {ratio:.2f}x the reference's "
                "wall on equal-node unrouted work]"
            )
        print(f"    {v.instance:20s} [{v.kind}] {v.detail}{note}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
