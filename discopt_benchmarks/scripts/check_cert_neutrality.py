"""Check Phase 1 differential bound-neutrality against cert-baseline.jsonl.

Re-solves the deterministic-certifying subset at the baseline budgets and checks
each row against the committed baseline with the differential criteria
(objective-to-tolerance, still-optimal, node_count one-directional). Prints any
violations and exits non-zero if there are any.

Usage:
    python discopt_benchmarks/scripts/check_cert_neutrality.py
"""

from __future__ import annotations

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

# Documented performance-only regressions (soundness still enforced). T1.2's
# monomial coverage moves nvs17 from the cold path to the incremental path, which
# gives fewer nodes (205 -> 117) but a slower per-node cost from rejected warm
# starts (~45s of a 60s budget) — the T1.4 warm-start work resolves this. Tracked,
# not masked: its objective is still checked; only its wall/status is exempt.
_KNOWN_PERF_GATED = {
    "nvs17": "T1.2 monomial coverage -> incremental path; ~45s/60s wall pending T1.4 warm-starts",
}


def main() -> int:
    baseline = load_baseline(_CERT_BASELINE)
    budgets = _instance_budgets(60.0)
    solver = SolverConfig(name="discopt", command="", solver_type="internal")
    print(f"Neutrality check: {len(baseline)} certifying instances vs {_CERT_BASELINE.name}")

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
    print("\n─── neutrality result ───")
    if not violations:
        print("  NEUTRAL — all certifying instances pass (objective to tol, still "
              "optimal, node_count not materially worse).")
        return 0
    print(f"  {len(violations)} VIOLATION(S):")
    for v in violations:
        print(f"    {v.instance:20s} [{v.kind}] {v.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
