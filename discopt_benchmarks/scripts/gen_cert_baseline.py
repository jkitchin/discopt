"""Generate the certification baseline (cert:T0.5).

Runs discopt over the global50 panel (``config/baron_global50.txt``) plus the
perf panel (``perf/panel.py``) with the T0.1–T0.3 instrumentation on, and writes:

  * ``reports/cert0_<timestamp>.json`` — a ``BenchmarkResults`` the
    ``run_benchmarks.py --gate cert0`` check consumes; and
  * ``docs/dev/data/cert-baseline.jsonl`` — the committed reference the §0.2.5
    bound-neutrality check reads (node_count + objective per instance, plus the
    new root_gap/root_time fields).

Deterministic-ish: single run per instance, per-instance time limits (the perf
panel's own budgets where defined, else ``--time-limit``). The baseline is the
frozen reference, so re-generate it deliberately (not in CI).

Usage:
    python discopt_benchmarks/scripts/gen_cert_baseline.py [--time-limit 60]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
# Both roots: ``benchmarks.*`` resolves from _BENCH_ROOT; the ``perf`` package's
# __init__ imports the fully-qualified ``discopt_benchmarks.perf.*`` (needs the
# repo root).
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.metrics import (  # noqa: E402
    BenchmarkResults,
    incorrect_count,
    root_gap_populated_fraction,
)
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402
from perf.panel import PANEL  # noqa: E402

_GLOBAL50 = _BENCH_ROOT / "config" / "baron_global50.txt"
_CERT_BASELINE = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline.jsonl"
_CERT_OPTIMA = _REPO_ROOT / "docs" / "dev" / "data" / "cert-optima.json"


def _instance_budgets(default_tl: float) -> dict[str, float]:
    """Per-instance time limits: global50 at the default, perf-panel instances at
    their own (usually larger) panel budget so the flagships still expose signal."""
    names = _GLOBAL50.read_text().split()
    budgets = dict.fromkeys(names, default_tl)
    for inst in PANEL:
        budgets[inst.name] = max(budgets.get(inst.name, 0.0), float(inst.time_limit))
    return budgets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0, help="default per-instance seconds")
    args = ap.parse_args()

    budgets = _instance_budgets(args.time_limit)
    order = sorted(budgets)
    print(f"Certification baseline: {len(order)} instances (global50 + perf panel)")

    solver = SolverConfig(name="discopt", command="", solver_type="internal")
    results = BenchmarkResults(suite="cert0", timestamp=datetime.now().isoformat())

    for i, name in enumerate(order, 1):
        cfg = BenchmarkConfig(
            suite_name="cert0",
            time_limit=int(budgets[name]),
            num_runs=1,
            solvers=[solver],
        )
        runner = BenchmarkRunner(cfg)
        if runner._find_nl_file(name) is None:
            print(f"  [{i}/{len(order)}] SKIP {name} (not vendored)", flush=True)
            continue
        res = runner._run_discopt(solver, name, 0)
        results.add_result(res)
        rg = "None" if res.root_gap is None else f"{res.root_gap:.3g}"
        print(
            f"  [{i}/{len(order)}] {name:20s} {res.status.value:10s} "
            f"obj={res.objective} nodes={res.node_count} root_gap={rg} "
            f"({budgets[name]:.0f}s cap)",
            flush=True,
        )

    # reports/cert0_<ts>.json for the --gate cert0 consumer.
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    reports_dir = _REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    results_path = reports_dir / f"cert0_{ts}.json"
    results.save(results_path)

    # docs/dev/data/cert-baseline.jsonl — the committed neutrality reference.
    rows = sorted(results.get_results("discopt"), key=lambda r: r.instance)
    os.makedirs(_CERT_BASELINE.parent, exist_ok=True)
    with open(_CERT_BASELINE, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    # Summary + self-check.
    coverage = root_gap_populated_fraction(rows)
    optima = json.loads(_CERT_OPTIMA.read_text()) if _CERT_OPTIMA.exists() else {}
    incorrect = incorrect_count(rows, optima) if optima else None
    print("\n─── summary ───")
    print(f"  rows: {len(rows)}")
    print(f"  root_gap coverage: {coverage:.3f} (gate: >= 0.90)")
    print(f"  incorrect_count (vs {len(optima)} oracles): {incorrect}")
    print(f"  results: {results_path}")
    print(f"  baseline: {_CERT_BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
