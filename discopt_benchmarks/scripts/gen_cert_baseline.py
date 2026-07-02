"""Generate the certification baseline (cert:T0.5).

Runs discopt over the global50 panel (``config/baron_global50.txt``) plus the
perf panel (``perf/panel.py``) with the T0.1–T0.3 instrumentation on, and writes:

  * ``reports/cert0_<timestamp>.json`` — a ``BenchmarkResults`` the
    ``run_benchmarks.py --gate cert0`` check consumes; and
  * ``docs/dev/data/cert-baseline.jsonl`` — the committed §0.2.5 bound-neutrality
    reference, restricted to the **deterministic certifying** subset: each
    instance is solved twice and included only if both runs reach OPTIMAL with a
    bit-identical node_count and objective. Time-limited / non-deterministic rows
    are excluded, so the reference is reproducible by construction.

Per-instance time limits use the perf panel's own budgets where defined, else
``--time-limit``. The baseline is the frozen reference, so re-generate it
deliberately (not in CI).

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
    SolveStatus,
    incorrect_count,
    root_gap_populated_fraction,
)
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig  # noqa: E402
from perf.panel import PANEL  # noqa: E402

_GLOBAL50 = _BENCH_ROOT / "config" / "baron_global50.txt"
_CERT_BASELINE = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline.jsonl"
_CERT_OPTIMA = _REPO_ROOT / "docs" / "dev" / "data" / "cert-optima.json"

# Objective reproducibility tolerance (absolute + relative). A certified optimum
# reproduces only to ~1e-10 across independent runs, so this is the tolerance the
# §0.2.5 neutrality check compares objectives at (node_count stays bit-exact).
_OBJ_TOL = 1e-8
_OBJ_RTOL = 1e-9


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
    # Full-panel results feed the cert0 coverage gate (root_gap on all of
    # global50). The committed neutrality reference (cert-baseline.jsonl) is the
    # stricter *deterministic certifying* subset built below.
    results = BenchmarkResults(suite="cert0", timestamp=datetime.now().isoformat())
    certifying: list = []
    dropped: list[tuple[str, str]] = []

    def _solve(name: str):
        cfg = BenchmarkConfig(
            suite_name="cert0", time_limit=int(budgets[name]), num_runs=1, solvers=[solver]
        )
        return BenchmarkRunner(cfg)._run_discopt(solver, name, 0)

    for i, name in enumerate(order, 1):
        cfg = BenchmarkConfig(suite_name="cert0", time_limit=1, num_runs=1, solvers=[solver])
        if BenchmarkRunner(cfg)._find_nl_file(name) is None:
            print(f"  [{i}/{len(order)}] SKIP {name} (not vendored)", flush=True)
            continue
        # Solve twice: an instance qualifies for the neutrality reference only if
        # it certifies to OPTIMAL both times with a bit-identical node_count and
        # objective. Time-limited / non-deterministic rows (which made the old
        # baseline unusable — nvs05 feasible, nvs22 stale) are excluded.
        r1 = _solve(name)
        r2 = _solve(name)
        results.add_result(r1)  # full panel = first run
        # node_count must be bit-identical (it is stable at fixed conditions);
        # the objective is compared to a tolerance because a certified optimum
        # jitters at the ~1e-10 level across independent runs (JAX/BLAS
        # non-determinism) — bit-exact objective equality is not a meaningful
        # invariant. This is also the tolerance the §0.2.5 neutrality check uses.
        obj_ok = (
            r1.objective is not None
            and r2.objective is not None
            and abs(r1.objective - r2.objective) <= _OBJ_TOL + _OBJ_RTOL * abs(r1.objective)
        )
        det = (
            r1.status == SolveStatus.OPTIMAL
            and r2.status == SolveStatus.OPTIMAL
            and r1.node_count == r2.node_count
            and obj_ok
        )
        if det:
            certifying.append(r1)
            tag = "CERTIFY"
        else:
            if r1.status != SolveStatus.OPTIMAL or r2.status != SolveStatus.OPTIMAL:
                reason = f"not-optimal({r1.status.value}/{r2.status.value})"
            elif r1.node_count != r2.node_count:
                reason = f"node_count {r1.node_count}!={r2.node_count}"
            else:
                reason = f"obj drift {abs(r1.objective - r2.objective):.2e}"
            dropped.append((name, reason))
            tag = f"drop:{reason}"
        rg = "None" if r1.root_gap is None else f"{r1.root_gap:.3g}"
        print(
            f"  [{i}/{len(order)}] {name:20s} {r1.status.value:10s} "
            f"nodes={r1.node_count} root_gap={rg} {tag}",
            flush=True,
        )

    # reports/cert0_<ts>.json for the --gate cert0 consumer (full panel).
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    reports_dir = _REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    results_path = reports_dir / f"cert0_{ts}.json"
    results.save(results_path)

    # docs/dev/data/cert-baseline.jsonl — the §0.2.5 neutrality reference: the
    # deterministic-certifying subset only.
    cert_rows = sorted(certifying, key=lambda r: r.instance)
    os.makedirs(_CERT_BASELINE.parent, exist_ok=True)
    with open(_CERT_BASELINE, "w") as fh:
        for r in cert_rows:
            fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    # Summary + self-check.
    full_rows = results.get_results("discopt")
    coverage = root_gap_populated_fraction(full_rows)
    optima = json.loads(_CERT_OPTIMA.read_text()) if _CERT_OPTIMA.exists() else {}
    incorrect = incorrect_count(full_rows, optima) if optima else None
    print("\n─── summary ───")
    print(f"  full panel rows: {len(full_rows)}  (cert0 gate)")
    print(f"  root_gap coverage: {coverage:.3f} (gate: >= 0.90)")
    print(f"  incorrect_count (vs {len(optima)} oracles): {incorrect}")
    print(f"  deterministic-certifying subset: {len(cert_rows)}  (neutrality reference)")
    print(f"  dropped ({len(dropped)}): " + ", ".join(f"{n}[{r}]" for n, r in dropped))
    print(f"  results: {results_path}")
    print(f"  baseline: {_CERT_BASELINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
