"""Performance regression gate (perf plan Stage 0).

Runs the fixed panel, then enforces two kinds of gate against a committed
baseline (``docs/dev/data/perf-baseline.jsonl``):

* **Correctness (hard, non-negotiable):** every panel instance must return a
  *sound* result vs its oracle optimum — no false-feasible (incumbent beats the
  optimum), no false-optimal (a gap=0 "optimal" off the optimum), no
  false-infeasible / -unbounded, and a dual bound that never crosses the optimum
  or the incumbent. This reuses the soundness invariants the adversarial suite
  pins, as a second tripwire on the perf panel.

* **Perf regression (hard on *deterministic* metrics):** ``node_count`` (for
  instances that certify in both runs) and ``compiles_per_node`` must not
  regress > 15 %. Wall time and time-to-first-incumbent are reported as warnings
  only — they vary run-to-run and under the compile-logging instrumentation, so
  they are advisory, not blocking.

Usage::

    python -m discopt_benchmarks.perf.gate                 # check vs baseline (CI)
    python -m discopt_benchmarks.perf.gate --update-baseline   # regenerate baseline

Exit code is non-zero iff a hard gate fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from discopt_benchmarks.perf.measure import PerfRecord, measure_solve
from discopt_benchmarks.perf.panel import PANEL, PanelInstance

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "docs",
    "dev",
    "data",
    "perf-baseline.jsonl",
)

REGRESSION_FRAC = 0.15  # > 15 % worse on a deterministic metric is a hard fail
WALL_WARN_FRAC = 0.25  # advisory only
_NODE_FLOOR = 50  # don't gate node_count on tiny trees (noise)
_COMPILE_FLOOR = 5  # don't gate compiles_per_node when compiles are negligible


def soundness_violation(rec: PerfRecord, inst: PanelInstance) -> str | None:
    """Return a description if the record is unsound vs the oracle, else None.

    Sense-aware; tolerances mirror ``test_adversarial_recent_fixes`` (rel 0.5 %
    band, abs floor 1e-4; bound side gets a small numerical slack).
    """
    opt, sense = inst.oracle, inst.sense
    tol = max(5e-3 * abs(opt), 1e-4)
    bslack = tol + 1e-2
    obj, bnd, status = rec.objective, rec.bound, rec.status

    if status == "infeasible":
        return "FALSE-INFEASIBLE (instance is feasible)"
    if status == "unbounded":
        return "FALSE-UNBOUNDED (instance is bounded)"
    if obj is not None:
        if sense == "min" and obj < opt - tol:
            return f"FALSE-FEASIBLE incumbent {obj:.6g} < opt {opt:.6g}"
        if sense == "max" and obj > opt + tol:
            return f"FALSE-FEASIBLE incumbent {obj:.6g} > opt {opt:.6g}"
        if status == "optimal" and abs(obj - opt) > tol:
            return f"FALSE-OPTIMAL certified {obj:.6g} != opt {opt:.6g}"
    if bnd is not None:
        if sense == "min" and bnd > opt + bslack:
            return f"INVALID BOUND {bnd:.6g} > opt {opt:.6g}"
        if sense == "max" and bnd < opt - bslack:
            return f"INVALID BOUND {bnd:.6g} < opt {opt:.6g}"
        if obj is not None:
            if sense == "min" and bnd > obj + bslack:
                return f"UNSOUND CERT bound {bnd:.6g} > incumbent {obj:.6g}"
            if sense == "max" and bnd < obj - bslack:
                return f"UNSOUND CERT bound {bnd:.6g} < incumbent {obj:.6g}"
    return None


def run_panel() -> list[PerfRecord]:
    records = []
    for inst in PANEL:
        if not os.path.exists(inst.path):
            print(f"  SKIP {inst.name} (not vendored)", flush=True)
            continue
        rec = measure_solve(inst.path, inst.time_limit)
        records.append(rec)
        print(
            f"  {inst.name:10s} {rec.status:10s} obj={rec.objective} "
            f"nodes={rec.node_count} wall={rec.wall_time:.1f}s "
            f"compiles={rec.xla_compile_count} cpn={rec.compiles_per_node}",
            flush=True,
        )
    return records


def write_baseline(records: list[PerfRecord], path: str = BASELINE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json(), sort_keys=True) + "\n")


def load_baseline(path: str = BASELINE_PATH) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["instance"]] = d
    return out


def _certifies(status: str) -> bool:
    return status in ("optimal", "infeasible")


def _regressed(cur: float, base: float) -> bool:
    return base > 0 and cur > base * (1.0 + REGRESSION_FRAC)


def check_gate(records: list[PerfRecord], baseline: dict[str, dict]):
    """Return (correctness_failures, perf_regressions, warnings) as lists of str."""
    by_name = {inst.name: inst for inst in PANEL}
    correctness, regressions, warnings = [], [], []

    for rec in records:
        inst = by_name[rec.instance]
        # --- correctness (hard) ---
        v = soundness_violation(rec, inst)
        if v:
            correctness.append(f"{rec.instance}: {v}")

        base = baseline.get(rec.instance)
        if not base:
            warnings.append(f"{rec.instance}: no baseline entry (new instance?)")
            continue

        # --- node_count (hard, only when both runs certify and the tree is non-trivial) ---
        if (
            _certifies(rec.status)
            and _certifies(base["status"])
            and base["node_count"] >= _NODE_FLOOR
            and _regressed(rec.node_count, base["node_count"])
        ):
            regressions.append(
                f"{rec.instance}: node_count {base['node_count']} -> {rec.node_count} "
                f"(+{100 * (rec.node_count / base['node_count'] - 1):.0f}%)"
            )

        # --- compiles_per_node (hard, the cost-center metric) ---
        base_cpn = (base["xla_compile_count"] / base["node_count"]) if base["node_count"] else None
        cur_cpn = rec.compiles_per_node
        if (
            base.get("xla_compile_count", 0) >= _COMPILE_FLOOR
            and base_cpn
            and cur_cpn is not None
            and _regressed(cur_cpn, base_cpn)
        ):
            regressions.append(
                f"{rec.instance}: compiles/node {base_cpn:.3f} -> {cur_cpn:.3f} "
                f"(+{100 * (cur_cpn / base_cpn - 1):.0f}%) "
                f"[compiles {base['xla_compile_count']} -> {rec.xla_compile_count}]"
            )

        # --- wall (advisory) ---
        if _regressed(rec.wall_time, base["wall_time"]) and rec.wall_time - base["wall_time"] > 2.0:
            warnings.append(
                f"{rec.instance}: wall {base['wall_time']:.1f}s -> {rec.wall_time:.1f}s "
                f"(+{100 * (rec.wall_time / base['wall_time'] - 1):.0f}%, advisory)"
            )

    return correctness, regressions, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="discopt performance regression gate")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="run the panel and overwrite the committed baseline (no gating)",
    )
    args = ap.parse_args()

    print("Running perf panel...", flush=True)
    records = run_panel()

    if args.update_baseline:
        write_baseline(records)
        print(f"\nWrote baseline ({len(records)} instances) -> {BASELINE_PATH}")
        return 0

    baseline = load_baseline()
    if not baseline:
        print("\nNo baseline found. Run with --update-baseline first.", file=sys.stderr)
        return 2

    correctness, regressions, warnings = check_gate(records, baseline)

    print("\n" + "=" * 70)
    for w in warnings:
        print(f"  [warn] {w}")
    for r in regressions:
        print(f"  [REGRESSION] {r}")
    for c in correctness:
        print(f"  [INCORRECT] {c}")

    if correctness:
        print(f"\nPERF GATE FAILED — {len(correctness)} correctness violation(s).")
        return 1
    if regressions:
        print(f"\nPERF GATE FAILED — {len(regressions)} performance regression(s).")
        return 1
    print(f"\nPERF GATE PASSED — {len(records)} instances, 0 incorrect, 0 regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
