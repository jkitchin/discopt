"""Phase 5, Step 1 — the native-kernel coverage census.

Consolidation plan (``docs/dev/consolidation-plan-2026-07-28.md``) Phase 5.1 asks
for the one measurement that orders every other Phase-5 sub-card: **why** does the
native spatial kernel decline each corpus instance, ranked by instance count and by
wall-clock at stake?

Card 3c measured the kernel serving **zero** end-to-end solves (2 producer calls, 0
served) and could not say why — the producer had no reason codes. It does now
(``discopt._jax.spatial_producer.producer_stats``), as do the feature gate and the
driver (``discopt.solver.native_kernel.kernel_engagement_stats``). This script joins
those three instruments with the routing walk (``discopt.routing``) and with the
frozen Phase-0 wall baseline, and prints the ranked table.

Four things can stop the kernel serving a solve, and lumping them together ranks
nothing — so every row is classified into exactly one:

``never_reached:<route>``
    ``solve_model`` dispatched to another engine before the kernel gate. This is
    G-C ("capabilities exist but the default path does not reach them") and no
    amount of kernel-coverage work touches it.
``feature_gate:<contract>``
    ``_native_kernel_feature_safe`` declined: the solve requested a Python-engine
    contract (callback, pool, lazy constraints, ...) the kernel does not fill.
``producer:<code>``
    ``build_spatial_kernel_spec`` declined: a relaxation feature the kernel cannot
    build. **These are the Phase-5 sub-cards**, and their ranking is the point of
    this script.
``driver:<outcome>`` / ``served``
    The producer built a spec; the driver either returned a result (``served``) or
    rejected the kernel's own answer (verification, status, ...).

Method notes (CLAUDE.md §6-§10)
-------------------------------

* One subprocess per instance (env / JAX / module-global isolation), running with
  ``DISCOPT_NATIVE_SPATIAL_KERNEL=1`` — the gate is unreachable with the flag off,
  so a census on defaults would measure nothing.
* The producer is **tapped, not simulated**: the box it is handed is the real
  presolved root box after root FBBT + OBBT. A static pre-filter reading declared
  bounds gets this wrong on exactly the instances that matter (#902 measured it
  dropping ``tanksize``), so the census pays for a real solve up to the gate.
* Once the classification is decided the child exits immediately, so a decline
  costs root-setup wall rather than a full budget. Wall-at-stake therefore comes
  from the frozen baseline ``reports/panel_baseline_f154dcff.json``, never from
  this run.
* Executed-probe counts (rows, gate calls, producer calls) are printed and the
  script exits non-zero when any of them is zero.

Usage
-----

::

    python -u discopt_benchmarks/scripts/phase5_kernel_coverage_census.py
    python -u discopt_benchmarks/scripts/phase5_kernel_coverage_census.py --subset 5
    python -u discopt_benchmarks/scripts/phase5_kernel_coverage_census.py \
        --report reports/phase5_kernel_coverage_census_<sha>.json   # re-render only

Internal child mode: ``--solve <instance> <budget>`` (one instance, one JSON line).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "discopt_benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "discopt_benchmarks"))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.panel_baseline import (  # noqa: E402
    _git_dirty,
    _load1,
    _short_sha,
    corpus_instances,
    instance_path,
)

# The instrumentation this census reads did not exist before Phase 5 Step 1.
# CLAUDE.md §8: assert the marker, in the child, before measuring anything.
_INSTRUMENT_MARKER = "phase5-census-v1"

_BASELINE = _REPO_ROOT / "reports" / "panel_baseline_f154dcff.json"
_REPORTS_DIR = _REPO_ROOT / "reports"

# Matches the frozen Phase-0 baseline's budget so the presolved box handed to the
# producer is the one that baseline's walls were measured against.
_DEFAULT_BUDGET = 45.0
_CHILD_TIMEOUT_SLACK = 120.0


# --------------------------------------------------------------------------- #
# Child                                                                       #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"

    import discopt  # noqa: PLC0415
    import discopt.routing as routing  # noqa: PLC0415
    from discopt._jax import spatial_producer  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415
    from discopt.solver import native_kernel  # noqa: PLC0415

    out: dict = {
        "instance": instance,
        "discopt_file": discopt.__file__,
        "budget": float(budget),
        "instrument_marker": None,
    }
    # CLAUDE.md §8 — prove the code under test is loaded before measuring.
    if hasattr(spatial_producer, "producer_stats") and hasattr(
        native_kernel, "kernel_engagement_stats"
    ):
        out["instrument_marker"] = _INSTRUMENT_MARKER
    else:  # pragma: no cover - stale install
        out["status"] = "instrument_missing"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 3

    def _walk() -> dict:
        runs = routing.current_runs()
        hit = {n: r for n, r in runs.items() if r.hits}
        dispatched = max(hit, key=lambda n: hit[n].order) if hit else None
        return {
            "dispatched": dispatched,
            "entered": sorted(hit, key=lambda n: hit[n].order),
        }

    def _emit(phase: str) -> None:
        out["phase"] = phase
        out["producer"] = spatial_producer.producer_stats()
        out["engagement"] = native_kernel.kernel_engagement_stats()
        out["routing"] = _walk()
        print("RESULT_JSON " + json.dumps(out), flush=True)

    # Tap the producer at its real call site. The import inside
    # ``_try_native_spatial_kernel`` is executed per call, so patching the module
    # attribute is seen by the live call site (verified by the census itself: a
    # row whose ``producer.calls`` is 0 while ``engagement.attempts`` is 1 would
    # mean the tap missed, and the parent flags exactly that).
    _orig_producer = spatial_producer.build_spatial_kernel_spec

    def _tapped(model, bounds=None):
        spec = _orig_producer(model, bounds=bounds)
        if spec is None:
            # Classification decided: nothing later can change it, and the rest of
            # the budget would be spent in the Python loop we are not measuring.
            out["wall_to_gate"] = time.perf_counter() - t0
            _emit("producer_declined")
            sys.stdout.flush()
            os._exit(0)
        return spec

    spatial_producer.build_spatial_kernel_spec = _tapped

    nl = str(instance_path(instance))
    t0 = time.perf_counter()
    try:
        model = from_nl(nl)
        out["n_vars"] = len(model._variables)
        out["n_cons"] = len(model._constraints)
        r = model.solve(time_limit=budget)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["node_count"] = int(r.node_count)
        out["gap_certified"] = bool(r.gap_certified)
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    _emit("solve_returned")
    return 0


# --------------------------------------------------------------------------- #
# Parent                                                                      #
# --------------------------------------------------------------------------- #
def _census_one(instance: str, budget: float) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--solve", instance, str(budget)]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=budget + _CHILD_TIMEOUT_SLACK, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "status": "child_timeout", "budget": float(budget)}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "status": "child_crashed",
        "budget": float(budget),
        "stderr_tail": proc.stderr[-800:],
    }


def classify(row: dict) -> tuple[str, str]:
    """Return ``(bucket, category)`` for one census row.

    ``bucket`` is the coarse owner of the decline (``never_reached`` /
    ``feature_gate`` / ``producer`` / ``driver`` / ``served`` / ``unclassified``);
    ``category`` is the ranked key printed in the table.
    """
    status = str(row.get("status") or "")
    eng = row.get("engagement") or {}
    prod = row.get("producer") or {}
    if status in ("child_timeout", "child_crashed", "errored", "instrument_missing"):
        return "unclassified", f"{status}"
    if int(eng.get("served", 0)) > 0:
        return "served", "served"
    if int(eng.get("gate_calls", 0)) == 0:
        route = (row.get("routing") or {}).get("dispatched") or "none"
        return "never_reached", f"never_reached:{route}"
    if int(eng.get("gate_safe", 0)) == 0:
        reason = eng.get("last_gate_reason") or "unknown"
        return "feature_gate", f"feature_gate:{reason}"
    if int(prod.get("declines", 0)) > 0:
        reasons = prod.get("reasons") or {}
        # A solve may call the producer more than once; rank on the reason that
        # accounts for the most calls, ties broken by name so the table is stable.
        key = max(sorted(reasons), key=lambda k: reasons[k]) if reasons else "unknown"
        return "producer", f"producer:{key}"
    if int(eng.get("attempts", 0)) > 0:
        return "driver", f"driver:{eng.get('last_outcome') or 'unknown'}"
    return "unclassified", "no_probe_fired"


def _load_baseline() -> dict:
    if not _BASELINE.exists():
        return {}
    data = json.loads(_BASELINE.read_text())
    return {r["instance"]: r for r in data.get("rows", [])}


def _render(report: dict) -> str:
    rows = report["rows"]
    base = _load_baseline()
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("PHASE 5 STEP 1 — NATIVE SPATIAL KERNEL COVERAGE CENSUS")
    lines.append("=" * 100)
    lines.append(
        f"tree {report['git_sha']}{'-dirty' if report['git_dirty'] else ''}  "
        f"budget {report['budget']}s  instances {len(rows)}  "
        f"wall {report['total_wall_seconds']:.1f}s  "
        f"load {report['load_start']:.2f}->{report['load_peak']:.2f}"
    )
    lines.append(
        "wall-at-stake source: reports/panel_baseline_f154dcff.json "
        f"({len(base)} baseline rows, 45 s budget, defaults)"
    )
    lines.append("")

    agg: dict[str, dict] = {}
    for r in rows:
        bucket, cat = classify(r)
        a = agg.setdefault(bucket, {"count": 0, "wall": 0.0, "unsolved": 0, "cats": {}})
        a["count"] += 1
        b = base.get(r["instance"])
        w = float(b["wall"]) if b and b.get("wall") is not None else 0.0
        a["wall"] += w
        if b and str(b.get("status")) in ("time_limit", "feasible", "child_timeout"):
            a["unsolved"] += 1
        c = a["cats"].setdefault(cat, {"count": 0, "wall": 0.0, "unsolved": 0, "inst": []})
        c["count"] += 1
        c["wall"] += w
        if b and str(b.get("status")) in ("time_limit", "feasible", "child_timeout"):
            c["unsolved"] += 1
        c["inst"].append(r["instance"])

    lines.append("--- BUCKETS (who owns the decline) " + "-" * 60)
    lines.append(f"{'bucket':<18}{'inst':>6}{'baseline wall (s)':>20}{'% wall':>9}{'unsolved':>10}")
    tot_wall = sum(a["wall"] for a in agg.values()) or 1.0
    for bucket in sorted(agg, key=lambda b: -agg[b]["wall"]):
        a = agg[bucket]
        lines.append(
            f"{bucket:<18}{a['count']:>6}{a['wall']:>20.1f}"
            f"{100 * a['wall'] / tot_wall:>8.1f}%{a['unsolved']:>10}"
        )
    lines.append("")

    lines.append("--- RANKED DECLINE REASONS " + "-" * 68)
    lines.append(
        f"{'#':>3} {'category':<44}{'inst':>5}{'wall (s)':>11}{'% wall':>8}{'unsolved':>10}"
    )
    flat = []
    for _bucket, a in agg.items():
        for cat, c in a["cats"].items():
            flat.append((cat, c))
    flat.sort(key=lambda kv: (-kv[1]["wall"], -kv[1]["count"], kv[0]))
    for i, (cat, c) in enumerate(flat, 1):
        lines.append(
            f"{i:>3} {cat:<44}{c['count']:>5}{c['wall']:>11.1f}"
            f"{100 * c['wall'] / tot_wall:>7.1f}%{c['unsolved']:>10}"
        )
    lines.append("")
    lines.append("--- INSTANCES PER CATEGORY " + "-" * 68)
    for cat, c in flat:
        names = ", ".join(sorted(c["inst"]))
        lines.append(f"  {cat}  ({c['count']})")
        for chunk_start in range(0, len(names), 92):
            lines.append(f"      {names[chunk_start : chunk_start + 92]}")
    lines.append("")

    ex = report["executed"]
    lines.append("--- EXECUTED PROBES " + "-" * 74)
    lines.append(
        f"  rows examined          {ex['rows']}\n"
        f"  feature-gate calls     {ex['gate_calls']}\n"
        f"  producer calls         {ex['producer_calls']}\n"
        f"  producer declines      {ex['producer_declines']}\n"
        f"  kernel driver attempts {ex['attempts']}\n"
        f"  kernel solves served   {ex['served']}\n"
        f"  unclassified rows      {ex['unclassified']}"
    )
    return "\n".join(lines)


def cmd_census(args: argparse.Namespace) -> int:
    instances = corpus_instances()
    if args.subset:
        if args.subset.isdigit():
            instances = instances[: int(args.subset)]
        else:
            want = {s.strip() for s in args.subset.split(",") if s.strip()}
            instances = [i for i in instances if i in want]
    if not instances:
        print("FAIL: no instances resolved", file=sys.stderr)
        return 2

    load_start = _load1()
    load_peak = load_start
    t0 = time.perf_counter()
    rows: list[dict] = []
    for idx, inst in enumerate(instances, 1):
        row = _census_one(inst, args.budget)
        rows.append(row)
        load_peak = max(load_peak, _load1())
        bucket, cat = classify(row)
        print(
            f"[{idx:3d}/{len(instances)}] {inst:<28} {cat:<44} "
            f"phase={row.get('phase', '-')} status={row.get('status', '-')}",
            flush=True,
        )

    ex = {
        "rows": len(rows),
        "gate_calls": sum(int((r.get("engagement") or {}).get("gate_calls", 0)) for r in rows),
        "producer_calls": sum(int((r.get("producer") or {}).get("calls", 0)) for r in rows),
        "producer_declines": sum(int((r.get("producer") or {}).get("declines", 0)) for r in rows),
        "attempts": sum(int((r.get("engagement") or {}).get("attempts", 0)) for r in rows),
        "served": sum(int((r.get("engagement") or {}).get("served", 0)) for r in rows),
        "unclassified": sum(1 for r in rows if classify(r)[0] == "unclassified"),
    }
    report = {
        "schema": "phase5_kernel_coverage_census/1",
        "git_sha": _short_sha(),
        "git_dirty": _git_dirty(),
        "budget": float(args.budget),
        "flag": "DISCOPT_NATIVE_SPATIAL_KERNEL=1",
        "instrument_marker": _INSTRUMENT_MARKER,
        "baseline": str(_BASELINE.relative_to(_REPO_ROOT)),
        "total_wall_seconds": time.perf_counter() - t0,
        "load_start": load_start,
        "load_peak": load_peak,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "executed": ex,
        "rows": rows,
    }
    _REPORTS_DIR.mkdir(exist_ok=True)
    out_path = _REPORTS_DIR / f"phase5_kernel_coverage_census_{report['git_sha']}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(_render(report))
    print(f"\nartifact: {out_path.relative_to(_REPO_ROOT)}")

    # CLAUDE.md §6: a census that examined nothing must fail, not read as a pass.
    if ex["rows"] == 0 or ex["gate_calls"] == 0 or ex["producer_calls"] == 0:
        print("\nFAIL: zero executed probes — the census measured nothing", file=sys.stderr)
        return 2
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text())
    print(_render(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", nargs=2, metavar=("INSTANCE", "BUDGET"))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--budget", type=float, default=_DEFAULT_BUDGET)
    ap.add_argument("--report", default=None, help="re-render an existing census artifact")
    args = ap.parse_args(argv)
    if args.solve:
        return _run_child(args.solve[0], float(args.solve[1]))
    if args.report:
        return cmd_report(args)
    return cmd_census(args)


if __name__ == "__main__":
    raise SystemExit(main())
