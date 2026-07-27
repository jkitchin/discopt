#!/usr/bin/env python3
"""3-way global50 head-to-head: discopt (warm daemon) vs BARON (GAMS) vs SCIP (.nl).

Runs the curated 50-instance global-opt set (``config/baron_global50.txt``) through
three solvers under one per-problem budget and reports **solve time** (performance)
and **correctness** (verdict vs the MINLPLib known optimum) for each.

Lanes are the existing, tested ones — nothing reinvented:
- **discopt** via the warm daemon (``run_discopt_daemon``), so the JAX/Python
  import cost is paid once (an excluded warm-up solve) instead of per instance.
- **BARON** via GAMS 53 (``run_baron``: fetch ``.gms``, ``gams ... minlp=baron``).
- **SCIP** via the ``.nl`` (``run_external`` + the runner command/parsers).

Correctness oracle + verdict classifier are shared with both head-to-head harnesses
(``classify`` vs MINLPLib ``primalbound``; ``VIOLATION`` = false global / incumbent
better than the proven optimum).

Usage:
    python -m discopt_benchmarks.scripts.global50_3way_daemon [--time-limit 60] \
        [--instances a,b,c] [--out-dir reports]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from discopt_benchmarks.scripts.global_opt_baron_vs_discopt import (
    GAP,
    NA,
    OK,
    REPO,
    VIOLATION,
    classify,
    fmt,
    load_known_optima,
    nl_is_maximize,
    run_baron,
    run_discopt_daemon,
)
from discopt_benchmarks.scripts.global_opt_nl_solvers import (
    load_solver_commands,
    make_runner,
    run_external,
)

from benchmarks.runner import SolverConfig

GLOBAL50 = REPO / "discopt_benchmarks" / "config" / "baron_global50.txt"
SOLVER_ORDER = ["discopt", "baron", "scip"]


def read_global50() -> list[str]:
    return [
        ln.strip()
        for ln in GLOBAL50.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def geomean(xs: list[float]) -> float:
    xs = [max(x, 1e-9) for x in xs if x is not None]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def write_report(rows: list[dict], tl: float, out_dir: Path, ts: str) -> Path:
    n_oracle = sum(r["known"] is not None for r in rows)
    tallies = {s: {OK: 0, GAP: 0, VIOLATION: 0, NA: 0} for s in SOLVER_ORDER}
    walls: dict[str, list[float]] = {s: [] for s in SOLVER_ORDER}
    for r in rows:
        for s in SOLVER_ORDER:
            run = r["runs"][s]
            tallies[s][run["verdict"]] += 1
            walls[s].append(run.get("wall_time") or 0.0)

    lines = [
        "# 3-way global50 head-to-head — discopt (daemon) vs BARON vs SCIP",
        "",
        f"- Instances: **{len(rows)}** (`config/baron_global50.txt`); "
        f"oracle-known optima: **{n_oracle}/{len(rows)}**",
        f"- Per-problem time limit: **{int(tl)} s**; correctness tol abs=1e-6 rel=1e-4 "
        "vs MINLPLib `primalbound`",
        "- discopt via **warm daemon** (import cost amortized; one excluded warm-up "
        "solve). BARON via GAMS `minlp=baron`. SCIP via `.nl`.",
        f"- Generated: {ts}",
        "",
        "## Correctness (vs known optimum)",
        "",
        "| verdict | meaning |",
        "|---|---|",
        "| `ok` | incumbent matches the known global within tolerance |",
        "| `GAP` | honest feasible incumbent worse than the global (time-budget "
        "convergence gap, not a bug) |",
        "| `VIOLATION` | **red line**: false-certified global, or an incumbent "
        "strictly better than the proven optimum |",
        "| `n/a` | no oracle / no incumbent |",
        "",
        "| solver | ok | GAP | VIOLATION | n/a |",
        "|---|---|---|---|---|",
    ]
    for s in SOLVER_ORDER:
        t = tallies[s]
        lines.append(f"| {s} | {t[OK]}/{n_oracle} | {t[GAP]} | **{t[VIOLATION]}** | {t[NA]} |")

    lines += [
        "",
        "## Performance (solve time)",
        "",
        "| solver | total wall (s) | geomean wall (s) |",
        "|---|---|---|",
    ]
    for s in SOLVER_ORDER:
        lines.append(f"| {s} | {sum(walls[s]):.1f} | {geomean(walls[s]):.3f} |")

    lines += [
        "",
        "## Per-instance (obj / verdict / time)",
        "",
        "| instance | known | "
        + " | ".join(f"{s} obj | v | t(s)" for s in SOLVER_ORDER)
        + " |",
        "|---|---|" + "---|---|---|" * len(SOLVER_ORDER),
    ]
    for r in sorted(rows, key=lambda x: x["instance"]):
        cells = f"| {r['instance']} | {fmt(r['known'])} |"
        for s in SOLVER_ORDER:
            run = r["runs"][s]
            cells += f" {fmt(run['objective'])} | {run['verdict']} | {(run.get('wall_time') or 0.0):.2f} |"
        lines.append(cells)

    md = out_dir / f"global50_3way_{ts}.md"
    md.write_text("\n".join(lines) + "\n")
    (out_dir / f"global50_3way_{ts}.json").write_text(
        json.dumps({"time_limit": tl, "timestamp": ts, "solvers": SOLVER_ORDER, "rows": rows}, indent=2)
    )
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--instances", type=str, default="")
    ap.add_argument("--out-dir", type=str, default=str(REPO / "reports"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tl = args.time_limit
    known = load_known_optima()
    insts = (
        [s.strip() for s in args.instances.split(",") if s.strip()]
        if args.instances
        else read_global50()
    )

    runner = make_runner(tl)
    scip_cmd = load_solver_commands().get("scip", "scip")
    scip_cfg = SolverConfig(name="scip", command=scip_cmd, solver_type="external", nl_interface=True)

    print(f"# 3-way global50: {len(insts)} instances, solvers={SOLVER_ORDER}, "
          f"time_limit={int(tl)}s", flush=True)

    # Warm the daemon once (excluded): the first request pays spawn + JAX import.
    print("# warming discopt daemon (excluded solve) ...", flush=True)
    run_discopt_daemon(insts[0], min(tl, 30.0))

    rows: list[dict] = []
    for i, name in enumerate(insts, 1):
        kn = known.get(name)
        mx = nl_is_maximize(name)
        d = run_discopt_daemon(name, tl)
        b = run_baron(name, tl, maximize=mx)
        s = run_external(runner, scip_cfg, name, tl)
        runs = {
            "discopt": {**vars(d), "verdict": classify(d.status, d.objective, kn, mx)},
            "baron": {**vars(b), "verdict": classify(b.status, b.objective, kn, mx)},
            "scip": {**vars(s), "verdict": classify(s.status, s.objective, kn, mx)},
        }
        rows.append({"instance": name, "known": kn, "maximize": mx, "runs": runs})
        cells = " | ".join(
            f"{k}:{fmt(runs[k]['objective'], 8)} {runs[k]['verdict']:9} {(runs[k].get('wall_time') or 0.0):5.1f}s"
            for k in SOLVER_ORDER
        )
        print(f"[{i:2}/{len(insts)}] {name:22} {cells}", flush=True)

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    md = write_report(rows, tl, out_dir, ts)

    # Console summary
    n_oracle = sum(r["known"] is not None for r in rows)
    print("\n================ 3-WAY SUMMARY ================")
    for s in SOLVER_ORDER:
        ok = sum(r["runs"][s]["verdict"] == OK for r in rows)
        vio = sum(r["runs"][s]["verdict"] == VIOLATION for r in rows)
        tot = sum((r["runs"][s].get("wall_time") or 0.0) for r in rows)
        gm = geomean([r["runs"][s].get("wall_time") or 0.0 for r in rows])
        print(f"{s:8} ok={ok}/{n_oracle}  VIOLATIONS={vio}  total={tot:.1f}s  geomean={gm:.3f}s")
    total_vio = sum(
        r["runs"][s]["verdict"] == VIOLATION for r in rows for s in SOLVER_ORDER
    )
    print(f"\nReport: {md}")
    print(f"CORRECTNESS GATE: {'PASS (0 violations)' if total_vio == 0 else f'FAIL ({total_vio} violations)'}")
    return 1 if total_vio else 0


if __name__ == "__main__":
    raise SystemExit(main())
