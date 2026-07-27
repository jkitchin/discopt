#!/usr/bin/env python3
"""Smoke head-to-head: discopt (warm daemon) vs SCIP (.nl) — timing + correctness.

A quick 2-way sanity run on a small curated instance set. Reuses the tested lanes
(``run_discopt_daemon`` for warm, import-cost-amortized discopt timing;
``run_external`` for SCIP via the .nl) and the shared MINLPLib ``primalbound``
oracle + ``classify`` verdict (``VIOLATION`` = false global / better-than-optimum).

Usage:
    python -m discopt_benchmarks.scripts.smoke_discopt_scip [--time-limit 20] \
        [--instances a,b,c]
"""

from __future__ import annotations

import argparse
import time

from discopt_benchmarks.scripts.global_opt_baron_vs_discopt import (
    OK,
    VIOLATION,
    classify,
    fmt,
    load_known_optima,
    nl_is_maximize,
    run_discopt_daemon,
)
from discopt_benchmarks.scripts.global_opt_nl_solvers import (
    load_solver_commands,
    make_runner,
    run_external,
)

from benchmarks.runner import SolverConfig

# Small spread across MINLP / MIQP / QP / global-opt classes, all with oracles.
SMOKE = [
    "alan", "ex1221", "ex1223a", "ex1225", "gbd", "gkocis",
    "nvs03", "nvs06", "st_e13", "st_miqp1", "clay0303hfsg", "fac2",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--instances", type=str, default="")
    args = ap.parse_args()
    tl = args.time_limit
    insts = [s.strip() for s in args.instances.split(",") if s.strip()] or SMOKE

    known = load_known_optima()
    runner = make_runner(tl)
    scip_cmd = load_solver_commands().get("scip", "scip")
    scip = SolverConfig(name="scip", command=scip_cmd, solver_type="external", nl_interface=True)

    print(f"# smoke: discopt (daemon) vs SCIP — {len(insts)} instances, time_limit={int(tl)}s")
    print("# warming discopt daemon (excluded solve) ...", flush=True)
    run_discopt_daemon(insts[0], min(tl, 10.0))

    print(f"\n{'instance':14} {'known':>12} | {'discopt':>10} {'v':>4} {'t(s)':>6} | "
          f"{'scip':>12} {'v':>4} {'t(s)':>6}")
    rows = []
    for name in insts:
        kn = known.get(name)
        mx = nl_is_maximize(name)
        d = run_discopt_daemon(name, tl)
        s = run_external(runner, scip, name, tl)
        dv = classify(d.status, d.objective, kn, mx)
        sv = classify(s.status, s.objective, kn, mx)
        rows.append((name, kn, d, dv, s, sv))
        print(f"{name:14} {fmt(kn):>12} | {fmt(d.objective,9):>10} {dv:>4} "
              f"{(d.wall_time or 0):6.2f} | {fmt(s.objective,9):>12} {sv:>4} {(s.wall_time or 0):6.2f}",
              flush=True)

    n_or = sum(r[1] is not None for r in rows)
    dok = sum(r[3] == OK for r in rows); sok = sum(r[5] == OK for r in rows)
    dvio = sum(r[3] == VIOLATION for r in rows); svio = sum(r[5] == VIOLATION for r in rows)
    dwall = sum((r[2].wall_time or 0) for r in rows); swall = sum((r[4].wall_time or 0) for r in rows)
    print("\n================ SMOKE SUMMARY ================")
    print(f"instances with oracle: {n_or}/{len(rows)}")
    print(f"discopt  matched-optimum={dok}/{n_or}  VIOLATIONS={dvio}  total={dwall:.2f}s")
    print(f"scip     matched-optimum={sok}/{n_or}  VIOLATIONS={svio}  total={swall:.2f}s")
    tot_vio = dvio + svio
    print(f"CORRECTNESS: {'PASS (0 violations)' if tot_vio == 0 else f'FAIL ({tot_vio})'}")
    return 1 if tot_vio else 0


if __name__ == "__main__":
    raise SystemExit(main())
