#!/usr/bin/env python3
"""Fuzz loop: discopt (warm daemon) vs SCIP (.nl) over the big MINLPLib corpus.

Runs ``--iters`` iterations of ``--batch`` distinct instances each (default
10 x 50 = 500), drawn from the curated runtime lists (short, then medium) of the
full ~1610-instance MINLPLib snapshot in ``~/Dropbox/projects/discopt-minlp-
benchmark``. Each instance is solved by discopt (warm daemon lane) and SCIP
(native .nl) under a fixed time limit, and every run is checked against the
MINLPLib ``primalbound`` oracle.

Reuses the tested lanes from ``global_opt_baron_vs_discopt`` /
``global_opt_nl_solvers`` unchanged, monkeypatching ``NL_DIR`` and the runner's
``data_dir`` to point at the big corpus instead of the 66 vendored instances.

Flags (written to a JSONL + a summary):
  * VIOLATION       - discopt/scip reports an incumbent objective better than the
                      oracle optimum (false global / cardinal soundness bug).
  * UNSOUND_BOUND   - discopt's dual bound crosses the oracle optimum (min: bound
                      > opt; max: bound < opt) => a FALSE dual bound. The nvs16
                      class checks the *bound*, which ``classify`` does not.
  * SLOWDOWN_5X     - discopt wall > 5x scip wall (with a 1 s floor to suppress
                      sub-second noise), scip having actually produced a result.
  * CRASH           - discopt errored on a valid model (#810 class).

Usage:
    python -m discopt_benchmarks.scripts.fuzz_discopt_scip_loop \
        [--iters 10] [--batch 50] [--time-limit 15] [--out <jsonl>]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from discopt_benchmarks.scripts import global_opt_baron_vs_discopt as G
from discopt_benchmarks.scripts.global_opt_baron_vs_discopt import (
    OK,
    VIOLATION,
    classify,
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

BIG = Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark"
BIG_NL = BIG / "minlplib" / "nl"
LISTS = ["problems_short.txt", "problems_medium.txt", "problems_small.txt"]

# Bound-soundness tolerance (looser than the incumbent tol to avoid flagging
# numerical noise as a false bound; a real garbage bound crosses by orders of
# magnitude, e.g. nvs16's -5e11 vs opt 0.70).
_B_ABS = 1e-4
_B_REL = 1e-3
_SLOW_FACTOR = 5.0
_SLOW_FLOOR_S = 1.0  # ignore 5x ratios when discopt already finished under 1 s


def select_instances(n: int, oracle: dict[str, float]) -> list[str]:
    """Deterministic pool: curated lists, in order, filtered to (exists in the
    big corpus) AND (has an oracle optimum), de-duplicated, truncated to n."""
    seen: set[str] = set()
    out: list[str] = []
    for lst in LISTS:
        p = BIG / lst
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            name = line.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            if (BIG_NL / f"{name}.nl").exists() and name in oracle:
                out.append(name)
            if len(out) >= n:
                return out
    return out


def bound_is_unsound(bound: float | None, opt: float | None, is_max: bool) -> bool:
    """A dual bound that crosses the oracle optimum is a false certificate.

    min sense: a valid dual bound is a LOWER bound => must be <= opt.
    max sense: a valid dual bound is an UPPER bound => must be >= opt.
    """
    if bound is None or opt is None or not math.isfinite(bound):
        return False
    tol = _B_ABS + _B_REL * abs(opt)
    if is_max:
        return bound < opt - tol
    return bound > opt + tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--time-limit", type=float, default=15.0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    tl = args.time_limit
    total = args.iters * args.batch

    # Point every tested lane at the big corpus.
    G.NL_DIR = BIG_NL
    oracle = load_known_optima()
    runner = make_runner(tl)
    runner.config.data_dir = str(BIG_NL)  # SCIP _find_nl_file resolution
    scip_cmd = load_solver_commands().get("scip", "scip")
    scip = SolverConfig(name="scip", command=scip_cmd, solver_type="external", nl_interface=True)

    pool = select_instances(total, oracle)
    out_path = Path(args.out) if args.out else Path(
        "/private/tmp/claude-501/-Users-jkitchin-projects-discopt/"
        "f2daf772-3a9a-4191-bb80-684893cb0543/scratchpad/fuzz_results.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")

    print(f"# fuzz: discopt (daemon) vs SCIP | {len(pool)} distinct instances "
          f"({args.iters} iters x {args.batch}), time_limit={int(tl)}s", flush=True)
    if len(pool) < total:
        print(f"# NOTE: pool has {len(pool)} < requested {total}; running {len(pool)}.", flush=True)

    # Warm the daemon (excluded from timing).
    print("# warming discopt daemon (excluded) ...", flush=True)
    try:
        from discopt.daemon import spawn_daemon
        spawn_daemon()
    except Exception as e:  # noqa: BLE001 — daemon optional; fallback lane covers it
        print(f"# (daemon spawn skipped: {e}); using per-instance fallback", flush=True)
    run_discopt_daemon(pool[0], min(tl, 8.0))

    flags = {"VIOLATION": [], "UNSOUND_BOUND": [], "SLOWDOWN_5X": [], "CRASH": []}
    n_done = 0
    dwall_tot = swall_tot = 0.0

    for it in range(args.iters):
        batch = pool[it * args.batch:(it + 1) * args.batch]
        if not batch:
            break
        print(f"\n===== iteration {it + 1}/{args.iters} — {len(batch)} instances =====",
              flush=True)
        for name in batch:
            opt = oracle.get(name)
            is_max = nl_is_maximize(name)
            d = run_discopt_daemon(name, tl)
            s = run_external(runner, scip, name, tl)
            dv = classify(d.status, d.objective, opt, is_max)
            sv = classify(s.status, s.objective, opt, is_max)
            dwall = d.wall_time or 0.0
            swall = s.wall_time or 0.0
            dwall_tot += dwall
            swall_tot += swall
            n_done += 1

            rec_flags = []
            if dv == VIOLATION:
                rec_flags.append("VIOLATION")
                flags["VIOLATION"].append(name)
            if bound_is_unsound(d.lower_bound, opt, is_max):
                rec_flags.append("UNSOUND_BOUND")
                flags["UNSOUND_BOUND"].append(name)
            if d.status in ("ERROR", "NO_NL", "NO_BINARY") or (d.error and d.status not in
                                                               ("optimal", "feasible",
                                                                "infeasible", "TIME_LIMIT")):
                rec_flags.append("CRASH")
                flags["CRASH"].append(name)
            slow_ratio = (dwall / swall) if swall > 1e-9 else None
            if (slow_ratio is not None and slow_ratio > _SLOW_FACTOR
                    and dwall > _SLOW_FLOOR_S and s.status not in ("NO_NL", "NO_BINARY")):
                rec_flags.append("SLOWDOWN_5X")
                flags["SLOWDOWN_5X"].append((name, round(slow_ratio, 1), round(dwall, 2),
                                             round(swall, 2)))

            rec = {
                "iter": it + 1, "name": name, "opt": opt, "is_max": is_max,
                "discopt": {"status": d.status, "obj": d.objective, "bound": d.lower_bound,
                            "gap": d.gap, "wall": round(dwall, 3), "verdict": dv,
                            "error": d.error},
                "scip": {"status": s.status, "obj": s.objective, "bound": s.lower_bound,
                         "wall": round(swall, 3), "verdict": sv},
                "slow_ratio": (round(slow_ratio, 2) if slow_ratio else None),
                "flags": rec_flags,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            marker = (" <<" + ",".join(rec_flags)) if rec_flags else ""
            print(f"  {name:22} opt={G.fmt(opt,10):>11} | d:{d.status[:9]:9} "
                  f"{G.fmt(d.objective,9):>10} {dwall:6.2f}s | s:{s.status[:9]:9} "
                  f"{swall:6.2f}s{marker}", flush=True)

    fh.close()

    print("\n================ FUZZ SUMMARY ================", flush=True)
    print(f"instances run:        {n_done}", flush=True)
    print(f"discopt total wall:   {dwall_tot:.1f}s   scip total wall: {swall_tot:.1f}s", flush=True)
    print(f"CORRECTNESS violations (incumbent past optimum): {len(flags['VIOLATION'])}", flush=True)
    if flags["VIOLATION"]:
        print("  " + ", ".join(flags["VIOLATION"]), flush=True)
    print(f"UNSOUND dual bounds (bound crosses optimum):     {len(flags['UNSOUND_BOUND'])}",
          flush=True)
    if flags["UNSOUND_BOUND"]:
        print("  " + ", ".join(flags["UNSOUND_BOUND"]), flush=True)
    print(f"CRASHES on valid models:                         {len(flags['CRASH'])}", flush=True)
    if flags["CRASH"]:
        print("  " + ", ".join(flags["CRASH"]), flush=True)
    print(f">5x slower than SCIP (discopt wall > 1s):        {len(flags['SLOWDOWN_5X'])}",
          flush=True)
    for name, ratio, dw, sw in sorted(flags["SLOWDOWN_5X"], key=lambda x: -x[1]):
        print(f"  {name:22} {ratio:5.1f}x   discopt {dw:6.2f}s  vs  scip {sw:6.2f}s", flush=True)
    print(f"\n# results JSONL: {out_path}", flush=True)

    hard = len(flags["VIOLATION"]) + len(flags["UNSOUND_BOUND"]) + len(flags["CRASH"])
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
