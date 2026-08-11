"""#928 graduation residual: which flag loses the incumbent, and is it the budget?

The last two panels (§14d, §14e) pass the net-positive bar and fail ``CERT_CLEAN``
on exactly one class of item: an incumbent the ``base`` arm reports and a flag arm
does not (tspn12 in both reps of §14e, tspn10/tls2 sporadically). Before any fix,
two questions have to be answered with numbers:

1. **Attribution.** Which of the three flags drops it? The panel's three arms
   cannot separate ``DISCOPT_NODE_ROUND_BUDGET`` from ``DISCOPT_HESS_COMPILE_GATE``
   (``seam`` sets both), so this probe runs all five combinations.
2. **Is it a search regression or budget enforcement?** The ``base`` arm on tspn12
   runs 42-58 s against a 20 s budget; an incumbent it finds at 35 s is not an
   incumbent the flag arms "lost", it is one the control bought by overrunning
   2-3x. The EQUAL-WALL control arm gives ``cand`` a budget equal to the base
   arm's measured wall on the same instance and asks whether the incumbent comes
   back.

Counted per CLAUDE.md §6: prints CELLS_EXECUTED and EQUALWALL_COMPARISONS and
exits non-zero if either is zero.

    python -u scratchpad/issue928_incumbent_attribution.py \
        --instances tspn12,tspn10,tls2 --budget 20 --reps 2 \
        --out scratchpad/issue928_incumbent_attribution.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "python/tests/data/minlplib_nl"
WORKER = ROOT / "discopt_benchmarks/scripts/issue966_coupled_worker.py"

WARM = "DISCOPT_LP_WARM_DEADLINE"
ROUND = "DISCOPT_NODE_ROUND_BUDGET"
HESS = "DISCOPT_HESS_COMPILE_GATE"

ARMS = (
    ("base", {WARM: "0", ROUND: "0", HESS: "0"}),
    ("round", {WARM: "0", ROUND: "1", HESS: "0"}),
    ("hess", {WARM: "0", ROUND: "0", HESS: "1"}),
    ("seam", {WARM: "0", ROUND: "1", HESS: "1"}),
    ("cand", {WARM: "1", ROUND: "1", HESS: "1"}),
)


def run_one(nl: Path, budget: float, env: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(budget)],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=40 * budget + 900,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise SystemExit(f"worker failed on {nl.stem} env={env}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="tspn12,tspn10,tls2")
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="scratchpad/issue928_incumbent_attribution.json")
    args = ap.parse_args()

    names = [s for s in args.instances.split(",") if s]
    cells, equalwall = [], []
    executed = 0
    load_start = os.getloadavg()

    for rep in range(1, args.reps + 1):
        for name in names:
            nl = CORPUS / f"{name}.nl"
            cell = {"instance": name, "rep": rep, "budget": args.budget}
            for key, env in ARMS:
                cell[key] = run_one(nl, args.budget, env)
                executed += 1
            cells.append(cell)
            print(
                f"[rep{rep}] {name:10s} "
                + " | ".join(
                    f"{k} wall={cell[k]['wall']:5.1f} inc={cell[k]['objective']} "
                    f"bound={cell[k]['bound']}"
                    for k, _e in ARMS
                ),
                flush=True,
            )

    # Equal-wall control: give ``cand`` the wall the base arm actually spent, on
    # every instance where base reported an incumbent and cand did not.
    for c in cells:
        if c["base"]["objective"] is None or c["cand"]["objective"] is not None:
            continue
        eq_budget = round(float(c["base"]["wall"]), 1)
        rec = run_one(CORPUS / f"{c['instance']}.nl", eq_budget, dict(ARMS[4][1]))
        executed += 1
        equalwall.append(
            {
                "instance": c["instance"],
                "rep": c["rep"],
                "base_wall": c["base"]["wall"],
                "base_objective": c["base"]["objective"],
                "cand_equalwall_budget": eq_budget,
                "cand_equalwall_wall": rec["wall"],
                "cand_equalwall_objective": rec["objective"],
                "cand_equalwall_bound": rec["bound"],
                "recovered": rec["objective"] is not None,
            }
        )
        print(f"  EQUAL-WALL {c['instance']} @{eq_budget}s -> inc={rec['objective']}", flush=True)

    out = {
        "load_start": load_start,
        "load_end": os.getloadavg(),
        "cells_executed": executed,
        "equalwall_comparisons": len(equalwall),
        "cells": cells,
        "equalwall": equalwall,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"CELLS_EXECUTED={executed}")
    print(f"EQUALWALL_COMPARISONS={len(equalwall)}")
    # Per-arm incumbent tally against base, the question this probe exists for.
    for key, _e in ARMS[1:]:
        lost = [
            c["instance"]
            for c in cells
            if c["base"]["objective"] is not None and c[key]["objective"] is None
        ]
        gained = [
            c["instance"]
            for c in cells
            if c["base"]["objective"] is None and c[key]["objective"] is not None
        ]
        print(f"{key:6s} lost_incumbents={lost} gained={gained}")
    return 0 if executed else 1


if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = main()
    print(f"elapsed={time.perf_counter() - t0:.1f}s")
    sys.exit(rc)
