"""Is the panel's `cert_regression` on tls2 / syn05hfsg a flag effect or a coin flip?

The 20 s graduation panel (``discopt_benchmarks/results/issue928_grad20.json``)
reports two certification regressions, and both look like knife-edge cells rather
than flag effects: tls2 certifies in the ``wr`` arm in rep1 and in the ``base`` arm
in rep3 and in no other cell; syn05hfsg certifies in base/warm/wr in rep2 only.
Same instance, same flags, different rep — which is the signature of a solve that
closes its last node right at the budget.

Asserting that from three reps is not evidence, so this measures the certification
RATE per arm over N reps of the same cell. The discriminator, fixed before the run:

* **flag effect** — an arm certifies in >=4/5 reps while another certifies in <=1/5.
  Then the panel's regression is real and the flag is on the hook for it.
* **coin flip** — every arm lands strictly between those, and in particular the
  BASE arm itself flips. Then the panel's per-rep ``cert_regressions`` on this
  instance is noise the metric cannot distinguish from a flag effect, and it has to
  be reported as such rather than scored against either arm.

Counted per CLAUDE.md §6: prints CELLS_EXECUTED and exits non-zero if it is zero.

    python -u scratchpad/issue928_certflip_noise.py --instances tls2,syn05hfsg \
        --budget 20 --reps 5 --out scratchpad/issue928_certflip_noise.json
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
    ("warm", {WARM: "1", ROUND: "0", HESS: "0"}),
    ("wr", {WARM: "1", ROUND: "1", HESS: "0"}),
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
    ap.add_argument("--instances", default="tls2,syn05hfsg")
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="scratchpad/issue928_certflip_noise.json")
    args = ap.parse_args()

    names = [s for s in args.instances.split(",") if s]
    cells, executed = [], 0
    load_start = os.getloadavg()

    for rep in range(1, args.reps + 1):
        for name in names:
            cell = {"instance": name, "rep": rep, "budget": args.budget}
            for key, env in ARMS:
                cell[key] = run_one(CORPUS / f"{name}.nl", args.budget, env)
                executed += 1
            cells.append(cell)
            print(
                f"[rep{rep}] {name:12s} "
                + " | ".join(
                    f"{k} cert={int(bool(cell[k]['gap_certified']))} "
                    f"wall={cell[k]['wall']:5.1f} b={cell[k]['bound']}"
                    for k, _e in ARMS
                ),
                flush=True,
            )

    rates: dict[str, dict[str, str]] = {}
    verdicts = {}
    for name in names:
        rows = [c for c in cells if c["instance"] == name]
        rates[name] = {
            k: f"{sum(bool(c[k]['gap_certified']) for c in rows)}/{len(rows)}" for k, _e in ARMS
        }
        counts = {k: sum(bool(c[k]["gap_certified"]) for c in rows) for k, _e in ARMS}
        n = len(rows)
        flag_effect = any(
            counts[a] >= n - 1 and counts[b] <= 1 for a, _x in ARMS for b, _y in ARMS if a != b
        )
        base_flips = 0 < counts["base"] < n
        verdicts[name] = "flag_effect" if flag_effect else ("coin_flip" if base_flips else "stable")
        print(f"{name:12s} cert rate {rates[name]}  verdict={verdicts[name]}")

    Path(args.out).write_text(
        json.dumps(
            {
                "load_start": load_start,
                "load_end": os.getloadavg(),
                "cells_executed": executed,
                "cert_rates": rates,
                "verdicts": verdicts,
                "cells": cells,
            },
            indent=2,
        )
    )
    print(f"CELLS_EXECUTED={executed}")
    return 0 if executed else 1


if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = main()
    print(f"elapsed={time.perf_counter() - t0:.1f}s")
    sys.exit(rc)
