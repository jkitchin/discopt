"""#1013 "what remains": the QPLIB_0911 cells, run against the full corpus.

The issue's five perturbation levers (`DISCOPT_LP_REFACTOR_INTERVAL`, an LU
`pivot_threshold` knob, `DISCOPT_LU_SYMBOLIC_REUSE`) were `perf/1008` branch
flags and do **not** exist on this branch -- verified by enumerating every
`DISCOPT_*` name under `crates/discopt-core/src/lp/`. They cannot be run as
written. What is reproducible is the *class* the issue was actually testing:
levers that perturb the LU/pivot path at rounding level without touching any
tolerance, guard, or bound formula. This tree has three such knobs, used below.

For each (LP, perturbation) cell the bail is run OFF and ON, interleaved within
each rep. Status, iteration count and bail count are deterministic and are the
claim; wall is recorded but is NOT load-gated here (see the report caveat).

Each cell is a child process because the flags are read once per process.
Prints one line per cell as it completes (§10) and a comparison count (§6).
"""

import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LPS = os.path.join(HERE, "lps")
TL = float(os.environ.get("I1013_TL", "60"))
REPS = int(os.environ.get("I1013_REPS", "2"))

# Rounding-level perturbations available on THIS branch. Each changes the LU or
# refinement path, none changes a tolerance, guard, or bound formula -- the
# property the issue's levers had.
PERTURBATIONS = [
    ("baseline", {}),
    ("density_route", {"DISCOPT_LU_DENSITY_ROUTE": "1"}),
    ("hardening", {"DISCOPT_LP_FACTORIZATION_HARDENING": "1"}),
    ("refinement", {"DISCOPT_LP_ITERATIVE_REFINEMENT": "1"}),
]
ARMS = [("off", "0"), ("bail", "1")]

lps = sorted(f for f in os.listdir(LPS) if f.startswith("QPLIB_0911") and f.endswith(".npz"))
if not lps:
    raise SystemExit(f"no QPLIB_0911 LPs under {LPS} - run capture.py first")
print(f"LPs: {lps}", flush=True)

rows = []
for rep in range(REPS):
    for lp in lps:
        for pname, penv in PERTURBATIONS:
            for arm, val in ARMS:  # interleaved within the rep (§9)
                env = dict(os.environ)
                env["PYTHONPATH"] = os.path.join(ROOT, "python")
                # Without this every counter reads 0 and each cell looks clean
                # while measuring nothing; lprun.py now refuses if it is missing.
                env["DISCOPT_PROFILE"] = "1"
                env["DISCOPT_LP_DUAL_STALL_BAIL"] = val
                env.update(penv)
                out = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        os.path.join(HERE, "lprun.py"),
                        os.path.join(LPS, lp),
                        str(TL),
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                if out.returncode != 0:
                    raise SystemExit(
                        f"cell {lp}/{pname}/{arm} FAILED rc={out.returncode}\n"
                        f"{out.stdout}\n{out.stderr}"
                    )
                line = [x for x in out.stdout.splitlines() if x.startswith("RES ")]
                if not line:
                    raise SystemExit(f"cell {lp}/{pname}/{arm} produced no RES line:\n{out.stdout}")
                rec = json.loads(line[0][4:])
                rec.update(rep=rep, perturb=pname, arm=arm)
                rows.append(rec)
                print(
                    f"rep{rep} {lp[:-4]:22} {pname:14} {arm:4} "
                    f"status={rec['status']:10} iters={rec['iters']:6} "
                    f"bails={rec['DualDegenerateStallBails']} "
                    f"maxrun={rec['DualDegenerateRunMax']:6} wall={rec['wall']:.2f}s",
                    flush=True,
                )

outp = os.path.join(HERE, "cells_0911.jsonl")
with open(outp, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

# ------------------------------------------------------------------ comparison
print("\n=== off vs bail, per cell ===", flush=True)
compared = 0
converted, regressed, unchanged = [], [], []
for lp in lps:
    for pname, _ in PERTURBATIONS:

        def pick(arm):
            return [
                r for r in rows if r["lp"] == lp[:-4] and r["perturb"] == pname and r["arm"] == arm
            ]

        o, b = pick("off"), pick("bail")
        if not o or not b:
            continue
        compared += 1
        os_, bs_ = {r["status"] for r in o}, {r["status"] for r in b}
        oi = statistics.median(r["iters"] for r in o)
        bi = statistics.median(r["iters"] for r in b)
        nb = max(r["DualDegenerateStallBails"] for r in b)
        mr = max(r["DualDegenerateRunMax"] for r in o)
        tag = "same"
        if os_ != bs_:
            if "iter_limit" in os_ and bs_ == {"optimal"}:
                tag = "CONVERTED iter_limit->optimal"
                converted.append((lp, pname))
            elif "optimal" in os_ and "optimal" not in bs_:
                tag = "REGRESSED"
                regressed.append((lp, pname, os_, bs_))
            else:
                tag = f"changed {os_}->{bs_}"
        else:
            unchanged.append((lp, pname))
        print(
            f"  {lp[:-4]:22} {pname:14} off={sorted(os_)[0]:10} iters={oi:7.0f} | "
            f"bail={sorted(bs_)[0]:10} iters={bi:7.0f} | bails={nb} maxrun_off={mr} "
            f"-> {tag}",
            flush=True,
        )

print(f"\ncompared cells: {compared}")
print(f"  converted iter_limit->optimal : {len(converted)} {converted}")
print(f"  regressed                     : {len(regressed)} {regressed}")
print(f"  unchanged status              : {len(unchanged)}")
if compared == 0:
    print("PROBE FIRED ZERO COMPARISONS", flush=True)
    sys.exit(1)
