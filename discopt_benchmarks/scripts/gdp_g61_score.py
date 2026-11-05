"""G6.1 gate scorer (#1917): per-family solved, incorrect, and SGM ratio with a bootstrap CI.

Usage::

    python discopt_benchmarks/scripts/gdp_g61_score.py [data_dir]

``data_dir`` defaults to ``docs/dev/data/gdp_g61`` and holds ``rep1.jsonl`` (all
instance x arm runs), ``timing_rep2.jsonl`` / ``timing_rep3.jsonl`` (the
(instance, family) pairs both sides solved in rep 1, re-run) and ``serial.jsonl``
(a jobs=1 control sample), all written by ``gdp_baseline.py``.

SGM follows spec section 1.2: 1 s shift, every run that is not solved counts at
the 300 s limit. Each (instance, arm) contributes the mean of ``log(wall + 1)``
over the repeats it has. Unsolved runs were not repeated: they sit at the
limit by definition. The CI resamples instances (10,000 draws, seed 0).
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

FAMILIES = {
    "big-m": ("discopt-big-m", "pyomo-bigm"),
    "hull": ("discopt-hull", "pyomo-hull"),
    "loa": ("discopt-loa", "pyomo-loa"),
    "gloa": ("discopt-gloa", "pyomo-gloa"),
    "ric": ("discopt-ric", "pyomo-ric"),
    "enumerate": ("discopt-enumerate", "pyomo-enumerate"),
    "lbb": ("discopt-lbb", "pyomo-lbb"),
}


def load(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[(r["instance"], r["arm"])] = r
    return rows


def wall(r: dict) -> float:
    if r["verdict"] == "solved":
        return min(float(r["wall"]), r["time_limit"])
    return r["time_limit"]


def main(argv: list[str]) -> int:
    d = Path(argv[0] if argv else "docs/dev/data/gdp_g61")
    r1 = load(d / "rep1.jsonl")
    timing = [load(d / "timing_rep2.jsonl"), load(d / "timing_rep3.jsonl")]
    serial = load(d / "serial.jsonl")
    reps = [r1, *timing]
    checks = 0

    flips = [
        (k, r1[k]["verdict"], t[k]["verdict"])
        for t in timing
        for k in t
        if t[k]["verdict"] != r1[k]["verdict"]
    ]
    checks += sum(len(t) for t in timing)
    print(f"timing-rep rows {[len(t) for t in timing]}; verdict flips vs rep 1: {flips}")
    bad = [
        k
        for rep in reps
        for k, r in rep.items()
        if k[1].startswith("discopt") and r["verdict"] in ("incorrect", "claimed_unverified")
    ]
    print(f"discopt incorrect/unverified rows, all reps: {bad}")

    insts = sorted({i for i, _ in r1})
    rng = random.Random(0)
    print(
        f"\n{'family':10s} {'solved d/p':>10s} {'inc d/p':>8s} {'SGM d':>7s} {'SGM p':>7s} "
        f"{'ratio':>6s} {'95% CI':>15s} {'per-rep ratios':>16s}  gate: point / CI-upper <= 1"
    )
    for da, pa in FAMILIES.values():

        def logw(arm: str, inst: str) -> float:
            vals = [math.log(wall(rep[(inst, arm)]) + 1) for rep in reps if (inst, arm) in rep]
            return sum(vals) / len(vals)

        def sgm(arm: str) -> float:
            return math.exp(sum(logw(arm, i) for i in insts) / len(insts)) - 1

        diff = [logw(da, i) - logw(pa, i) for i in insts]
        checks += len(diff)
        ratio = math.exp(sum(diff) / len(diff))
        boot = sorted(
            math.exp(sum(rng.choices(diff, k=len(diff))) / len(diff)) for _ in range(10000)
        )
        lo, hi = boot[250], boot[9750]

        per_rep = []
        for rep in reps:
            w = {
                (i, a): wall(rep[(i, a)] if (i, a) in rep else r1[(i, a)])
                for i in insts
                for a in (da, pa)
            }
            per_rep.append(
                math.exp(
                    sum(math.log(w[(i, da)] + 1) - math.log(w[(i, pa)] + 1) for i in insts)
                    / len(insts)
                )
            )
        sd = sum(r1[(i, da)]["verdict"] == "solved" for i in insts)
        sp = sum(r1[(i, pa)]["verdict"] == "solved" for i in insts)
        idd = sum(r1[(i, da)]["verdict"] == "incorrect" for i in insts)
        ip = sum(r1[(i, pa)]["verdict"] == "incorrect" for i in insts)
        base = idd == 0 and sd >= sp
        pt = "PASS" if base and ratio <= 1.0 else "FAIL"
        ci = "PASS" if base and hi <= 1.0 else "FAIL"
        reps_s = " ".join(f"{x:.2f}" for x in per_rep)
        fam = da.removeprefix("discopt-")
        print(
            f"{fam:10s} {sd:4d}/{sp:<4d}  {idd:3d}/{ip:<3d} {sgm(da):7.2f} {sgm(pa):7.2f} "
            f"{ratio:6.2f} [{lo:5.2f},{hi:5.2f}] {reps_s:>16s}  {pt} / {ci}"
        )

    # jobs=4 vs jobs=1 inflation on runs that were solved both ways and took >= 1 s.
    logs: dict[str, list[float]] = {"discopt": [], "pyomo": []}
    for k, s in serial.items():
        for rep in reps:
            if k in rep and s["verdict"] == rep[k]["verdict"] == "solved" and s["wall"] >= 1:
                logs[k[1].split("-")[0]].append(math.log(rep[k]["wall"] / s["wall"]))
    allv = logs["discopt"] + logs["pyomo"]
    checks += len(allv)
    print(
        f"\nserial control, jobs=4 / jobs=1 wall: gm {math.exp(statistics.mean(allv)):.2f} "
        f"(sd log {statistics.stdev(allv):.2f}, n {len(allv)}); "
        + "; ".join(
            f"{side} gm {math.exp(statistics.mean(v)):.2f} n {len(v)}"
            for side, v in logs.items()
            if v
        )
    )
    print(f"comparisons: {checks}")
    return 0 if checks > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
