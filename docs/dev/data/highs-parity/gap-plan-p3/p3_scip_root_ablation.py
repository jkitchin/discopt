"""Gap plan P3 zero-code entry: which SCIP separator closes the root gap on the P3 rows?

SCIP (pyscipopt, presolve off, threads 1, node limit 1, so the bound is the root's after
separation) per instance and arm. Arms switch separators by `separating/<name>/freq`
(-1 = off, 0 = root only):
  none  -- no separation at all (the root LP bound)
  all   -- SCIP defaults (root only)
  agg   -- `aggregation` with its c-MIR / flow-cover / knapsack-cover generators
           (in SCIP 6 the `cmir`, `flowcover`, `knapsackcover` entries are dummy
           separators whose frequency switches those generators inside `aggregation`;
           turning them off leaves `aggregation` silent)
  gom   -- `gomory` only
  impl  -- `impliedbounds` only (variable-bound / implied-bound cuts)
  mcf   -- `mcf` only (multi-commodity-flow network cuts)
  zh    -- `zerohalf` only
  noagg -- every default separator except the four aggregation entries
Root gap = |opt - bound| / max(1, |opt|) in %. Prints an executed count; exits
non-zero on 0 (CLAUDE.md §6).
Usage: python p3_scip_root_ablation.py --instances a,b --out X.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
import p42_rens_panel as R  # noqa: E402, N812

SEPAS = [
    "aggregation", "gomory", "clique", "zerohalf", "mcf", "impliedbounds", "intobj",
    "strongcg", "cmir", "flowcover", "knapsackcover", "oddcycle", "disjunctive",
    "rlt", "minor", "mixing", "interminor", "convexproj", "gauge", "eccuts",
    "closecuts", "rapidlearning", "cgmip", "lagromory",
]


def run(path: str, arm: str, time_limit: float) -> dict:
    import pyscipopt

    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(path)
    m.setParam("limits/time", time_limit)
    m.setParam("limits/nodes", 1)
    m.setParam("presolving/maxrounds", 0)
    m.setParam("lp/threads", 1)
    m.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    agg4 = {"aggregation", "cmir", "flowcover", "knapsackcover"}
    keep = {
        "none": set(),
        "agg": agg4,
        "gom": {"gomory"},
        "impl": {"impliedbounds"},
        "mcf": {"mcf"},
        "zh": {"zerohalf"},
    }.get(arm)
    present = []
    for s in SEPAS:
        key = f"separating/{s}/freq"
        try:
            cur = m.getParam(key)
        except LookupError:
            continue
        present.append(s)
        if arm == "all":
            m.setParam(key, 0 if cur >= 0 else cur)
        elif arm == "noagg":
            m.setParam(key, -1 if s in agg4 else (0 if cur >= 0 else cur))
        else:
            m.setParam(key, 0 if s in keep else -1)
    if arm == "none":
        m.setParam("separating/maxroundsroot", 0)
    m.optimize()
    return {"bound": m.getDualbound(), "time": m.getSolvingTime(), "sepas_present": present}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--arms", default="none,all,agg,gom,impl,mcf,zh,noagg")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pop = {m["name"]: m for m in R.population()}
    executed = 0
    with open(a.out, "w") as fh:
        for nm in a.instances.split(","):
            mem = pop[nm]
            opt = mem["oracle_objective"]
            for arm in a.arms.split(","):
                r = run(str(REPO / mem["path"]), arm, a.time_limit)
                gap = 100.0 * abs(opt - r["bound"]) / max(1.0, abs(opt))
                row = {"name": nm, "arm": arm, "opt": opt, **r, "root_gap_pct": gap}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                executed += 1
                print(f"{nm:22} {arm:6} bound={r['bound']:.6g} gap={gap:7.3f}% "
                      f"t={r['time']:.1f}s", flush=True)  # fmt: skip
    print(f"executed: {executed}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
