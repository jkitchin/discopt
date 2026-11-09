"""Gap plan P3 entry, step 3: dump SCIP's root cuts from its aggregation separator.

SCIP (pyscipopt, presolve off, heuristics off, node limit 1), separators restricted to
`aggregation` with its c-MIR / flow-cover / knapsack-cover generators (the
`p3_scip_root_ablation.py` "agg" arm). An event handler on LPSOLVED at the root (depth 0; the last one wins) copies every LP
row that is not a model constraint (a cut) with its name, sides, support (variable
names, coefficients, types, bounds), LP activity and dual value, and the root LP point.
Writes one JSON object per instance; prints an executed count; exits non-zero on 0
(CLAUDE.md §6). Usage: python p3_scip_root_cuts.py --instances a,b --out X.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
import p42_rens_panel as R  # noqa: E402, N812

AGG4 = {"aggregation", "cmir", "flowcover", "knapsackcover"}


def dump(path: str) -> dict:
    import pyscipopt as P

    m = P.Model()
    m.hideOutput()
    m.readProblem(path)
    m.setParam("limits/nodes", 1)
    m.setParam("limits/time", 300)
    m.setParam("presolving/maxrounds", 0)
    m.setHeuristics(P.SCIP_PARAMSETTING.OFF)
    for k, v in m.getParams().items():
        if k.startswith("separating/") and k.endswith("/freq"):
            name = k.split("/")[1]
            m.setParam(k, 0 if name in AGG4 else -1)
    cons_names = {c.name for c in m.getConss()}
    out: dict = {}

    class H(P.Eventhdlr):
        def eventinit(self):
            self.model.catchEvent(P.SCIP_EVENTTYPE.LPSOLVED, self)

        def eventexec(self, event):
            if self.model.getDepth() != 0:
                return
            rows = self.model.getLPRowsData()
            cuts, model_rows = [], 0
            for r in rows:
                if r.name in cons_names:
                    model_rows += 1
                    continue
                cols = r.getCols()
                vals = r.getVals()
                cuts.append({
                    "name": r.name,
                    "lhs": r.getLhs(),
                    "rhs": r.getRhs(),
                    "constant": r.getConstant(),
                    "support": [
                        {"var": c.getVar().name, "coef": v, "type": c.getVar().vtype(),
                         "lb": c.getLb(), "ub": c.getUb(), "lp": c.getPrimsol()}
                        for c, v in zip(cols, vals)
                    ],
                    "activity": self.model.getRowLPActivity(r),
                    "dual": self.model.getRowDualSol(r),
                })  # fmt: skip
            out["n_model_rows_in_lp"] = model_rows
            out["cuts"] = cuts
            out["lp_obj"] = self.model.getLPObjVal()
            out["lp_x"] = {v.name: v.getLPSol() for v in self.model.getVars()}

    m.includeEventhdlr(H(), "dumpcuts", "dump root cuts")
    m.optimize()
    out["dual_bound"] = m.getDualbound()
    stats = Path(path).with_suffix(".scipstats.txt").name
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        fn = str(Path(td) / stats)
        m.writeStatistics(fn)
        txt = Path(fn).read_text()
    i = txt.find("Separators")
    out["separator_stats"] = txt[i : txt.find("\n\n", i)] if i >= 0 else ""
    out["n_cuts_applied"] = m.getNCutsApplied()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pop = {m["name"]: m for m in R.population()}
    res, executed = {}, 0
    for nm in a.instances.split(","):
        d = dump(str(REPO / pop[nm]["path"]))
        d["opt"] = pop[nm]["oracle_objective"]
        res[nm] = d
        executed += 1
        cuts = d.get("cuts", [])
        tight = sum(1 for c in cuts if abs(c["dual"]) > 1e-9)
        print(f"{nm}: dual bound {d['dual_bound']:.6g} (opt {d['opt']}), "
              f"{len(cuts)} cuts in the root LP, {tight} with nonzero dual", flush=True)  # fmt: skip
    Path(a.out).write_text(json.dumps(res, separators=(",", ":")))
    print(f"executed: {executed}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
