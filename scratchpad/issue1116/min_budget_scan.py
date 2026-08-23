"""#1116: find the SMALLEST node budget at which the instance is already unstable.

Usage: python -u min_budget_scan.py <instance-stem> <reps> <budget> [<budget> ...]

Every subsequent #1116 experiment costs one solve per rep, so the whole
investigation is priced by the smallest budget that still reproduces the bug.
For each budget this solves the instance ``reps`` times in one process and
reports STABLE/VARIES on (nodes, bound, objective).

Per-rep progress (§10), loaded-module assertion (§8), executed-comparison count
with a non-zero exit when it is zero (§6), and no swallowed exceptions (§7).
"""

import json
import sys

import discopt
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, reps = sys.argv[1], int(sys.argv[2])
budgets = [int(x) for x in sys.argv[3:]]

comparisons = 0
for budget in budgets:
    rows = []
    for rep in range(reps):
        model = from_nl(NL.format(stem))
        r = model.solve(max_nodes=budget)
        row = {
            "budget": budget,
            "rep": rep,
            "nodes": int(r.node_count or 0),
            "bound": repr(float(r.bound)) if r.bound is not None else None,
            "objective": repr(float(r.objective)) if r.objective is not None else None,
            "status": r.status,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    verdicts = []
    for key in ("nodes", "bound", "objective", "status"):
        distinct = {repr(x[key]) for x in rows}
        comparisons += len(rows) - 1
        verdicts.append(f"{key}={'STABLE' if len(distinct) == 1 else 'VARIES'}({len(distinct)})")
    print(f"BUDGET {budget}: " + "  ".join(verdicts), flush=True)

print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
