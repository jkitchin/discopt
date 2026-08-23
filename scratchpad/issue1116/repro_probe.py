"""#1116 reproduction probe: is <instance> bit-reproducible run to run?

Usage: python -u repro_probe.py <instance-stem> <max_nodes> <reps>

Solves the same instance REPS times in ONE process with a node budget and NO wall
limit, and reports the distinct (node_count, bound, objective) values seen. Prints
per-rep progress as it goes (CLAUDE.md §10) and an executed-comparison count, and
exits non-zero when it made no comparisons (§6).
"""

import hashlib
import json
import sys
import time

from discopt.modeling.core import from_nl

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"

stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
path = NL.format(stem)

rows = []
for rep in range(reps):
    model = from_nl(path)
    t0 = time.perf_counter()
    r = model.solve(max_nodes=max_nodes)
    wall = time.perf_counter() - t0
    row = {
        "rep": rep,
        "nodes": int(r.node_count or 0),
        "bound": float(r.bound) if r.bound is not None else None,
        "objective": float(r.objective) if r.objective is not None else None,
        "status": r.status,
        "wall": round(wall, 2),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
for key in ("nodes", "bound", "objective", "status"):
    vals = [r[key] for r in rows]
    distinct = sorted({repr(v) for v in vals})
    comparisons += len(vals) - 1
    verdict = "STABLE" if len(distinct) == 1 else "VARIES"
    print(f"{key:10s} {verdict}  distinct={len(distinct)}  {distinct}", flush=True)

digest = hashlib.sha1(
    json.dumps([[r["nodes"], r["bound"]] for r in rows], sort_keys=True).encode()
).hexdigest()[:10]
print(f"comparisons={comparisons} digest={digest}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
