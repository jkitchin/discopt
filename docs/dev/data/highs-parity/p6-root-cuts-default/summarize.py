"""Summarize a root_cuts_engine_panel.py run JSON: per-row cert/bound/nodes/wall, totals.

rel(r-h) is (rust - highs)/max(1,|highs|) of the median bound; its sign means tighter or
looser by the objective sense (syn05m is a max problem, the rest are min).
"""

import json
import statistics
import sys

d = json.load(open(sys.argv[1]))
tot = {"highs": 0.0, "rust": 0.0}
nodes = {"highs": 0, "rust": 0}
cert = {"highs": 0, "rust": 0}
tight = looser = 0
n = 0
for r in d["rows"]:
    h, ru = r["highs"], r["rust"]
    n += 1
    bh, br = statistics.median(h["bound"]), statistics.median(ru["bound"])
    rel = (br - bh) / max(1.0, abs(bh))
    for arm, x in (("highs", h), ("rust", ru)):
        tot[arm] += x["wall_med"]
        nodes[arm] += sum(x["nodes"])
        cert[arm] += sum(x["cert"])
    print(
        f"{r['instance'] + '@' + r['config']:28s} cert {sum(h['cert'])}/{sum(ru['cert'])} "
        f"bound_med {bh:.10g} {br:.10g} rel(r-h)={rel:+.1e} "
        f"nodes {h['nodes']} {ru['nodes']} wall_med {h['wall_med']:.2f} {ru['wall_med']:.2f} "
        f"ratio {ru['wall_med'] / h['wall_med']:.3f}"
    )
print(
    "rows",
    n,
    "certified runs h/r",
    cert,
    "nodes",
    nodes,
    f"node ratio {nodes['rust'] / nodes['highs']:.3f}",
    "wall sum h/r",
    {k: round(v, 1) for k, v in tot.items()},
    f"ratio {tot['rust'] / tot['highs']:.3f}",
)
print("checks", d["checks"], "violations", len(d["violations"]), "errors", len(d["errors"]))
