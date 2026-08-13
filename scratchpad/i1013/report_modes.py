"""#1013: compare the stall-escape arms against `off` over the captured panel.

Reports, per arm: status changes vs `off` (the hard gate — an `optimal` that
becomes anything else is a regression, and vice versa), objective agreement on
optimal/optimal pairs, iteration and wall medians, and how often the mechanism
fired at all. Prints a compared-cell count and exits non-zero at zero (§6).
"""

import json
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(dict)
for r in rows:
    by[r["lp"]][r["arm"]] = r
arms = [a for a in ("harris", "bland", "cold") if any(a in v for v in by.values())]

compared = 0
for arm in arms:
    changes, drift, sp_wall, sp_iter, fired = [], [], [], [], 0
    for lp, v in sorted(by.items()):
        if "off" not in v or arm not in v:
            continue
        o, a = v["off"], v[arm]
        compared += 1
        if o["status"] != a["status"]:
            changes.append(
                (lp, o["status"], a["status"], o["wall"], a["wall"], o["iters"], a["iters"])
            )
        if o["status"] == a["status"] == "optimal":
            den = 1.0 + abs(o["obj"])
            drift.append(abs(o["obj"] - a["obj"]) / den)
            sp_wall.append(o["wall"] / max(a["wall"], 1e-9))
            sp_iter.append(a["iters"] - o["iters"])
        if (
            a.get("DualStabilityRepivots", 0)
            or a.get("DualDegenerateStallBails", 0)
            or (arm == "bland" and a.get("DualBlandActivations", 0))
        ):
            fired += 1
    print(f"\n=== arm `{arm}` vs `off` ===")
    print(
        f"cells compared: {sum(1 for v in by.values() if 'off' in v and arm in v)}  fired on: {fired} LPs"
    )
    print(f"status changes: {len(changes)}")
    for ch in changes:
        print(
            f"   {ch[0]:26s} {ch[1]:10s} -> {ch[2]:10s}  wall {ch[3]:.2f}->{ch[4]:.2f}  it {ch[5]}->{ch[6]}"
        )
    if drift:
        print(f"max objective drift on optimal/optimal: {max(drift):.3e} (n={len(drift)})")
        print(
            f"wall speedup: median {statistics.median(sp_wall):.3f}x  min {min(sp_wall):.3f}x  max {max(sp_wall):.3f}x"
        )
        nz = [d for d in sp_iter if d]
        print(
            f"iteration delta: median {statistics.median(sp_iter):+.0f}  changed on {len(nz)}/{len(sp_iter)} LPs"
        )
print("\ncompared cells:", compared)
if compared == 0:
    sys.exit(1)
