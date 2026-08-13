"""#1013 graduation panel report: `off` vs the shipped bail, over the captured LPs.

Gates, in the CLAUDE.md §5 order: (1) cert-clean — no status regression, no
objective drift beyond tolerance on optimal/optimal pairs; (2) net-positive —
what the mechanism buys where it fires, and that it is inert elsewhere.

Per-LP values are the median over reps. Prints a compared count and exits
non-zero at zero (§6).
"""

import json
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r["lp"]][r["arm"]].append(r)


def med(v, k):
    return statistics.median([x[k] for x in v])


def status(v):
    s = sorted(set(x["status"] for x in v))
    return "/".join(s)


compared = changed = fired = identical = 0
drift, speed, regress, improve = [], [], [], []
for lp, v in sorted(by.items()):
    if "off" not in v or "bail" not in v:
        continue
    compared += 1
    o, a = v["off"], v["bail"]
    if max(x.get("DualDegenerateStallBails", 0) for x in a):
        fired += 1
    if med(o, "iters") == med(a, "iters") and status(o) == status(a):
        identical += 1
    if status(o) != status(a):
        changed += 1
        (improve if status(o) != "optimal" and status(a) == "optimal" else regress).append(
            (lp, status(o), status(a), med(o, "wall"), med(a, "wall"))
        )
    elif status(o) == "optimal":
        drift.append((abs(med(o, "obj") - med(a, "obj")) / (1 + abs(med(o, "obj"))), lp))
        speed.append((med(o, "wall") / max(med(a, "wall"), 1e-9), lp))

print(
    f"LPs compared: {compared}   bail fired on: {fired}   unchanged (same status+iters): {identical}"
)
print("\n[gate 1: cert-clean]")
print(f"  status regressions (optimal -> not, or any loss): {len(regress)}")
for r in regress:
    print(f"     {r}")
print(f"  status improvements: {len(improve)}")
for r in improve:
    print(f"     {r[0]:24s} {r[1]} -> {r[2]}   {r[3]:.2f}s -> {r[4]:.2f}s")
if drift:
    d = max(drift)
    print(f"  max objective drift on optimal/optimal pairs: {d[0]:.2e} ({d[1]}, n={len(drift)})")
print("\n[gate 2: net-positive]")
speed.sort()
if speed:
    print(f"  wall speedup median {statistics.median([s for s, _ in speed]):.3f}x")
    print("  worst 5: " + ", ".join(f"{s:.2f}x {l}" for s, l in speed[:5]))
    print("  best 5:  " + ", ".join(f"{s:.2f}x {l}" for s, l in speed[-5:]))
print("  cells where the bail fired:")
for lp, v in sorted(by.items()):
    if "bail" in v and max(x.get("DualDegenerateStallBails", 0) for x in v["bail"]):
        o, a = v["off"], v["bail"]
        print(
            f"     {lp:24s} {status(o):10s} {med(o, 'wall'):7.2f}s -> {status(a):10s} {med(a, 'wall'):7.2f}s"
        )
print("\ncompared LPs:", compared)
if compared == 0:
    sys.exit(1)
