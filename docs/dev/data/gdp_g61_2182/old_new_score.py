import json, math, statistics, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(dict)
for r in rows:
    by[(r["instance"], r["arm"])][r["tree"]] = r
fam = defaultdict(list); flips = []; n = 0
for (i, a), t in sorted(by.items()):
    if set(t) != {"old", "new"}:
        continue
    o, w = t["old"], t["new"]
    so, sn = o.get("status") == "optimal", w.get("status") == "optimal"
    if so != sn:
        flips.append((i, a, o.get("status"), w.get("status"), o.get("wall"), w.get("wall")))
    if so and sn and o.get("wall") and w.get("wall"):
        n += 1
        fam[a].append((math.log((w["wall"] + 1) / (o["wall"] + 1)), i, o["wall"], w["wall"]))
allr = []
print(f"{'arm':18s} n  gm new/old (1 s shift)  worst cells")
for a, v in sorted(fam.items()):
    allr += [x[0] for x in v]
    worst = sorted(v, reverse=True)[:3]
    print(f"{a:18s} {len(v):2d} {math.exp(statistics.mean(x[0] for x in v)):6.2f}   " +
          "; ".join(f"{i} {o:.1f}->{w:.1f}" for _, i, o, w in worst))
print(f"all: n={n} gm={math.exp(statistics.mean(allr)):.2f} sd(log)={statistics.pstdev(allr):.2f}")
print("status flips (old vs new):", flips)
assert n > 0
print("executed-assertions: 1")
