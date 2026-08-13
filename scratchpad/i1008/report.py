"""Tabulate the #1008 cadence sweep: wall / iters / status per (LP, interval).

Prints a compared-cell count and exits non-zero when it is zero (§6).
"""

import json
import sys
from collections import defaultdict

path = sys.argv[1]
rows = [json.loads(ln) for ln in open(path) if ln.strip()]
by = defaultdict(dict)
meta = {}
for r in rows:
    by[r["tag"]][r["interval"]] = r
    meta[r["tag"]] = (r["rows"], r["nnz"], r["highs_wall"], r["highs_iters"])

ivs = sorted({r["interval"] for r in rows}, key=lambda s: int(s) if s.isdigit() else 0)
hdr = f"{'tag':22s} {'rows':>5s} {'highs':>7s}"
for iv in ivs:
    hdr += f" {iv:>10s}"
hdr += "   best  vs48"
print(hdr)
cells = 0
speedups = []
for tag in sorted(by):
    nrow, nnz, hw, hit = meta[tag]
    line = f"{tag:22s} {nrow:5d} {hw:7.3f}"
    best_iv, best_w = None, float("inf")
    w48 = None
    for iv in ivs:
        r = by[tag].get(iv)
        if r is None:
            line += f" {'-':>10s}"
            continue
        cells += 1
        mark = "" if r["status"] == "optimal" else "!"
        line += f" {r['wall']:9.3f}{mark or ' '}"
        if r["status"] == "optimal":
            if iv == "48":
                w48 = r["wall"]
            if r["wall"] < best_w:
                best_w, best_iv = r["wall"], iv
    su = (w48 / best_w) if (w48 and best_w < float("inf")) else float("nan")
    if w48:
        speedups.append(su)
    line += f"  {str(best_iv):>5s} {su:5.2f}x"
    print(line)

print()
print("iters by interval")
for tag in sorted(by):
    line = f"{tag:22s} highs={meta[tag][3]:6d}"
    for iv in ivs:
        r = by[tag].get(iv)
        line += f" {iv}:{r['iters'] if r else '-'}"
    print(line)

if speedups:
    speedups.sort()
    n = len(speedups)
    med = speedups[n // 2] if n % 2 else 0.5 * (speedups[n // 2 - 1] + speedups[n // 2])
    print(f"\nbest-interval speedup vs 48: n={n} min={min(speedups):.2f}x "
          f"median={med:.2f}x max={max(speedups):.2f}x")
print("compared cells:", cells)
if cells == 0:
    sys.exit(1)
