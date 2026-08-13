"""Tabulate the #1008 H2 pivot-threshold sweep against its kill criterion.

Prints a compared-cell count; exits non-zero when it is zero (CLAUDE.md #6).
"""

import json
import sys
from collections import defaultdict

rows = [json.loads(ln) for ln in open(sys.argv[1]) if ln.strip()]
by = defaultdict(dict)
for r in rows:
    by[r["tag"]][r["u"]] = r
us = ["1.0", "0.5", "0.1", "0.01"]
us = [u for u in us if any(u in v for v in by.values())]

cells = 0
print("FILL  nnz(L+U)/nnz(B)")
print(f"{'tag':22s} " + " ".join(f"{u:>8s}" for u in us))
for tag in sorted(by):
    line = f"{tag:22s} "
    for u in us:
        r = by[tag].get(u)
        line += f" {r['fill']:8.2f}" if r else f" {'-':>8s}"
    print(line)

print("\nWALL (s)   ! = not optimal")
print(f"{'tag':22s} {'highs':>7s} " + " ".join(f"{u:>9s}" for u in us) + "   best  vs1.0")
speed = []
for tag in sorted(by):
    hw = next(iter(by[tag].values()))["highs_wall"]
    line = f"{tag:22s} {hw:7.3f} "
    w1, best, bu = None, float("inf"), None
    for u in us:
        r = by[tag].get(u)
        if not r:
            line += f" {'-':>9s}"
            continue
        cells += 1
        ok = r["status"] == "optimal"
        line += f" {r['wall']:8.3f}{'' if ok else '!'}"
        if ok:
            if u == "1.0":
                w1 = r["wall"]
            if r["wall"] < best:
                best, bu = r["wall"], u
    su = (w1 / best) if (w1 and best < float("inf")) else float("nan")
    if w1 and best < float("inf"):
        speed.append((su, tag, bu))
    print(line + f"  {str(bu):>5s} {su:5.2f}x")

print("\nITERS")
for tag in sorted(by):
    hit = next(iter(by[tag].values()))["highs_iters"]
    line = f"{tag:22s} highs={hit:6d} "
    for u in us:
        r = by[tag].get(u)
        line += f" {u}:{r['iters'] if r else '-'}"
    print(line)

print("\nSTATUS / objective-vs-HiGHS check")
bad = 0
for tag in sorted(by):
    for u in us:
        r = by[tag].get(u)
        if not r:
            continue
        if r["status"] != "optimal":
            print(f"  {tag} u={u}: status={r['status']}")
            continue
        rel = abs(r["obj"] - r["highs_obj"]) / max(1.0, abs(r["highs_obj"]))
        if rel > 1e-6:
            print(f"  {tag} u={u}: OBJECTIVE DRIFT rel={rel:.3e}")
            bad += 1
print(f"  objective drifts > 1e-6: {bad}")

if speed:
    s = sorted(x[0] for x in speed)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    print(f"\nbest-u speedup vs u=1.0: n={n} min={min(s):.2f}x median={med:.2f}x max={max(s):.2f}x")
    print("  best u per instance:", sorted({(t, b) for _, t, b in speed}))

# Fill reduction at u=0.1 on the instances the hypothesis targets (fill > 5 at u=1).
print("\nkill-criterion 1: fill drop at u=0.1 on high-fill (>5) instances")
n1 = 0
for tag in sorted(by):
    a, b = by[tag].get("1.0"), by[tag].get("0.1")
    if not (a and b) or a["fill"] <= 5.0:
        continue
    drop = 1.0 - b["fill"] / a["fill"]
    print(f"  {tag:22s} {a['fill']:6.2f} -> {b['fill']:6.2f}  drop={drop * 100:5.1f}%")
    n1 += 1
print(f"  high-fill instances evaluated: {n1}")

print("compared cells:", cells)
if cells == 0:
    sys.exit(1)
