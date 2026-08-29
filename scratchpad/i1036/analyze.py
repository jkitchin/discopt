"""Aggregate the design-size experiment: censored evals-to-tolerance per arm."""
import json, sys
import numpy as np

rows = json.load(open(sys.argv[1]))
budget = rows[0]["budget"]
funcs, arms = [], []
for r in rows:
    if r["func"] not in funcs: funcs.append(r["func"])
    if r["arm"] not in arms: arms.append(r["arm"])

print(f"censored evals-to-1e-2 (non-reached counted as the full budget {budget}); "
      f"mean over 12 seeds\n")
w = max(len(f) for f in funcs) + 2
print(f"{'function':<{w}}" + "".join(f"{a:>15}" for a in arms))
tot = {a: [] for a in arms}
for f in funcs:
    line = f"{f:<{w}}"
    for a in arms:
        sub = [r for r in rows if r["func"] == f and r["arm"] == a]
        vals = [budget if r["first"] is None else r["first"] for r in sub]
        tot[a].extend(vals)
        line += f"{np.mean(vals):>15.1f}"
    print(line)
print("-" * (w + 15 * len(arms)))
print(f"{'PANEL mean':<{w}}" + "".join(f"{np.mean(tot[a]):>15.1f}" for a in arms))
print(f"{'PANEL sd':<{w}}" + "".join(f"{np.std(tot[a]):>15.1f}" for a in arms))
print(f"{'reached (of 96)':<{w}}" +
      "".join(f"{sum(1 for r in rows if r['arm']==a and r['first'] is not None):>15d}" for a in arms))

print("\nper-function win/loss vs 'shipped' (censored mean; lower is better):")
for a in arms:
    if a == "shipped": continue
    wins = losses = ties = 0
    for f in funcs:
        base = np.mean([budget if r["first"] is None else r["first"]
                        for r in rows if r["func"] == f and r["arm"] == "shipped"])
        cur = np.mean([budget if r["first"] is None else r["first"]
                       for r in rows if r["func"] == f and r["arm"] == a])
        if cur < base - 1e-9: wins += 1
        elif cur > base + 1e-9: losses += 1
        else: ties += 1
    print(f"  {a:<15} {wins} better, {ties} tied, {losses} worse (of {len(funcs)})")
print(f"\nexecuted comparisons: {len(rows)}")
if not rows:
    sys.exit(1)
