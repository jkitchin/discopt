"""Compare cutoff-weight arms with the polluted (before) and guard-only (after) panels.

Usage: ``python pseudocost_cutoff_arms.py docs/dev/data/pseudocost-sentinel 0 1e-4 1e-2 1``.
The arms were produced by ``pseudocost_sentinel_panel.py run`` with a temporary
weight override (removed before merge); the shipped weight is 1e-4.
"""

import json
import sys
from pathlib import Path

D = Path(sys.argv[1])  # docs/dev/data/pseudocost-sentinel


def load(p):
    return {r["name"]: r for r in map(json.loads, Path(p).read_text().splitlines())}


arms = {"before": load(D / "before.jsonl"), "guard": load(D / "after.jsonl")}
for w in sys.argv[2:]:
    arms[f"w={w}"] = load(D / f"cutoff_w{w}.jsonl")
names = sorted(set.intersection(*(set(a) for a in arms.values())))
n_checks = 0
print(f"{len(names)} instances; arms {list(arms)}")
for k, a in arms.items():
    viol = sum(len(a[x]["violations"]) for x in names)
    cert = sum(int(a[x]["gap_certified"]) for x in names)
    noinc = [x for x in names if a[x]["objective"] is None]
    print(f"{k:8s} certified={cert} violations={viol} no-incumbent={len(noinc)} {noinc}")
    n_checks += 1
# certified in every arm: node totals
both = [x for x in names if all(a[x]["gap_certified"] for a in arms.values())]
print(f"\ncertified in all arms ({len(both)}): node totals")
for k, a in arms.items():
    print(f"  {k:8s} {sum(a[x]['nodes'] for x in both)}")
print("\nper-instance nodes (certified in all arms, changed somewhere):")
for x in both:
    v = [arms[k][x]["nodes"] for k in arms]
    if len(set(v)) > 1:
        print(f"  {x:24s} " + " ".join(f"{n:6d}" for n in v))
lost = {
    k: [x for x in names if arms["before"][x]["gap_certified"] and not a[x]["gap_certified"]]
    for k, a in arms.items()
}
print(f"\ncert lost vs before: {lost}")
if "w=0" in arms:
    diff = [x for x in both if arms["w=0"][x]["nodes"] != arms["guard"][x]["nodes"]]
    print(f"w=0 vs guard-only, certified rows with different node counts: {diff}")
print(f"executed comparisons: {n_checks * len(names)}")
sys.exit(0 if names else 2)
