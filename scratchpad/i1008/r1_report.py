"""Join the two #1008 R1 panel arms and apply the CLAUDE.md §5 graduation bars.

Bar 1 — **cert-clean** (hard, zero slack): no arm may report an objective BELOW
its HiGHS reference by more than the tolerance (a false, too-good bound on a
minimization LP), and no LP that the OFF arm bounds may lose its bound in the ON
arm (a certification regression).

Bar 2 — **net-positive**: the ON arm must be measurably helpful broadly, not
merely sound. For this flag the currency is *bound retention* — how many LPs come
back with a usable bound — with wall clock as a secondary watch.

Prints an executed-comparison count and exits non-zero if it is zero (§6).
"""

import json
import os
import sys

WT = "/private/tmp/wtR1"
TOL_ABS, TOL_REL = 1e-6, 1e-4


def load(path):
    return {json.loads(line)["tag"]: json.loads(line) for line in open(path)}


off = load(os.path.join(WT, "scratchpad/i1008/r1_arm0.jsonl"))
on = load(os.path.join(WT, "scratchpad/i1008/r1_arm1.jsonl"))
tags = sorted(set(off) & set(on))
assert tags, "the two arms share no LP"


def bounded(r):
    """Did this arm come back with a usable bound?"""
    return r["status"] == "optimal"


def tol(ref):
    return max(TOL_ABS, TOL_REL * abs(ref))


n_cmp = 0
false_bounds, regressions, gains, drift = [], [], [], []
fired_on, fired_off = [], []

for t in tags:
    a, b = off[t], on[t]
    ref = a["ref"]
    n_cmp += 1

    if b["recoveries"]:
        fired_on.append((t, b["recoveries"]))
    if a["bails"]:
        fired_off.append((t, a["bails"]))

    # Bar 1a: a bound below the true optimum is a false bound, in EITHER arm.
    if ref is not None:
        for name, r in (("off", a), ("on", b)):
            if bounded(r) and r["obj"] < ref - tol(ref):
                false_bounds.append((t, name, r["obj"], ref))
        # Objective drift on LPs both arms bound.
        if bounded(a) and bounded(b) and abs(a["obj"] - b["obj"]) > tol(ref):
            drift.append((t, a["obj"], b["obj"], ref))

    # Bar 1b / Bar 2: bound retention, both directions.
    if bounded(a) and not bounded(b):
        regressions.append((t, a["obj"], b["status"]))
    if bounded(b) and not bounded(a):
        gains.append((t, a["status"], b["obj"], ref))

n_capped = sum(1 for t in tags if off[t].get("iter_capped") or on[t].get("iter_capped"))
n_off = sum(bounded(off[t]) for t in tags)
n_on = sum(bounded(on[t]) for t in tags)
w_off = sum(off[t]["wall"] for t in tags)
w_on = sum(on[t]["wall"] for t in tags)

print(f"LPs compared: {n_cmp}")
print(f"\nmechanism fired:")
print(f"  ON  arm recoveries : {len(fired_on)} LPs {fired_on}")
print(f"  OFF arm bails      : {len(fired_off)} LPs {fired_off}")
print(f"  LPs that hit the iteration cap in either arm: {n_capped}")
print(f"\nbound retention:")
print(f"  OFF bounded: {n_off}/{n_cmp}")
print(f"  ON  bounded: {n_on}/{n_cmp}")
print(f"  gains (OFF lost -> ON bounds): {len(gains)}")
for g in gains:
    print(f"    {g[0]}: off={g[1]} -> on obj={g[2]:.9g} (HiGHS {g[3]:.9g})")
print(f"  regressions (OFF bounded -> ON lost): {len(regressions)}")
for r in regressions:
    print(f"    {r[0]}: off obj={r[1]:.9g} -> on status={r[2]}")
print(f"\nfalse bounds (below the HiGHS optimum): {len(false_bounds)}")
for f in false_bounds:
    print(f"    {f[0]} [{f[1]}]: obj={f[2]:.9g} < ref={f[3]:.9g}")
print(f"objective drift on commonly-bounded LPs: {len(drift)}")
for d in drift:
    print(f"    {d[0]}: off={d[1]:.9g} on={d[2]:.9g} ref={d[3]:.9g}")
print(f"\nwall (sum, single unreplicated run - directional only):")
print(f"  OFF {w_off:.2f}s   ON {w_on:.2f}s")

cert_clean = not false_bounds and not regressions and not drift
# A mechanism that never fires is an EMPTY probe, not a passing one (CLAUDE.md
# §6). With zero recoveries the ON arm is bit-identical to the OFF arm, so
# "no regressions" is arithmetic, not evidence; net-positive cannot be earned.
fired = len(fired_on) > 0
net_positive = fired and len(gains) > 0
print(f"\ncert-clean : {'PASS' if cert_clean else 'FAIL'}")
if not fired:
    print("\nNOTE: the recovery fired on 0 panel LPs. The ON arm is therefore")
    print("      bit-identical to the OFF arm and this panel can establish")
    print("      cert-cleanliness only — net-positive is UNMEASURED, not passed.")
print(f"net-positive: {'PASS' if net_positive else 'FAIL'}  "
      f"({len(gains)} bounds recovered, mechanism fired on {len(fired_on)} LPs)")
print(f"GRADUATE    : {'YES' if (cert_clean and net_positive) else 'NO'}")

print(f"\nexecuted: comparisons={n_cmp} mechanism_firings={len(fired_on) + len(fired_off)}")
if not n_cmp:
    sys.exit(1)
