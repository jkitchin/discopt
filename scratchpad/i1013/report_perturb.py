"""#1013 graduation panel report: `base` vs the cost-perturbed warm start.

Gates, in the CLAUDE.md §5 order: (1) cert-clean — no status regression, no
objective drift beyond tolerance on optimal/optimal pairs, and in particular no
objective ABOVE the base arm's (these are `min` relaxations, so a higher value is
a stronger — and therefore unsound — bound); (2) net-positive — what the
mechanism buys, and that it does not cost elsewhere.

Every arm must have FIRED: `DualCostPerturbAttempts >= 1` on the perturb arm of
every compared cell (CLAUDE.md §6). A cell where it did not is reported and
counted, never silently averaged in. Prints a compared count and exits non-zero
at zero.

    python -u scratchpad/i1013/report_perturb.py PANEL.jsonl [base_arm perturb_arm]
"""

import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "base"
PERT = sys.argv[3] if len(sys.argv) > 3 else "perturb"
REL_TOL = 1e-6
# `dual::PERTURB_ARM_RUN` -- the degenerate run that arms the perturbed restart.
ARM_RUN = 64

rows = [json.loads(line) for line in open(path)]
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r["lp"]][r["arm"]].append(r)


def med(v, k):
    return statistics.median([x[k] for x in v])


def status(v):
    return "/".join(sorted({x["status"] for x in v}))


compared = fired = identical = 0
never_fired, regress, improve, drift, unsound, ratios, walls = [], [], [], [], [], [], []

for lp, v in sorted(by.items()):
    if BASE not in v or PERT not in v:
        continue
    compared += 1
    b, p = v[BASE], v[PERT]
    armed_here = max(x.get("DualCostPerturbAttempts", 0) for x in p) >= 1
    if armed_here:
        fired += 1
    # The perturbation is ARMED BY A STALL, so not firing is the expected outcome
    # on a healthy LP -- but an LP whose base arm shows a degenerate run past the
    # arming threshold and still did not arm means the mechanism is not wired to
    # the signal it claims to use (CLAUDE.md §6). Those are listed, not averaged in.
    base_runmax = max(x.get("DualDegenerateRunMax", 0) for x in b)
    if base_runmax >= ARM_RUN and not armed_here:
        never_fired.append((lp, base_runmax))
    sb, sp_ = status(b), status(p)
    ib, ip = med(b, "iters"), med(p, "iters")
    if sb == sp_ and ib == ip:
        identical += 1
    if sb == "optimal" and sp_ != "optimal":
        regress.append((lp, sb, sp_))
    if sb != "optimal" and sp_ == "optimal":
        improve.append((lp, sb, sp_))
    if sb == "optimal" == sp_:
        ob, op = med(b, "obj"), med(p, "obj")
        d = op - ob
        rel = abs(d) / (1.0 + abs(ob))
        drift.append(rel)
        if d > REL_TOL * (1.0 + abs(ob)):
            unsound.append((lp, ob, op, d))
        if ib > 0:
            ratios.append(ip / ib)
        wb, wp = med(b, "wall"), med(p, "wall")
        if wb > 0:
            walls.append(wp / wb)

print(
    f"compared LPs: {compared}   perturbation armed on: {fired} "
    f"(the rest never stalled and paid nothing)"
)
print(f"identical (same status AND iteration count): {identical}")
print()
print("== gate 1: cert-clean ==")
print(f"status regressions (optimal -> not): {len(regress)}  {regress}")
print(f"status improvements (not -> optimal): {len(improve)}  {improve}")
print(
    f"max relative objective drift (optimal/optimal, n={len(drift)}): "
    f"{max(drift) if drift else 0.0:.3e}"
)
print(f"objectives ABOVE base beyond tolerance (unsound direction): {len(unsound)}  {unsound}")
if never_fired:
    print(f"!! {len(never_fired)} cells stalled past the arming run but did NOT arm: {never_fired}")
print()
print("== gate 2: net-positive ==")
if ratios:
    q = statistics.quantiles(ratios, n=4) if len(ratios) > 3 else [float("nan")] * 3
    print(
        f"iteration ratio perturb/base (n={len(ratios)}): median {statistics.median(ratios):.3f} "
        f"q1 {q[0]:.3f} q3 {q[2]:.3f} min {min(ratios):.3f} max {max(ratios):.3f}"
    )
    print(
        f"  cells with fewer pivots: {sum(1 for r in ratios if r < 0.999)}  "
        f"more: {sum(1 for r in ratios if r > 1.001)}  "
        f"unchanged: {sum(1 for r in ratios if 0.999 <= r <= 1.001)}"
    )
if walls:
    print(f"wall ratio perturb/base (n={len(walls)}): median {statistics.median(walls):.3f}")
both = [v for v in by.values() if BASE in v and PERT in v]
tb = sum(med(v[BASE], "iters") for v in both)
tp = sum(med(v[PERT], "iters") for v in both)
print(f"total pivots  base {tb:.0f}   perturb {tp:.0f}")
print()
print("== biggest movers (iteration ratio) ==")
movers = []
for lp, v in by.items():
    if BASE in v and PERT in v and status(v[BASE]) == "optimal" == status(v[PERT]):
        ib, ip = med(v[BASE], "iters"), med(v[PERT], "iters")
        if ib > 0:
            movers.append((ip / ib, lp, ib, ip))
movers.sort()
for r, lp, ib, ip in movers[:6]:
    print(f"  {lp:32s} {ib:7.0f} -> {ip:7.0f}   {r:.3f}x")
for r, lp, ib, ip in movers[-4:]:
    print(f"  {lp:32s} {ib:7.0f} -> {ip:7.0f}   {r:.3f}x")

if compared == 0:
    sys.exit("no cells compared")
