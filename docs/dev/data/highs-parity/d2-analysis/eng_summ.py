"""Summarize d2_probe.py rows per (instance, arm).

Usage: python eng_summ.py ROWS.jsonl[.gz] ENVKEY FIELDS
ENVKEY is the env var that names the arm (absent = OFF); FIELDS a comma list of the keys below.
"""

import gzip
import json
import sys
from collections import defaultdict

key = sys.argv[2] if len(sys.argv) > 2 else "D2_ARM"
opener = gzip.open if sys.argv[1].endswith(".gz") else open
rows = [json.loads(ln) for ln in opener(sys.argv[1], "rt")]
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    if "error" in r:
        print("ERROR", r["name"], r["env"], r["error"][-300:])
        continue
    arm = r["env"].get(key, "OFF")
    ph = {}
    ct = {}
    for t in r["profile"]:
        for k, (c, s) in t["phases"].items():
            a = ph.setdefault(k, [0, 0.0])
            a[0] += c
            a[1] += s
        for k, v in t["counters"].items():
            ct[k] = ct.get(k, 0) + v
    d = agg[r["name"]][arm]
    d.append(
        dict(
            status=r["status"],
            nodes=r["nodes"],
            wall=r["wall"],
            obj=r["objective"],
            bound=r["bound"],
            cert=r["gap_certified"],
            prp=ph.get("PivotRowPrice", [0, 0])[1],
            prpn=ph.get("PivotRowPrice", [0, 0])[0],
            sweep=ph.get("PriceSweep", [0, 0])[1],
            nodelp=ph.get("NodeLpSolve", [0, 0])[1],
            sparse=ct.get("PrimalRowPriceSparse", 0),
            dense=ct.get("PrimalRowPriceDense", 0),
            vfail=ct.get("PrimalRowPriceVerifyFail", 0),
            p1=ct.get("Phase1Pivots", 0),
            p2=ct.get("Phase2Pivots", 0),
            dcd=ct.get("DualColdDecided", 0),
            dca=ct.get("DualColdAttempts", 0),
            dcu=ct.get("DualColdUndecided", 0),
            dcp=ct.get("DualColdPrepRefused", 0),
            dcs=ct.get("DualColdShortBasis", 0),
            dcx=ct.get("DualColdExchanges", 0),
            rej=ct.get("DualPrepRejectShape", 0),
            xf=ct.get("CutLpExchangeFactor", 0),
            xp=ct.get("CutLpExchangePivots", 0),
            xfail=ct.get("CutLpExchangeFail", 0),
            relaxed=ct.get("CutLpRelaxed", 0),
            deleted=ct.get("CutLpDeleted", 0),
        )
    )
fields = (
    sys.argv[3].split(",")
    if len(sys.argv) > 3
    else ["status", "nodes", "wall", "prp", "prpn", "sparse", "dense", "vfail", "nodelp"]
)
for n, arms in agg.items():
    for arm, ds in sorted(arms.items()):
        out = []
        for f in fields:
            vals = [d[f] for d in ds]
            if (
                isinstance(vals[0], (int, float))
                and not isinstance(vals[0], bool)
                and vals[0] is not None
            ):
                out.append(
                    f"{f}={'/'.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in vals)}"
                )
            else:
                out.append(f"{f}={'/'.join(str(v) for v in vals)}")
        print(f"{n:20s} {arm:4s} " + " ".join(out))
