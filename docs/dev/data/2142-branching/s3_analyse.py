"""#2142 step 3: agreement between our branching decisions and SCIP's on the same boxes."""
import json, sys
import numpy as np
from collections import Counter
cap = np.load(sys.argv[1]); rows = json.load(open(sys.argv[2]))
n_orig = int(cap["n_orig"])
br = [r for r in rows if "scip_col" in r]
print("boxes", len(rows), "SCIP branched", len(br), "SCIP status", dict(Counter(r["status"] for r in rows)))
agree = [r for r in br if r["scip_col"] == r["our_col"]]
print(f"variable agreement: {len(agree)}/{len(br)} = {len(agree)/len(br):.3f}")
print("SCIP picks integer:", sum(r["scip_col"] < 0 or bool(cap["integrality"][r["scip_col"]]) for r in br),
      " ours integer:", sum(r["our_int"] for r in br))
print("SCIP picks original col:", sum(r["scip_col"] < n_orig for r in br), " ours original:", sum(r["our_col"] < n_orig for r in br))
print("top SCIP cols:", Counter(r["scip_col"] for r in br).most_common(8))
print("top our cols :", Counter(r["our_col"] for r in br).most_common(8))
# point comparison where the column agrees and is continuous
pos = lambda v, lo, hi: (v - lo) / (hi - lo)
d = [pos(r["scip_val"], r["lo_c"], r["hi_c"]) - pos(r["our_val"], r["lo_c"], r["hi_c"]) for r in agree if not r["our_int"]]
d = np.array(d)
if len(d):
    print(f"same column ({len(d)} continuous): |rel point diff| quantiles", np.round(np.quantile(np.abs(d), [0, .25, .5, .75, .9, 1]), 4))
# SCIP's own rule check: point = 0.75 mid + 0.25 lp (root: the box is SCIP's global domain)
chk = [abs(r["scip_val"] - (0.375 * (r["scip_lo"] + r["scip_hi"]) + 0.25 * r["scip_lp_val"])) / (r["scip_hi"] - r["scip_lo"])
       for r in br if r["scip_lp_val"] is not None and not bool(cap["integrality"][r["scip_col"]])]
print("SCIP point vs 0.75*mid+0.25*lp (rel):", np.round(np.quantile(chk, [0, .5, .9, 1]), 6) if chk else None, len(chk))
# relative LP-value difference for the agreed column: ours inverted from p = 0.75 mid + 0.25 x
lpd = []
for r in agree:
    if r["our_int"] or r["scip_lp_val"] is None:
        continue
    mid = 0.5 * (r["lo_c"] + r["hi_c"]); x_ours = (r["our_val"] - 0.75 * mid) / 0.25
    lpd.append((r["scip_lp_val"] - x_ours) / (r["hi_c"] - r["lo_c"]))
if lpd:
    print("same column: (SCIP LP value - our LP value)/width quantiles of |.|", np.round(np.quantile(np.abs(lpd), [0, .25, .5, .75, 1]), 4))
print("executed assertions:", len(br))
