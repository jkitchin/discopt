"""Next-candidate entry measurement (#2142 item 3): node relaxation strength at FIXED boxes.
On each captured node's post-propagation box (the box our node LP was solved on), SCIP's
root dual bound with LP ON, enforcement ON (its nonlinear handler adds cuts to the LP when
the LP point violates a term), separators OFF, presolve/OBBT/redcost/heuristics/symmetry
OFF, no objective limit, limits/nodes=1 -- vs our node bound `rec_bound` on that box.
Prints comparison counts (§6)."""
import sys, os, json, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scip_prop
d = dict(np.load(sys.argv[1]))
KEYS = [k for k in d if not k.startswith("rec_") and k != "prop_rounds"]
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in KEYS}
extra = dict(kv.split("=") for kv in sys.argv[2:])
params = {"lp/solvefreq": 1, "numerics/boundstreps": 1e-9}
for k, v in extra.items():
    params[k] = int(v) if v.lstrip("-").isdigit() else float(v)
sp = scip_prop.ScipProp(kw, params)
rows = []
for b in range(len(d["rec_id"])):
    ours = float(d["rec_bound"][b])
    if not math.isfinite(ours):
        continue
    sp.m.freeTransform()
    st, _, _ = sp.propagate(d["rec_lo"][b], d["rec_hi"][b], None)
    db = sp.m.getDualbound() if st != "infeasible" else math.inf
    rows.append((int(d["rec_id"][b]), ours, db, st))
ours = np.array([r[1] for r in rows]); sc = np.array([r[2] for r in rows])
gap = sc - ours
fin = np.isfinite(sc)
print("boxes with a finite node bound:", len(rows), " SCIP infeasible:", int((~fin).sum()))
print("SCIP tighter by >1e-4 abs:", int((gap[fin] > 1e-4).sum()), " ours tighter by >1e-4:", int((gap[fin] < -1e-4).sum()))
print("SCIP - ours over finite: median %.4f mean %.4f max %.4f" % (np.median(gap[fin]), np.mean(gap[fin]), np.max(gap[fin])))
for r in sorted(rows, key=lambda r: -(r[2] - r[1]) if math.isfinite(r[2]) else 0)[:8]:
    print("  node", r[0], "ours %.4f scip %.4f" % (r[1], r[2]), r[3])
print("executed comparisons:", len(rows)); sys.exit(0 if rows else 1)
