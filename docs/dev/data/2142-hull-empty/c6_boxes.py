"""#2142 §114 step 2: ground truth for the 94 captured cstr-hull node boxes (§113's b1 capture).
Per box: (a) the §113 SCIP root-propagation instrument with big=1e19 (as §113) and big=1e29
(numerics/infinity 1e30, scip_prop2.py); (b) a WITNESS search: SCIP full B&B restricted to the
box (big=1e29, objective zero = pure feasibility, 60 s) -- any returned point is re-verified
against every spec row by the independent numpy evaluator (specio.residuals, max <= 1e-6) and
for box membership (<= 1e-6 abs + 1e-9 rel); a verified point refutes "empty" independently of
SCIP's propagation. With the cutoff: the same search with c^T x <= cutoff added.
  python -u c6_boxes.py BOXES.npz OUT.npz"""
import math, os, sys
import numpy as np
import pyscipopt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specio, scip_prop2  # noqa: E402

d, kw = specio.load(sys.argv[1])
integ = np.asarray(kw["integrality"]).astype(bool)
c = kw["c"]
nb = len(d["rec_id"])
P19 = scip_prop2.ScipProp(kw, big=1e19)
P29 = scip_prop2.ScipProp(kw, big=1e29)
FULL = {"limits/nodes": -1, "presolving/maxrounds": -1, "lp/solvefreq": 1,
        "separating/maxroundsroot": -1, "separating/maxrounds": -1, "limits/time": 60}
rows = []
n_exec = 0
for b in range(nb):
    lo, hi = d["rec_pre_lo"][b], d["rec_pre_hi"][b]
    cut = float(d["rec_cutoff"][b])
    cutv = cut if math.isfinite(cut) else None
    e19 = P19.propagate(lo, hi, cutv)[1] is None
    e29 = P29.propagate(lo, hi, cutv)[1] is None
    out = {"b": b, "id": int(d["rec_id"][b]), "e19": e19, "e29": e29}
    for use_cut in (False, True):
        sp = scip_prop2.ScipProp(kw, FULL, big=1e29)
        m = sp.m
        m.setHeuristics(pyscipopt.SCIP_PARAMSETTING.DEFAULT)
        inf = m.infinity()
        for j, v in enumerate(sp.x):
            lj = math.ceil(lo[j] - 1e-6) if integ[j] else lo[j]
            hj = math.floor(hi[j] + 1e-6) if integ[j] else hi[j]
            m.chgVarLb(v, -inf if lj <= -1e29 else float(lj))
            m.chgVarUb(v, inf if hj >= 1e29 else float(hj))
        if use_cut and cutv is not None:
            m.addCons(pyscipopt.quicksum(float(c[j]) * sp.x[j] for j in range(len(c)) if c[j] != 0) <= cutv)
        m.setObjective(0.0)
        m.setParam("limits/solutions", 1)
        m.optimize()
        st = m.getStatus()
        key = "cut" if use_cut else "nocut"
        out[key + "_status"] = st
        if m.getNSols() > 0:
            x = np.array([m.getSolVal(m.getBestSol(), v) for v in sp.x])
            r = specio.residuals(kw, x)
            tol = 1e-6 + 1e-9 * np.maximum(np.abs(lo), np.abs(hi))
            inbox = bool(np.all(x >= lo - tol) and np.all(x <= hi + tol))
            okcut = (not use_cut) or cutv is None or float(c @ x) <= cutv + 1e-6
            out[key + "_witness"] = bool(r["max"] <= 1e-6 and inbox and okcut)
            out[key + "_resid"] = float(r["max"])
            out[key + "_obj"] = float(c @ x)
        else:
            out[key + "_witness"] = False
        n_exec += 1
    rows.append(out)
    print(out, flush=True)
W = [r for r in rows if r["nocut_witness"]]
print(f"boxes {nb}: §113-instrument empty {sum(r['e19'] for r in rows)}, corrected-instrument empty "
      f"{sum(r['e29'] for r in rows)}")
print(f"verified feasible witness (no cutoff): {len(W)}/{nb}; of those, §113 called empty: "
      f"{sum(r['e19'] for r in W)}")
print(f"verified witness under the box's cutoff: {sum(r['cut_witness'] for r in rows)}/{nb}; "
      f"SCIP status counts (cut): {dict((s, sum(r['cut_status']==s for r in rows)) for s in set(r['cut_status'] for r in rows))}")
np.savez(sys.argv[2], rows=np.array([str(r) for r in rows]))
print(f"executed searches: {n_exec}")
sys.exit(0 if n_exec else 1)
