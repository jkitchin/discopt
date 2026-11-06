"""#2142 §114 step 1e: a witness for c4's minimal empty set. SCIP's root propagation calls
{col 318 in [9.99999997e15, 1e16]} empty with no cutoff (c4_minimal.log; col 318 = w316*w317, the
product of two eps-perspective terms of binary col 8). By hand, y8 = 0 gives w = 1e8, col 318 = 1e16
(inside the box) and col 313 = w308*w309*w310 = 1e24 -- above SCIP's default numerics/infinity
(1e20). H3: the §113 instrument cannot represent any perspective binary at 0.
Test: SCIP full solve (not the propagation instrument) with col 8 fixed to 0, under the §113
setting (big=1e19, infinity 1e20) and with big=1e29 (infinity 1e30); verify any returned point
against every spec row with the independent numpy evaluator. Kill of H3: no verified point with
y8 = 0 exists under big=1e29 either.   python -u c5_witness.py BOXES.npz OUT.npz"""
import os, sys
import numpy as np
import pyscipopt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specio, scip_prop2  # noqa: E402

d, kw = specio.load(sys.argv[1])
FULL = {"limits/nodes": -1, "presolving/maxrounds": -1, "lp/solvefreq": 1,
        "separating/maxroundsroot": -1, "separating/maxrounds": -1, "limits/time": 300}
n = 0
for big in (1e19, 1e29):
    sp = scip_prop2.ScipProp(kw, FULL, big=big)
    m = sp.m
    m.setHeuristics(pyscipopt.SCIP_PARAMSETTING.DEFAULT)
    m.chgVarUb(sp.x[8], 0.0)
    m.optimize()
    st = m.getStatus(); n += 1
    print(f"big={big:g}: SCIP full solve with y8=0: status {st}, nodes {m.getNNodes()}", flush=True)
    if m.getNSols() == 0:
        continue
    x = np.array([m.getSolVal(m.getBestSol(), v) for v in sp.x])
    r = specio.residuals(kw, x)
    print("  objective", float(kw["c"] @ x), "residual max %.2e (row %.2e term %.2e blf %.2e int %.2e bound %.2e)" % (
        r["max"], r["row"].max(), r["term"].max(), r["blf"].max(), r["int"].max(), r["bound"].max()))
    for j in (8, 316, 317, 318, 308, 309, 310, 312, 313, 158, 159, 319, 315, 306, 307):
        print(f"    x[{j}] = {x[j]!r}")
    assert r["max"] < 1e-6, r["max"]; n += 1
    print("  col 318 inside the box the §113 instrument calls empty:", bool(9.99999997e15 <= x[318] <= 1e16))
    np.savez(sys.argv[2], x=x)
for big in (1e19, 1e29):
    sp = scip_prop2.ScipProp(kw, big=big)
    lo, hi = kw["global_lo"].copy(), kw["global_hi"].copy()
    lo[318], hi[318] = 9.99999997e15, 1e16
    print(f"propagation instrument big={big:g} on {{col 318 in [9.99999997e15, 1e16]}}:",
          sp.propagate(lo, hi, None)[0]); n += 1
print(f"executed assertions: {n}")
