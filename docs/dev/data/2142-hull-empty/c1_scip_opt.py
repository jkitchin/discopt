"""#2142 §114 step 1a: solve the captured cstr-hull spec with SCIP (full B&B, global box) to get
an optimal point x* in the spec's 320 lifted columns; verify x* against every spec row with the
independent numpy evaluator (specio.residuals).  python -u c1_scip_opt.py BOXES.npz OUT.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2142-propagation"))
import specio, scip_prop  # noqa: E402

d, kw = specio.load(sys.argv[1])
sp = scip_prop.ScipProp(kw, {"limits/nodes": -1, "presolving/maxrounds": -1, "lp/solvefreq": 1,
                              "separating/maxroundsroot": -1, "separating/maxrounds": -1,
                              "conflict/enable": True})
m = sp.m
m.setHeuristics(__import__("pyscipopt").SCIP_PARAMSETTING.DEFAULT)
m.optimize()
print("status", m.getStatus(), "obj", m.getObjVal(), "dual", m.getDualbound(), "nodes", m.getNNodes())
s = m.getBestSol()
x = np.array([m.getSolVal(s, v) for v in sp.x])
r = specio.residuals(kw, x)
print("x* residual max", r["max"], "row", r["row"].max(), "term", r["term"].max(), "blf", r["blf"].max(),
      "int", r["int"].max(), "bound", r["bound"].max())
assert r["max"] < 1e-5, r["max"]
print("objective c@x", float(kw["c"] @ x))
np.savez(sys.argv[2], x=x)
print("executed assertions: 1")
