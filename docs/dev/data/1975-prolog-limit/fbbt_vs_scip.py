"""Per-column: discopt FBBT box vs SCIP presolved box (where they differ)."""
import sys
import numpy as np
from pyscipopt import Model
import discopt.modeling as dm
from discopt.tightening import fbbt_box
path = sys.argv[1]
m0 = dm.from_nl(path); bt = fbbt_box(m0); lb, ub = np.asarray(bt.lb, float), np.asarray(bt.ub, float)
m = Model(); m.hideOutput(); m.readProblem(path); m.presolve()
tv = {v.name.replace("t_", ""): v for v in m.getVars(transformed=True)}
def f(v): return "  -inf" if v <= -1e19 else (" +inf" if v >= 1e19 else f"{v:.4g}")
k = 0
for j in range(len(lb)):
    v = tv.get(f"x{j}")
    s = ("removed", "") if v is None else (f(v.getLbLocal()), f(v.getUbLocal()))
    if v is None or (abs(lb[j]) >= 1e19) != (v.getLbLocal() <= -1e19) or (abs(ub[j]) >= 1e19) != (v.getUbLocal() >= 1e19):
        print(f"x{j}: fbbt [{f(lb[j])}, {f(ub[j])}]  scip [{s[0]}, {s[1]}]"); k += 1
print("differing open sides:", k)
