"""#1975: is the SIDE the ray moves toward still open after SCIP presolve (and after FBBT)?"""
import json, sys
import numpy as np
from pyscipopt import Model
import discopt.modeling as dm
from discopt.tightening import fbbt_box
name, path, rayj = sys.argv[1], sys.argv[2], sys.argv[3]
d = {int(k): v for k, v in json.loads(rayj).items()}
bt = fbbt_box(dm.from_nl(path)); lb, ub = np.asarray(bt.lb, float), np.asarray(bt.ub, float)
m = Model(); m.hideOutput(); m.readProblem(path); m.presolve()
tv = {v.name.replace("t_", ""): v for v in m.getVars(transformed=True)}
BIG = 1e19
def side_open(lo, hi, dj): return (hi >= BIG) if dj > 0 else (lo <= -BIG)
fb = [j for j, dj in d.items() if side_open(lb[j], ub[j], dj)]
sc = []
for j, dj in d.items():
    v = tv.get(f"x{j}")
    if v is not None and side_open(v.getLbLocal(), v.getUbLocal(), dj):
        sc.append(j)
print(json.dumps({"name": name, "ray_cols": len(d), "fbbt_side_open": len(fb),
                  "scip_presolve_side_open": len(sc), "open_cols": sc}))
