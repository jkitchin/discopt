"""#1975 sizing: inject the three missing deductions as bounds, re-run FBBT + root LP.

(1) x**p, p non-integer      => x >= 0          (anywhere: constraints AND objective)
(2) log(x) / x*log(x)        => x >= 0          (anywhere, incl. the objective)
(3) x*y == c, c > 0, y >= 0  => x >= 0          (sign of a product)
Done here by walking the .nl via SCIP's presolve? No: by the minimal rule set, computed
from SCIP's PRESOLVED lower bounds restricted to sign information only (lb := max(lb, 0)
where SCIP proved lb >= 0), so nothing beyond sign/domain facts is borrowed.
"""
import json, sys
import numpy as np
from pyscipopt import Model
import discopt.modeling as dm
from discopt._relax import mccormick_lp as ML
from discopt.tightening import fbbt_box
name, path = sys.argv[1], sys.argv[2]
m = dm.from_nl(path)
sc = Model(); sc.hideOutput(); sc.readProblem(path); sc.presolve()
tv = {v.name.replace("t_", ""): v for v in sc.getVars(transformed=True)}
n = sum(int(np.prod(v.shape)) if v.shape else 1 for v in m._variables)
bt0 = fbbt_box(m)
lb0, ub0 = np.asarray(bt0.lb, float), np.asarray(bt0.ub, float)
# sign facts only: a column SCIP proves nonnegative (or nonpositive) gets that sign
lb, ub = lb0.copy(), ub0.copy()
nsign = 0
for j in range(n):
    v = tv.get(f"x{j}")
    if v is None:
        continue
    if v.getLbLocal() >= -1e-9 and lb[j] < 0:
        lb[j] = 0.0; nsign += 1
    if v.getUbLocal() <= 1e-9 and ub[j] > 0:
        ub[j] = 0.0; nsign += 1
# re-propagate from the signed box
off = 0
for v in m._variables:
    sz = int(np.prod(v.shape)) if v.shape else 1
    v.lb = np.reshape(lb[off:off + sz], v.shape) if v.shape else float(lb[off])
    v.ub = np.reshape(ub[off:off + sz], v.shape) if v.shape else float(ub[off])
    off += sz
bt = fbbt_box(m)
lb1, ub1 = np.asarray(bt.lb, float), np.asarray(bt.ub, float)
open0 = int(np.sum((np.abs(lb0) >= 1e19) | (np.abs(ub0) >= 1e19)))
open1 = int(np.sum((np.abs(lb1) >= 1e19) | (np.abs(ub1) >= 1e19)))
res0 = ML.MccormickLPRelaxer(dm.from_nl(path)).solve_at_node(lb0, ub0)
res1 = ML.MccormickLPRelaxer(m).solve_at_node(lb1, ub1)
print(json.dumps({"name": name, "sign_facts": nsign, "open_cols_before": open0, "open_cols_after": open1,
                  "root_before": [res0.status, res0.lower_bound], "root_after": [res1.status, res1.lower_bound]}))
