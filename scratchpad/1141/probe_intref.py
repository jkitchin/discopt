import os, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
from discopt._relax.model_utils import flat_variable_bounds
KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)
os.environ["DISCOPT_OA_NODE_CUTS"] = "0"
m = portfolio2.build(**KW)
r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
            time_limit=120, gap_tolerance=1e-4)
lb,_ = flat_variable_bounds(m)
x = np.zeros(len(lb)); k=0
isint=[]
for v in m._variables:
    x[k:k+v.size] = np.atleast_1d(np.asarray(r.x[v.name],float)).ravel()
    isint += [str(v.var_type)]*v.size
    k += v.size
bi = [i for i,t in enumerate(isint) if 'BINARY' in t or 'INTEGER' in t]
frac = max(abs(x[i]-round(x[i])) for i in bi)
print("binaries:", len(bi), "max integrality residual:", frac)
print("objective:", r.objective)
