"""#2142: pure McCormick LP bound of the hull-lowered model at its declared box (no tree,
no FBBT), separate=False and True.  PYTHONPATH=... python -u lp.py FLAG [INSTANCE]"""
import os, sys, time
import numpy as np
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = sys.argv[1]
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
from discopt import transformations as Tr
from discopt._relax.mccormick_lp import MccormickLPRelaxer
name = sys.argv[2] if len(sys.argv) > 2 else "hda"
m = Tr.create_using("gdp.hull", convert_pyomo_gdp(_spec(name).builder()).model)
lb = np.concatenate([np.broadcast_to(np.asarray(v.lb, float), v.shape).ravel() for v in m._variables])
ub = np.concatenate([np.broadcast_to(np.asarray(v.ub, float), v.shape).ravel() for v in m._variables])
R = MccormickLPRelaxer(m)
n = 0
for sep in (False, True):
    t = time.perf_counter()
    r = R.solve_at_node(lb, ub, time_limit=300, separate=sep)
    print(f"LP {name} flag={sys.argv[1]} pmc={'off' if os.environ.get('DISCOPT_2142_NO_PMC') else 'on'} separate={sep} status={r.status} lb={r.lower_bound!r} n={lb.size} t={time.perf_counter()-t:.1f}", flush=True)
    n += 1
print("executed assertions:", n)
