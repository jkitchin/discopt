"""Capture the first LOA/RIC master's matrix form on an instance; pickle it."""
import sys, pickle, warnings
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python")
sys.path.insert(0, W + "/discopt_benchmarks"); sys.path.insert(0, W + "/discopt_benchmarks/scripts")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
import discopt.solvers.lp_backend as lb
cap = []
orig = lb.get_milp_solver
def get(*a, **k):
    f = orig(*a, **k)
    def wrapped(**kw):
        cap.append(kw); raise SystemExit("captured")
    return wrapped
lb.get_milp_solver = get
conv = convert_pyomo_gdp(_spec(sys.argv[1]).builder())
try:
    conv.model.solve(gdp_method=sys.argv[2], time_limit=60)
except SystemExit:
    pass
assert cap, "probe did not fire"
pickle.dump(cap[0], open(sys.argv[3], "wb"))
kw = cap[0]
print("captured", {k: getattr(v, "shape", v if not hasattr(v, "__len__") else len(v)) for k, v in kw.items()})
