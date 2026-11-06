"""#2183: capture the StdForm(s) med_term_purchasing big-M hands to solve_milp_std."""
from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[4])
import sys, warnings
import numpy as np
import discopt
assert discopt.__file__.startswith(REPO + "/python"), discopt.__file__
from discopt.solvers import lp_milp_highs as L
sys.path.insert(0, REPO + "/discopt_benchmarks")
from benchmarks.pyomo_gdp_examples import builder
from discopt.pyomo.gdp import convert_pyomo_gdp
warnings.simplefilter("ignore")
calls = []
orig = L.solve_milp_std
def spy(sf, **kw):
    out = orig(sf, **kw)
    calls.append((sf, kw, out))
    return out
L.solve_milp_std = spy
conv = convert_pyomo_gdp(builder("med_term_purchasing")())
r = conv.model.solve(gdp_method="big-m", time_limit=300)
print("result", r.status, r.gap_certified, r.objective, r.bound)
assert calls, "spy never fired"
for i, (sf, kw, out) in enumerate(calls):
    print(i, sf.n, sf.m, sf.int_idx.size, sf.A.nnz, {k: v for k, v in kw.items() if k != "initial_point"},
          "init" if kw.get("initial_point") is not None else "noinit", out.status, out.gap_certified,
          out.objective, out.bound, {k: v for k, v in out.stats.items() if "fixed_int" in k})
sf, kw, out = calls[0]
np.savez_compressed(sys.argv[1], c=sf.c, data=sf.A.data, indices=sf.A.indices, indptr=sf.A.indptr,
                    shape=np.array(sf.A.shape), b=sf.b, xl=sf.xl, xu=sf.xu, obj_const=sf.obj_const,
                    int_idx=sf.int_idx, gap_tolerance=kw["gap_tolerance"],
                    n_struct=-1 if kw.get("n_struct") is None else kw["n_struct"])
print("saved", len(calls))
