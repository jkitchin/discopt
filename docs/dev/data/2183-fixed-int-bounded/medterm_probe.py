"""#2183 probe: med_term_purchasing big-M (in-repo Pyomo example) on the highs-milp route."""
from pathlib import Path as _P
REPO = str(_P(__file__).resolve().parents[4])
import os, sys, warnings, importlib.metadata as md
import discopt
assert discopt.__file__.startswith(REPO + "/python"), discopt.__file__
from discopt.solvers import lp_milp_highs as L
src = open(L.__file__).read()
print("highspy", md.version("highspy"), "fix-marker", "_fixed_integer_lp_cannot_refute" in src, flush=True)
sys.path.insert(0, REPO + "/discopt_benchmarks")
from benchmarks.pyomo_gdp_examples import builder
from discopt.pyomo.gdp import convert_pyomo_gdp
warnings.simplefilter("ignore")
conv = convert_pyomo_gdp(builder("med_term_purchasing")())
r = conv.model.solve(gdp_method="big-m", time_limit=300)
print(f"status={r.status} cert={r.gap_certified} obj={r.objective!r} bound={r.bound!r} nodes={r.node_count} wall={r.wall_time:.2f}")
print("route:", (r.algorithm_route or "")[:600])
