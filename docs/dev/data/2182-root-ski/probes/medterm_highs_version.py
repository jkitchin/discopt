import sys, os, warnings, discopt
assert discopt.__file__.startswith(os.getcwd() + "/python"), discopt.__file__
sys.path.insert(0,"discopt_benchmarks"); sys.path.insert(0,"discopt_benchmarks/scripts")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
warnings.simplefilter("ignore")
conv = convert_pyomo_gdp(_spec("med_term_purchasing").builder())
r = conv.model.solve(gdp_method="big-m", time_limit=300)
lab = r.algorithm_route or ""
print(f"{os.environ.get('TAG','')}: status={r.status} cert={r.gap_certified} obj={r.objective} bound={r.bound} nodes={r.node_count} wall={r.wall_time:.2f}")
print("   route:", lab[:400])
