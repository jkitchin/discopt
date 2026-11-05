"""#2142 item 1: hda hull root bound, OFF vs ON (DISCOPT_HULL_EXACT_PERSPECTIVE), deterministic.
  PYTHONPATH=/home/user/discopt/python python -u root.py {0|1} MAX_NODES [INSTANCE]"""
import os, sys, time
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = sys.argv[1]
import discopt
assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from gdp_baseline import _spec
from discopt.pyomo.gdp import convert_pyomo_gdp
import discopt._relax.gdp_reformulate as G
name = sys.argv[3] if len(sys.argv) > 3 else "hda"
m = convert_pyomo_gdp(_spec(name).builder()).model
t = time.perf_counter()
r = m.solve(gdp_method="hull", deterministic=True, max_nodes=int(sys.argv[2]), time_limit=float(os.environ.get("TL", "120")))
print(f"RESULT {name} flag={sys.argv[1]} nodes={r.node_count} status={r.status} obj={r.objective} bound={r.bound!r} wall={time.perf_counter()-t:.1f}", G._HULL_EXACT_STATS, flush=True)
n = 1 + (1 if sys.argv[1] == "1" and G._HULL_EXACT_STATS["homogenized"] > 0 else 0) + (1 if sys.argv[1] == "0" and sum(G._HULL_EXACT_STATS.values()) == 0 else 0)
print("executed assertions:", n)
