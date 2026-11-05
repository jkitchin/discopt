"""#2142 final check: GDPlib cstr on the DEFAULT path (no flags set), 300 s, as gdp_parity
solves it (gdp_baseline worker), with the Pyomo incumbent check.
  PYTHONPATH=/home/user/discopt/python python -u cstr300.py {big-m|hull}"""
import os, sys, json, time
for k in list(os.environ):
    assert not k.startswith("DISCOPT_"), f"non-default env {k}"
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts"); sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
import discopt
assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
from gdp_baseline import worker
t = time.perf_counter()
rec = worker("cstr", "discopt-" + sys.argv[1], 300.0)
print(json.dumps({k: rec.get(k) for k in ("instance", "arm", "status", "objective", "bound", "node_count", "wall")}, default=str))
print("load", os.getloadavg())
n = 1
assert rec["bound"] is None or rec["bound"] <= 3.0620073 + 1e-6; n += 1
print("executed assertions:", n)
