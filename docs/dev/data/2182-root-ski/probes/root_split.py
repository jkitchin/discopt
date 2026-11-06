"""#2182 task 2: split each big-m / hull solve into (setup + root) vs search.

One fresh process per (instance, method): python -u root_split.py <instance> <method>
Prints one JSON row. root_time is discopt's own snapshot after the root batch
(solver.py, cert:T0.1); search = wall - root_time.
"""

import json
import sys
import time
import warnings

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import discopt  # noqa: E402
from benchmarks import pyomo_gdp_examples as px  # noqa: E402

from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

assert discopt.__file__.startswith("/home/user/discopt/python/discopt"), discopt.__file__
name, method = sys.argv[1], sys.argv[2]
if name.startswith(px.PREFIX):
    pm = px.builder(name)()
else:
    from gdp_baseline import _spec

    pm = _spec(name).builder()
t0 = time.perf_counter()
conv = convert_pyomo_gdp(pm)
t1 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = conv.model.solve(gdp_method=method, time_limit=300)
t2 = time.perf_counter()
rt = r.root_time
row = {
    "instance": name,
    "method": method,
    "status": r.status,
    "obj": r.objective,
    "nodes": r.node_count,
    "convert": round(t1 - t0, 3),
    "solve": round(t2 - t1, 3),
    "root_time": None if rt is None else round(rt, 3),
    "search": None if rt is None else round((t2 - t1) - rt, 3),
    "n_vars": len(conv.model._variables),
    "n_cons": len(conv.model._constraints),
}
print(json.dumps(row), flush=True)
