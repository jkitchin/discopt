"""#2172 probe: one GDPlib batch_processing solve, cProfiled.

    MAXN=300 DISCOPT_GDP_CONFIG_FEAS_FALLBACK=0 python -u bp_probe.py hull 900 [v]

``MAXN`` caps ``max_nodes``; ``PROF_OUT`` names the cProfile dump; ``v`` echoes the
``discopt`` INFO log. Prints one ``RESULT`` JSON line and one ``STATS`` line.
"""

import cProfile
import json
import logging
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from benchmarks import gdplib_runner as gr  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

method, tl = sys.argv[1], float(sys.argv[2])
verbose = len(sys.argv) > 3 and sys.argv[3] == "v"
t0 = time.perf_counter()
if verbose:

    class H(logging.Handler):
        def emit(self, r):
            print(f"[{time.perf_counter() - t0:7.1f}] {r.getMessage()[:200]}", flush=True)

    lg = logging.getLogger("discopt")
    lg.setLevel(logging.INFO)
    lg.addHandler(H())
    lg.propagate = False
pm = gr.discover_models(include=["batch_processing"])[0].builder()
conv = convert_pyomo_gdp(pm)
pr = cProfile.Profile()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pr.enable()
    r = conv.model.solve(
        gdp_method=method, time_limit=tl, max_nodes=int(os.environ.get("MAXN", 10**9))
    )
    pr.disable()
pr.dump_stats(os.environ.get("PROF_OUT", "prof.out"))
env = {k: v for k, v in os.environ.items() if k.startswith("DISCOPT_")}
res = dict(
    method=method,
    status=r.status,
    obj=r.objective,
    bound=r.bound,
    nodes=getattr(r, "node_count", None),
    wall=time.perf_counter() - t0,
    env=env,
)
print("RESULT", json.dumps(res, default=str), flush=True)
st = getattr(r, "solver_stats", None) or {}
scalars = {k: v for k, v in st.items() if not hasattr(v, "__len__") or isinstance(v, str)}
print("STATS", json.dumps(scalars, default=str)[:6000], flush=True)
os._exit(0)
