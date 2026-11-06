"""#2172 probe: solve the GDP-lowered batch_processing model directly.

    python -u lowered_route.py hull 120

Lowering first lets the convex-MINLP router see the lowered (convex) model, which it
routes to ``mip-nlp/lp_nlp_bb``; ``Model.solve(gdp_method=...)`` judges the source GDP
instead. Prints the route/master log lines and one ``RESULT`` JSON line.
"""

import json
import logging
import os
import sys
import time
import warnings

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from benchmarks import gdplib_runner as gr  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
from discopt.transformations import get  # noqa: E402

meth, tl = sys.argv[1], float(sys.argv[2])
t0 = time.perf_counter()
KEYS = ("route", "LP/NLP", "master", "incumbent", "certif")


class H(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if any(k in m for k in KEYS):
            print(f"[{time.perf_counter() - t0:6.1f}] {m[:220]}", flush=True)


lg = logging.getLogger("discopt")
lg.setLevel(logging.INFO)
lg.addHandler(H())
lg.propagate = False
pm = gr.discover_models(include=["batch_processing"])[0].builder()
conv = convert_pyomo_gdp(pm)
w = get("gdp").apply(conv.model, method=meth)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = w.solve(time_limit=tl)
res = dict(
    meth=meth,
    status=r.status,
    obj=r.objective,
    bound=r.bound,
    gap_certified=getattr(r, "gap_certified", None),
    route=getattr(r, "algorithm_route", None),
    wall=time.perf_counter() - t0,
)
print("RESULT", json.dumps(res, default=str), flush=True)
os._exit(0)
