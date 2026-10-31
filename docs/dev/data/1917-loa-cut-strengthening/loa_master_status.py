import sys, warnings, collections
import numpy as np
warnings.simplefilter("ignore")
src = open("docs/dev/data/1917-loa-cut-strengthening/loa_strengthen.py").read().split("names = sys.argv")[0]
g = {"__name__": "ls"}; exec(src, g)
from discopt.gdp.algorithms import master as master_mod
log = []
import os, json
OPTS = json.loads(os.environ.get("OPTS", "{}"))
orig_solve = master_mod.MasterMILP.solve
def solve(self, *a, **k):
    r = orig_solve(self, *a, **k)
    log.append((str(getattr(r, "status", None)), getattr(r, "bound", None), getattr(r, "objective", None), self.config_of(r.x) if getattr(r, "x", None) is not None else None))
    return r
master_mod.MasterMILP.solve = solve
from discopt.pyomo.gdp import convert_pyomo_gdp
from benchmarks import gdplib_runner as gr
for on in (False, True):
    g["MODE"]["on"] = on; log.clear()
    conv = convert_pyomo_gdp(gr.discover_models(include=["pyomo.eight_process"])[0].builder())
    r = conv.model.solve(gdp_method="loa", time_limit=60, **OPTS)
    print("ON" if on else "OFF", r.status, r.bound, (r.solver_stats or {}).get("stop_reason"), (r.solver_stats or {}).get("lb_invalid_reason"))
    for i, (s, b, o, c) in enumerate(log[:6]): print("   master", i, s, b, o, c)
    print("   statuses:", collections.Counter(s for s, *_ in log), "distinct configs:", len({c for *_, c in log}))
print("probe rows:", len(log))
