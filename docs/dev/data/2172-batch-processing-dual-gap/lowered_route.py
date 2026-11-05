import sys, warnings, time, json, os, logging
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
from discopt.transformations import get
meth, tl = sys.argv[1], float(sys.argv[2])
t0=time.perf_counter()
class H(logging.Handler):
    def emit(s, r):
        m=r.getMessage()
        if any(k in m for k in ("route","LP/NLP","master","incumbent","certif")):
            print(f"[{time.perf_counter()-t0:6.1f}] {m[:220]}", flush=True)
lg=logging.getLogger("discopt"); lg.setLevel(logging.INFO); lg.addHandler(H()); lg.propagate=False
pm = gr.discover_models(include=["batch_processing"])[0].builder()
conv = convert_pyomo_gdp(pm)
w = get("gdp").apply(conv.model, method=meth)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = w.solve(time_limit=tl)
print("RESULT", json.dumps(dict(meth=meth, status=r.status, obj=r.objective, bound=r.bound, gap_certified=getattr(r,"gap_certified",None), route=getattr(r,"algorithm_route",None), wall=time.perf_counter()-t0), default=str), flush=True)
os._exit(0)
