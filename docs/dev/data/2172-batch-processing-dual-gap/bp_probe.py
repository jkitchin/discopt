import sys, time, os, json, warnings, logging
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
method, tl = sys.argv[1], float(sys.argv[2])
verbose = len(sys.argv) > 3 and sys.argv[3] == "v"
t0 = time.perf_counter()
if verbose:
    class H(logging.Handler):
        def emit(s, r):
            print(f"[{time.perf_counter()-t0:7.1f}] {r.getMessage()[:200]}", flush=True)
    lg = logging.getLogger("discopt"); lg.setLevel(logging.INFO); lg.addHandler(H()); lg.propagate=False
pm = gr.discover_models(include=["batch_processing"])[0].builder()
conv = convert_pyomo_gdp(pm)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cProfile; _pr = cProfile.Profile(); _pr.enable()
    r = conv.model.solve(gdp_method=method, time_limit=tl, max_nodes=int(os.environ.get("MAXN", 10**9)))
_pr.disable(); _pr.dump_stats(os.environ.get("PROF_OUT", "prof.out"))
st = getattr(r, "solver_stats", None) or {}
print("RESULT", json.dumps(dict(method=method, status=r.status, obj=r.objective, bound=r.bound,
      nodes=getattr(r,"node_count",None), wall=time.perf_counter()-t0, env={k:v for k,v in os.environ.items() if k.startswith("DISCOPT_")}), default=str), flush=True)
print("STATS", json.dumps({k: v for k, v in st.items() if not hasattr(v, "__len__") or isinstance(v, str)}, default=str)[:6000], flush=True)
for k in ("rust_time","jax_time","python_time","wall_time"):
    print(k, getattr(r, k, None))
os._exit(0)
