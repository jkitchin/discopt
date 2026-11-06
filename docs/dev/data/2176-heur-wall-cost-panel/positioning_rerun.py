import json, os, sys, time, warnings
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
meth = sys.argv[1]
pm = gr.discover_models(include=["positioning"])[0].builder()
conv = convert_pyomo_gdp(pm)
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = conv.model.solve(gdp_method=meth, time_limit=120)
print("ROW", json.dumps(dict(meth=meth, flag=os.environ.get("DISCOPT_HEUR_WALL_COST"), status=r.status, obj=r.objective, nodes=r.node_count, wall=round(time.perf_counter()-t0,2), load=round(os.getloadavg()[0],2))), flush=True)
os._exit(0)
