"""syngas hull 120 s; counts governor calls and wall-priced decisions (#2176)."""
import json, os, sys, time, warnings
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
import discopt.solver as S
calls = {"governor": 0, "wall_priced": 0, "clock_starts": 0}
real = S._improver_within_contingent
def spy(cost, **kw):
    calls["governor"] += 1
    st, key = kw.get("heur_state") or {}, kw.get("key")
    w = st.get("walls")
    if (key is not None and kw.get("node_wall") is not None and w is not None and w.get(key)
            and kw["tree"].incumbent() is not None):
        calls["wall_priced"] += 1
    return real(cost, **kw)
S._improver_within_contingent = spy
real_cs = S._improver_clock_start
def cs(st, key):
    calls["clock_starts"] += 1
    return real_cs(st, key)
S._improver_clock_start = cs
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
pm = gr.discover_models(include=["syngas"])[0].builder()
conv = convert_pyomo_gdp(pm)
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = conv.model.solve(gdp_method="hull", time_limit=120)
print("ROW", json.dumps(dict(flag=os.environ.get("DISCOPT_HEUR_WALL_COST"), status=r.status, obj=r.objective,
      bound=r.bound, nodes=r.node_count, wall=round(time.perf_counter() - t0, 1), load=os.getloadavg()[0], **calls)), flush=True)
os._exit(0)
