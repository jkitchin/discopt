"""#2142 scipx d1: discopt cstr big-M tree census at a deterministic node budget.
Records per LP-solved node (id, parent, status, bound, cutoff) via node_observer and the
kernel result; derives propagation-fathomed nodes = node_count - observed.
  python -u d1_tree_stats.py MAX_NODES OUT.npz"""
import sys, time, json
import numpy as np
import discopt, discopt._rust as R
assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
max_nodes, out = int(sys.argv[1]), sys.argv[2]
n_assert = 0
recs = []
def obs(r):
    recs.append((r["id"], -1 if r["parent"] is None else r["parent"], r["status"], r["bound"], np.nan if r["cutoff"] is None else r["cutoff"],
                 r["lo"].copy(), r["hi"].copy(), r.get("pre_lo", r["lo"]).copy(), r.get("pre_hi", r["hi"]).copy()))
kres, kcfg = [], []
_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    assert not a
    kcfg.append({kk: (v if isinstance(v, (int, float, bool, str, type(None))) else type(v).__name__) for kk, v in k.items() if not hasattr(v, "shape")})
    k["node_observer"] = obs
    r = _orig(*a, **k)
    kres.append(r)
    return r
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="big-m", time_limit=3600, deterministic=True, max_nodes=max_nodes)
print("solve", res.status, res.objective, repr(res.bound), res.node_count, round(time.perf_counter()-t0,1), flush=True)
assert kres; n_assert += 1
main = max(range(len(kres)), key=lambda i: (kres[i].get("node_count", 0) if isinstance(kres[i], dict) else 0))
r = kres[main]
print("kernel calls", len(kres))
print("config", json.dumps(kcfg[main], default=str))
print("result", json.dumps({k: v for k, v in r.items() if not hasattr(v, "shape") and not isinstance(v, (list, tuple))} if isinstance(r, dict) else str(r), default=str))
st = np.array([x[2] for x in recs])
print("observed LP nodes", len(recs), {s: int((st == s).sum()) for s in set(st.tolist())})
assert len(recs) > 0; n_assert += 1
np.savez(out, id=np.array([x[0] for x in recs]), parent=np.array([x[1] for x in recs]), status=st,
         bound=np.array([x[3] for x in recs], float), cutoff=np.array([x[4] for x in recs], float),
         lo=np.stack([x[5] for x in recs]), hi=np.stack([x[6] for x in recs]),
         pre_lo=np.stack([x[7] for x in recs]), pre_hi=np.stack([x[8] for x in recs]))
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
