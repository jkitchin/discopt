"""#2142 §114: GDPlib cstr through Model.solve with DISCOPT_INT_ROUND_EXACT from the environment.
  DISCOPT_INT_ROUND_EXACT=0|1 python -u e_arm.py MAX_NODES [hull|big-m] [TIME_LIMIT] [det|nodet]
Asserts the loaded extension is this tree's build carrying the flag (CLAUDE.md §8), prints the
kernel calls and an executed-assertion count (§6)."""
import os, sys, time
import discopt, discopt._rust as R

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert b"DISCOPT_INT_ROUND_EXACT" in open(R.__file__, "rb").read(), "stale _rust: no §114 flag"
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

max_nodes = int(sys.argv[1])
method = sys.argv[2] if len(sys.argv) > 2 else "hull"
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0
det = (sys.argv[4] if len(sys.argv) > 4 else "det") == "det"
flag = os.environ.get("DISCOPT_INT_ROUND_EXACT", "0")
calls, n_assert = [], 2
_o = R.solve_spatial_tree_py


def _w(**k):
    t = time.perf_counter()
    r = _o(**k)
    calls.append(dict(max_nodes=k.get("max_nodes"), status=r["status"], bound=r["bound"],
                      nodes=r["node_count"], undecided=r.get("n_undecided"),
                      wall=round(time.perf_counter() - t, 1)))
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
kw = dict(gdp_method=method, time_limit=tl)
if det:
    kw.update(deterministic=True, max_nodes=max_nodes)
res = m.solve(**kw)
print(f"METHOD={method} FLAG={flag} DET={det} MAX_NODES={max_nodes} TL={tl} status={res.status} "
      f"obj={res.objective} bound={res.bound!r} nodes={res.node_count} "
      f"gap_certified={getattr(res, 'gap_certified', None)} wall={time.perf_counter() - t0:.1f}s", flush=True)
for c in calls:
    print("  kernel:", c, flush=True)
assert calls, "the kernel never ran"; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
