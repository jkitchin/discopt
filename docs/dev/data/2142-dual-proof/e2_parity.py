"""#2142: cstr through Model.solve at the gdp_parity budget (time_limit, NOT deterministic, no node cap),
flag from the environment. Prints status/bound/wall/gap_certified and the kernel's dual-proof counters.
  DISCOPT_DUAL_PROOF_PROPAGATION=0|1 python -u e2_parity.py big-m|hull TIME_LIMIT"""

import sys, time
import discopt, discopt._rust as R

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert "dual_proof_propagation" in (R.solve_spatial_tree_py.__text_signature__ or ""), "stale _rust"
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr

method, tl = sys.argv[1], float(sys.argv[2])
calls = []
_o = R.solve_spatial_tree_py


def _w(**k):
    r = _o(**k)
    calls.append(
        (k.get("max_nodes"), r["status"], r["bound"], r["node_count"], dict(r["dual_proof"]))
    )
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method=method, time_limit=tl)
print(
    f"METHOD={method} TL={tl} status={res.status} obj={res.objective} bound={res.bound!r} nodes={res.node_count} "
    f"gap_certified={getattr(res, 'gap_certified', None)} wall={time.perf_counter() - t0:.1f}s",
    flush=True,
)
for c in calls:
    print("  kernel:", c)
n = int(bool(calls)) + int(res.objective is not None)
print(f"executed assertions: {n}")
sys.exit(0 if n else 1)
