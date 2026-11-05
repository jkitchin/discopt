"""#2142 §114 diagnosis after the E2 kill: what holds cstr hull's bound at 0.094?
cstr hull, deterministic, MAX_NODES; records every observed node (id, parent, bound, status)
via the kernel's node_observer and the profile counters (DISCOPT_PROFILE=1). Prints the status
histogram, how many nodes carry bound <= 0.0941 per status, and IntRoundExactSnaps.
  DISCOPT_PROFILE=1 DISCOPT_INT_ROUND_EXACT=0|1 python -u d1_pin.py MAX_NODES"""
import collections, os, sys, time
import discopt, discopt._rust as R

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert b"IntRoundExactSnaps" in open(R.__file__, "rb").read(), "stale _rust"
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

recs = []
_o = R.solve_spatial_tree_py


def obs(rec):
    recs.append((rec["id"], rec.get("parent"), rec.get("bound"), str(rec.get("status"))))


def _w(**k):
    if int(k.get("max_nodes", 0)) > 1:
        k["node_observer"] = obs
    return _o(**k)


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="hull", time_limit=7200, deterministic=True, max_nodes=int(sys.argv[1]))
print("FLAG", os.environ.get("DISCOPT_INT_ROUND_EXACT"), "status", res.status, "bound", res.bound,
      "nodes", res.node_count, f"wall {time.perf_counter() - t0:.0f}s", flush=True)
assert recs, "observer never fired"
st = collections.Counter(r[3] for r in recs)
low = collections.Counter(r[3] for r in recs if r[2] is not None and r[2] <= 0.0941)
print("observed", len(recs), "status histogram", dict(st))
print("nodes with bound <= 0.0941 by status", dict(low))
ctr = dict(R.profile_counters_py())
print("counters", {k: v for k, v in ctr.items() if k in (
    "IntRoundExactSnaps", "LpVerdictNumerical", "LpInfeasUncertified", "LpVerdictInfeasible",
    "LpVerdictOptimal", "FarkasRejectOpen", "FarkasRejectMargin")})
print("executed assertions: 2")
