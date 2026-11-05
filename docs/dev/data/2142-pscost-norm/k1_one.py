"""One deterministic solve with DISCOPT_SPATIAL_SCIP_SCORE as set in the env (#2142, perf-plan §109).
  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python DISCOPT_PROFILE=1 python -u k1_one.py FILE.nl MAX_NODES GAP_TOL"""
import os, sys, time
import discopt, discopt.modeling as dm
from discopt._rust import profile_counters_py, profile_reset_py
assert discopt.__file__.startswith(os.environ["DISCOPT_TREE"]), discopt.__file__
from discopt.solver import _spatial_viol_scip_kwargs  # marker absent on the base tree (§8)
kw = _spatial_viol_scip_kwargs()
m = dm.from_nl(sys.argv[1])
gt = {} if sys.argv[3] == "-" else {"gap_tolerance": float(sys.argv[3])}
profile_reset_py()
t = time.perf_counter()
r = m.solve(time_limit=3600, deterministic=True, max_nodes=int(sys.argv[2]), **gt)
c = dict(profile_counters_py())
mode = os.environ.get("DISCOPT_SPATIAL_SCIP_SCORE", "0")
n_assert = 1
if kw:
    assert c["SpatialViolScipPicks"] > 0, "flag on but the rule never ran"
    n_assert += 1
else:
    assert c["SpatialViolScipPicks"] == 0
    n_assert += 1
print(f"{os.path.basename(sys.argv[1])} MODE={mode} kw={kw} status={r.status} obj={r.objective} "
      f"bound={r.bound!r} certified={r.gap_certified} nodes={r.node_count} wall={time.perf_counter()-t:.1f}s "
      f"picks={c['SpatialViolScipPicks']} changed={c['SpatialViolScipChanged']}", flush=True)
print(f"executed assertions: {n_assert}")
