"""One deterministic solve, DISCOPT_SPATIAL_BRANCH_SCIP as set in the env (#2142, perf-plan §110).
  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python DISCOPT_PROFILE=1 python -u k_one.py FILE MAX_NODES GAP_TOL|-"""
import os, sys, time
import discopt, discopt.modeling as dm
from discopt._rust import profile_counters_py, profile_reset_py
assert discopt.__file__.startswith(os.environ["DISCOPT_TREE"]), discopt.__file__
from discopt.solver import _spatial_branch_scip_kwargs  # marker absent on the base tree (§8)
kw = _spatial_branch_scip_kwargs()
f = sys.argv[1]
if f.endswith(".gms"):
    from discopt.interfaces.gams import from_gams
    m = from_gams(f)
else:
    m = dm.from_nl(f)
gt = {} if sys.argv[3] == "-" else {"gap_tolerance": float(sys.argv[3])}
profile_reset_py()
t = time.perf_counter()
r = m.solve(time_limit=float(os.environ.get("TL", 3600)), deterministic=True, max_nodes=int(sys.argv[2]), **gt)
c = dict(profile_counters_py())
n_assert = 0
if kw:
    assert c["SpatialViolScipPicks"] > 0, "flag on but the scored rule never ran"
    assert c["SpatialSplitMoved"] > 0, "flag on but the point rule never moved a split"
    n_assert += 2
else:
    assert c["SpatialViolScipPicks"] == 0 and c["SpatialSplitMoved"] == 0
    n_assert += 1
print(f"{os.path.basename(f)} flag={os.environ.get('DISCOPT_SPATIAL_BRANCH_SCIP','0')} status={r.status} "
      f"obj={r.objective} bound={r.bound!r} certified={r.gap_certified} nodes={r.node_count} "
      f"wall={time.perf_counter()-t:.1f}s picks={c['SpatialViolScipPicks']} changed={c['SpatialViolScipChanged']} "
      f"splits={c['SpatialSplits']} moved={c['SpatialSplitMoved']}", flush=True)
print(f"executed assertions: {n_assert}")
if n_assert == 0:
    sys.exit(1)
