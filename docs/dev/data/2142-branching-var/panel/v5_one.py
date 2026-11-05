"""One deterministic solve of an .nl with DISCOPT_SPATIAL_VIOL_SCIP as set in the env.
  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u s4_one.py FILE.nl MAX_NODES"""
import os, sys, time
import discopt, discopt.modeling as dm
assert discopt.__file__.startswith(os.environ["DISCOPT_TREE"]), discopt.__file__
t = time.perf_counter()
r = dm.from_nl(sys.argv[1]).solve(time_limit=3600, deterministic=True, max_nodes=int(sys.argv[2]), gap_tolerance=1e-4)
print(f"{os.path.basename(sys.argv[1])} FLAG={os.environ.get('DISCOPT_SPATIAL_VIOL_SCIP')} status={r.status} obj={r.objective} "
      f"bound={r.bound!r} certified={r.gap_certified} nodes={r.node_count} wall={time.perf_counter()-t:.1f}s", flush=True)
print("executed assertions: 1")
