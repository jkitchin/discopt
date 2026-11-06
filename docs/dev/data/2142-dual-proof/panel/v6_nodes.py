"""One deterministic solve of an .nl at a node budget with a generous wall (the wall must not bind),
flag from the environment; prints bound/nodes/undecided/wall and the DualProof* counters.
  DISCOPT_TREE=<repo>/python DISCOPT_PROFILE=1 python -u v6_nodes.py FILE.nl MAX_NODES"""
import os, sys, time
import discopt, discopt.modeling as dm
from discopt._rust import profile_counters_py, profile_reset_py
assert discopt.__file__.startswith(os.environ["DISCOPT_TREE"]), discopt.__file__
m = dm.from_nl(sys.argv[1]); profile_reset_py()
t = time.perf_counter()
r = m.solve(time_limit=3600, deterministic=True, max_nodes=int(sys.argv[2]))
w = time.perf_counter() - t
c = dict(profile_counters_py())
print(f"{os.path.basename(sys.argv[1])} FLAG={os.environ.get('DISCOPT_DUAL_PROOF_PROPAGATION')} status={r.status} "
      f"obj={r.objective} bound={r.bound!r} nodes={r.node_count} wall={w:.1f}s "
      f"{ {k: int(v) for k, v in c.items() if k.startswith('DualProof') or k in ('LpVerdictNumerical',)} }", flush=True)
assert w < 3500, "the wall bound the run"
print("executed assertions: 1")
