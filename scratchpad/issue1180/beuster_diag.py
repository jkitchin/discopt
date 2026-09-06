"""Where do beuster's extra ~110 s go under the new arm? Phase-level diff."""
import json, sys, time
sys.path.insert(0, "discopt_benchmarks/scripts")
import issue1180_callback_ab as AB
from discopt import _timing

arm = sys.argv[1]
arms = AB.Arms(); arms.install(arm); arms.verify(arm)
from discopt.modeling.core import from_nl
m = from_nl("python/tests/data/minlplib_nl/beuster.nl")
before = _timing.snapshot()
t0 = time.perf_counter()
r = m.solve(time_limit=120.0, gap_tolerance=1e-4, deterministic=True, max_nodes=20)
wall = time.perf_counter() - t0
delta = _timing.since(before)
print("ARM " + json.dumps({
    "arm": arm, "wall_s": round(wall, 2), "nodes": int(r.node_count),
    "status": str(r.status), "root_time_s": r.root_time,
    "ffi": {k: round(v, 2) for k, v in delta.items()},
    "stats": {k: round(v, 3) for k, v in (r.solver_stats or {}).items()},
}, indent=1))
