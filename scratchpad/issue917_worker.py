"""#917 entry experiment worker: one instance, one isolated process.

Replicates ``Model.solve``'s #844 budget split explicitly so the two halves can be
measured separately:

  stage A (primary):  ``solve_model(time_limit=0.65*T)``  -- what ships today
  stage B (reserve):  ``solve_lp_spatial_bb(time_limit=0.35*T)`` -- today spent ONLY
                      when stage A found no incumbent; #917 asks whether spending it
                      when stage A DID find one buys a tighter dual bound.

Usage: python issue917_worker.py <path.nl> <time_limit>
Prints one JSON object on stdout. Any exception propagates (CLAUDE.md §7).
"""

import json
import sys
import time

from discopt._jax.deadline import deadline_scope
from discopt._jax.lp_spatial_bb import solve_lp_spatial_bb
from discopt.modeling.core import ObjectiveSense, from_nl
from discopt.solver import solve_model

path = sys.argv[1]
T = float(sys.argv[2])
primary_tl = 0.65 * T
reserve = 0.35 * T

m = from_nl(path)
sense = "max" if m._objective.sense == ObjectiveSense.MAXIMIZE else "min"

t0 = time.perf_counter()
with deadline_scope(primary_tl):
    res = solve_model(m, time_limit=primary_tl)
t_primary = time.perf_counter() - t0
try:
    del m._solve_deadline
except AttributeError:
    pass

out = {
    "instance": path.split("/")[-1].removesuffix(".nl"),
    "sense": sense,
    "time_limit": T,
    "primary_tl": primary_tl,
    "primary_wall": t_primary,
    "primary_status": res.status,
    "primary_objective": res.objective,
    "primary_bound": res.bound,
    "primary_gap_certified": bool(getattr(res, "gap_certified", False)),
    "primary_nodes": res.node_count,
}

# Stage B on the residual, unconditionally -- the whole point of the probe is the
# case today's code skips (stage A returned an incumbent).
t1 = time.perf_counter()
with deadline_scope(reserve):
    fb = solve_lp_spatial_bb(
        m, time_limit=reserve, use_obbt=False, require_incremental=True
    )
t_reserve = time.perf_counter() - t1

out["reserve_wall"] = t_reserve
out["reserve_ran"] = fb is not None
out["reserve_status"] = None if fb is None else fb.status
out["reserve_objective"] = None if fb is None else fb.objective
out["reserve_bound"] = None if fb is None else fb.bound
out["reserve_nodes"] = None if fb is None else fb.node_count

print(json.dumps(out))
