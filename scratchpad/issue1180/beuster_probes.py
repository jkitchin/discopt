"""Does the deterministic arm do the SAME work in both arms on beuster?

If deterministic=True renders the wall budgets inert, root OBBT runs to its
fixpoint and both arms should issue the same number of probe LPs -- in which
case the faster arm must finish sooner, and a 2x wall says the work is NOT the
same. Counts the probes rather than assuming either way.
"""
import sys, time, json
sys.path.insert(0, "discopt_benchmarks/scripts")
import issue1180_callback_ab as AB

arm = sys.argv[1]
det = sys.argv[2] == "1"
arms = AB.Arms(); arms.install(arm); arms.verify(arm)
from discopt._relax import obbt
n = {"probes": 0}
orig = obbt._PersistentProbeLP.solve
def counting(self, c, lb, ub, wb):
    n["probes"] += 1
    return orig(self, c, lb, ub, wb)
obbt._PersistentProbeLP.solve = counting
from discopt.modeling.core import from_nl
m = from_nl("python/tests/data/minlplib_nl/beuster.nl")
kw = dict(time_limit=120.0, gap_tolerance=1e-4, max_nodes=20)
if det:
    kw["deterministic"] = True
t0 = time.perf_counter()
r = m.solve(**kw)
wall = time.perf_counter() - t0
obbt._PersistentProbeLP.solve = orig
assert n["probes"] > 0, "no OBBT probe was observed -- the counter never fired"
print("RES " + json.dumps({"arm": arm, "deterministic": det, "wall_s": round(wall, 1),
                           "nodes": int(r.node_count), "probes": n["probes"],
                           "bound": r.bound, "status": str(r.status)}))
