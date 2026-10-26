"""#1975 class census: discopt's 30 s result on one instance, one JSON line."""
import json, sys, time
import discopt.modeling as dm
name, path = sys.argv[1], sys.argv[2]
t = time.time()
try:
    m = dm.from_nl(path)
    r = m.solve(time_limit=30)
    out = {"name": name, "status": r.status, "obj": r.objective, "bound": r.bound,
           "gap_certified": bool(getattr(r, "gap_certified", False)), "nodes": r.node_count,
           "root_bound": r.root_bound, "wall": time.time() - t}
except Exception as e:
    out = {"name": name, "error": f"{type(e).__name__}: {e}"[:300], "wall": time.time() - t}
print(json.dumps(out), flush=True)
