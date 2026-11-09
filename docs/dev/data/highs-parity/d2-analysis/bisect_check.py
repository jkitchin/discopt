"""Bisect predicate: does gt2 certify within 20 s on a direct dense driver call with the
September A14 options? exit 0 = good (certifies), 1 = bad, 125 = skip (build/API failure)."""
import json, sys, time
import numpy as np
import discopt
from discopt._rust import solve_milp_csc_py
name = sys.argv[1] if len(sys.argv) > 1 else "gt2"
d = np.load(f"/tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/bisect/{name}.npz")
BASE = dict(gap_tol=1e-4, max_nodes=10_000_000, gmi_cuts=True, root_cut_prune=True,
            time_limit_s=20.0, root_cuts=500, cut_rounds=50, cut_select=True, root_cut_time_s=10.0, presolve=__import__("os").environ.get("PRESOLVE","1")=="1")
t = time.perf_counter()
st, x, obj, bnd, nodes, it = solve_milp_csc_py(d["c"], int(d["m"]), int(d["n"]), d["cp"], d["ri"],
                                               d["v"], d["b"], d["lb"], d["ub"], d["ic"],
                                               int(d["ns"]), float(d["off"]), **BASE)
print(json.dumps({"name": name, "status": st, "obj": obj, "nodes": int(nodes),
                  "wall": time.perf_counter() - t, "file": discopt.__file__}), flush=True)
sys.exit(0 if st == "optimal" else 1)
