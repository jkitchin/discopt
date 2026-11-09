"""Per-node cost probe: Model.solve on the Rust MILP route, fixed node cap. Prints one JSON
line per instance and asserts which discopt was loaded (CLAUDE.md section 8)."""
import json, os, sys, tempfile, time
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import lp_milp_loader as L
import discopt
from discopt.modeling.core import from_nl
want = sys.argv[1]
assert discopt.__file__.startswith(want), (discopt.__file__, want)
os.environ["DISCOPT_LP_MILP_BACKEND"] = "rust"
maxn = int(sys.argv[2])
for name in sys.argv[3].split(","):
    inst = L.read_instance(f"/home/user/discopt/ref/miplib2017/{name}.mps.gz")
    td = tempfile.mkdtemp(); nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl)
    m = from_nl(nl)
    t = time.perf_counter()
    r = m.solve(time_limit=float(sys.argv[4]), max_nodes=maxn)
    w = time.perf_counter() - t
    print(json.dumps({"name": name, "status": str(r.status), "nodes": int(r.node_count or 0),
                      "wall": w, "nps": (r.node_count or 0) / w, "obj": r.objective, "bound": r.bound,
                      "route": (r.algorithm_route or "")[:60], "file": discopt.__file__}), flush=True)
