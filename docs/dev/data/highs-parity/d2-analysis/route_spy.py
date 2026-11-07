import json, os, sys, tempfile, time
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
os.environ["DISCOPT_LP_MILP_BACKEND"] = "rust"
import lp_milp_loader as L
from discopt import _rust
from discopt.modeling.core import from_nl
name = sys.argv[1]
inst = L.read_instance(f"/home/user/discopt/ref/miplib2017/{name}.mps.gz")
td = tempfile.mkdtemp(); nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); m = from_nl(nl)
real = _rust.solve_milp_csc_py
def spy(*a, **kw):
    t = time.perf_counter(); out = real(*a, **kw)
    print("CALL m=%d n=%d ns=%d maxn=%s -> %s obj=%s bnd=%s nodes=%d %.2fs kw=%s" % (a[1], a[2], a[10], a[12], out[0], out[2], out[3], out[4], time.perf_counter()-t,
          {k: v for k, v in kw.items() if not hasattr(v, "shape") and k != "debug_hook"}), flush=True)
    return out
_rust.solve_milp_csc_py = spy
t = time.perf_counter(); r = m.solve(time_limit=20.0, gap_tolerance=1e-4)
print("RESULT", r.status, r.objective, r.bound, r.node_count, "%.2fs" % (time.perf_counter()-t), r.algorithm_route)
print("inst m=%d n=%d" % (inst.m, inst.n))
# --- replay the route's FIRST driver call with the direct panel's options
import numpy as np
cap = {}
def grab(*a, **kw):
    if "a" not in cap: cap["a"], cap["kw"] = a, kw
    return real(*a, **kw)
_rust.solve_milp_csc_py = grab
m2 = from_nl(nl); m2.solve(time_limit=3.0, gap_tolerance=1e-4)
a, kw = cap["a"], cap["kw"]
for label, opts in (("route-opts", kw), ("direct-opts", dict(time_limit_s=20.0, root_cuts=500, cut_rounds=50, cut_select=True, gmi_cuts=True, root_cut_prune=True, root_cut_time_s=10.0))):
    o = dict(opts); o["time_limit_s"] = 20.0
    t = time.perf_counter(); out = real(*a[:12], 10_000_000, 1e-4, **{k: v for k, v in o.items() if k not in ("debug_hook",)})
    print("REPLAY", label, out[0], out[2], out[3], out[4], "%.2fs" % (time.perf_counter() - t))
lb, ub = np.asarray(a[7]), np.asarray(a[8]); ns = a[10]
print("route bounds: ub>=1e19 struct cols", int((ub[:ns] >= 1e19).sum()), "lb<=-1e19", int((lb[:ns] <= -1e19).sum()), "c nnz", int((np.asarray(a[0]) != 0).sum()), "obj_const", a[11])
