"""#2142 §114 step 2 (independent of SCIP): discopt's own spatial kernel run to closure on each
captured hull box (global box := the box, no cutoff, no incumbent, deterministic, max_nodes cap).
status 'infeasible' with every node fathomed = discopt PROVES the box empty (its kernel is the
certified one); 'node_limit' = undecided. Any incumbent_x is re-verified by specio.residuals.
  python -u c9_kernel_closure.py BOXES.npz MAX_NODES [b0 b1 ...]"""
import math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specio  # noqa: E402
import discopt, discopt._rust as R  # noqa: E402

print("discopt", discopt.__file__, "rust", R.__file__)
d, kw = specio.load(sys.argv[1])
cap = int(sys.argv[2])
boxes = [int(b) for b in sys.argv[3:]] or list(range(len(d["rec_id"])))
spec = {k: v for k, v in kw.items()}
cfg = dict(max_nodes=cap, gap_tol=1e300, rel_gap_tol=1e-4, abs_gap_tol=1e-6, obj_offset=0.0,
           time_limit_s=600.0, cut_rounds_root=30, cut_rounds_node=3, int_fix_probe=True)
n = 0
stat = {}
for b in boxes:
    lo, hi = d["rec_pre_lo"][b], d["rec_pre_hi"][b]
    t0 = time.perf_counter()
    res = R.solve_spatial_tree_py(**{**spec, "global_lo": lo.copy(), "global_hi": hi.copy()}, **cfg)
    n += 1
    st = res["status"]
    wit = None
    xi = np.asarray(res.get("incumbent_x", []))
    if xi.size:
        r = specio.residuals(kw, xi)
        wit = float(r["max"])
    stat[st] = stat.get(st, 0) + 1
    print(f"box {b} id {int(d['rec_id'][b])}: {st} nodes {res['node_count']} bound {res['bound']:.6g} "
          f"incumbent {res['incumbent']} witness_resid {wit} undecided {res.get('n_undecided')} "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
print("status counts", stat)
print(f"executed kernel solves: {n}")
sys.exit(0 if n else 1)
