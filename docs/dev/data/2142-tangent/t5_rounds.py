"""#2142: fixed-box sweep of the kernel's propagation round cap. For each captured node,
propagate_spec_py(pre box, node cutoff, R) then the kernel node LP on that box; closure of
the SCIP-vs-ours fixed-box gap (t2 npz) and propagation wall time per R. Also the number of
rounds the fixpoint actually needs (smallest R whose box equals R=100000's). Prints counts (§6)."""
import math, os, sys, time
import numpy as np
import discopt._rust as R
assert R.__file__.startswith(os.environ["DISCOPT_TREE"]) and hasattr(R, "solve_spatial_node_py")
d = dict(np.load(sys.argv[1])); pre = dict(np.load(sys.argv[2])); t2 = dict(np.load(sys.argv[3]))
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in d if not k.startswith("rec_")}
scip = dict(zip(t2["ids"].astype(int), t2["sc"]))
E = np.zeros(0); EI = np.zeros(0, dtype=np.int64)
def ours(lo, hi):
    a = dict(kw, lo=np.ascontiguousarray(lo), hi=np.ascontiguousarray(hi), cut_ptr=np.zeros(1, dtype=np.int64), cut_cols=EI, cut_coeffs=E, cut_rhs=E)
    r = R.solve_spatial_node_py(**a)
    if not (r["status"] == "infeasible" or (r["status"] == "optimal" and math.isfinite(r["bound"]))):
        r = R.solve_spatial_node_py(**a, cold_dual_start=True)
    return math.inf if r["status"] == "infeasible" else (r["bound"] if r["status"] == "optimal" else -math.inf)
RS = [15, 30, 50, 100, 200, 500, 1000, 5000, 100000]
res = {r: [] for r in RS}; tm = {r: 0.0 for r in RS}; need = []; n_cmp = 0
for b in range(len(d["rec_id"])):
    if not (math.isfinite(d["rec_bound"][b]) and d["rec_status"][b] == "optimal"):
        continue
    cut = float(pre["rec_cutoff"][b]); cut = cut if math.isfinite(cut) else None
    boxes = {}
    for r in RS:
        t0 = time.perf_counter()
        lv, lo, hi = R.propagate_spec_py(**kw, lo=pre["rec_pre_lo"][b], hi=pre["rec_pre_hi"][b], cutoff=cut, rounds=r)
        tm[r] += time.perf_counter() - t0
        boxes[r] = (lv, np.asarray(lo), np.asarray(hi))
        res[r].append(ours(lo, hi) if lv else math.inf)
    fl, flo, fhi = boxes[100000]
    need.append(next(r for r in RS if boxes[r][0] == fl and np.array_equal(boxes[r][1], flo) and np.array_equal(boxes[r][2], fhi)))
    n_cmp += 1
ids = [int(i) for i, s_ in zip(d["rec_id"], d["rec_status"]) if True]
sc = np.array([scip[int(d["rec_id"][b])] for b in range(len(d["rec_id"])) if math.isfinite(d["rec_bound"][b]) and d["rec_status"][b] == "optimal"])
b15 = np.array(res[15]); sel = np.isfinite(sc) & (sc - b15 > 1e-4)
print(f"boxes {n_cmp}; SCIP-tighter (vs R=15) {int(sel.sum())}; rounds to reach the R=1e5 box: " + str({r: need.count(r) for r in RS}))
for r in RS:
    br = np.array(res[r]); cap = np.minimum(br, sc)
    print(f"R={r:6d}: gap closed {np.sum(cap[sel]-b15[sel]):.4f}/{np.sum(sc[sel]-b15[sel]):.4f} = {np.sum(cap[sel]-b15[sel])/np.sum(sc[sel]-b15[sel]):6.1%}; "
          f"bound > R=15 by >1e-4 on {int((br - b15 > 1e-4).sum())}; empty {int(np.isinf(br).sum())}; prop wall {tm[r]*1e3/n_cmp:.3f} ms/box")
print("executed comparisons:", n_cmp * len(RS)); sys.exit(0 if n_cmp else 1)
