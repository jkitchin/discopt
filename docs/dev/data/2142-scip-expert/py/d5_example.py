"""#2142 scipx d5: on discopt's own 20k-tree node boxes (LP-optimal ones, post-propagation), compare the
kernel's propagation run to convergence (100000 rounds) WITHOUT vs WITH the 95 dual-proof rows appended,
and the node LP bound on each resulting box. Prints executed-comparison count (§6).
  python -u d5_example.py SPEC.npz D1.npz PROOFS.npz STRIDE"""
import sys, math
import numpy as np
import discopt._rust as R
assert R.__file__.startswith("/home/user/discopt/python")
d = dict(np.load(sys.argv[1])); d1 = np.load(sys.argv[2]); P = np.load(sys.argv[3]); stride = int(sys.argv[4])
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in d if not k.startswith("meta_")}
kwp = dict(kw)
ptr = list(d["fixed_row_ptr"]); off = ptr[-1]
kwp["fixed_row_ptr"] = np.array(ptr + [off + x for x in P["ptr"][1:]], dtype=np.int64)
kwp["fixed_cols"] = np.r_[d["fixed_cols"], P["cols"]].astype(np.int64)
kwp["fixed_coeffs"] = np.r_[d["fixed_coeffs"], P["coefs"]]
kwp["fixed_rhs"] = np.r_[d["fixed_rhs"], P["rhs"]]
E = np.zeros(0); EI = np.zeros(0, dtype=np.int64)
def lp(lo, hi):
    a = dict(kw, lo=np.ascontiguousarray(lo), hi=np.ascontiguousarray(hi), cut_ptr=np.zeros(1, dtype=np.int64), cut_cols=EI, cut_coeffs=E, cut_rhs=E)
    r = R.solve_spatial_node_py(**a)
    if not (r["status"] == "infeasible" or (r["status"] == "optimal" and math.isfinite(r["bound"]))):
        r = R.solve_spatial_node_py(**a, cold_dual_start=True)
    return math.inf if r["status"] == "infeasible" else (r["bound"] if r["status"] == "optimal" else -math.inf)
integ = d["integrality"].astype(bool)
n_cmp = n_empty = n_tight = n_binfix = 0; gains = []; best = None
idx = np.where(d1["status"] == "optimal")[0][::stride]
for k in idx:
    cut = float(d1["cutoff"][k]); cut = cut if math.isfinite(cut) else None
    lo0, hi0 = d1["lo"][k], d1["hi"][k]
    ok0, l0, h0 = R.propagate_spec_py(**kw, lo=lo0, hi=hi0, cutoff=cut, rounds=100000)
    ok1, l1, h1 = R.propagate_spec_py(**kwp, lo=lo0, hi=hi0, cutoff=cut, rounds=100000)
    l0, h0, l1, h1 = map(np.asarray, (l0, h0, l1, h1))
    n_cmp += 1
    b0 = lp(l0, h0) if ok0 else math.inf
    if not ok1:
        n_empty += 1; b1 = math.inf
    else:
        tolv = 1e-6 * (1 + np.abs(l0) + np.abs(h0)); n_loose = int(((l1 < l0 - tolv) | (h1 > h0 + tolv)).sum()); assert n_loose == 0, ("proof rows loosened a box beyond 1e-6 rel", n_loose)
        tight = (l1 > l0 + 1e-7) | (h1 < h0 - 1e-7)
        if tight.any(): n_tight += 1
        if (integ & (h1 - l1 < 0.5) & (h0 - l0 > 0.5)).any(): n_binfix += 1
        b1 = lp(l1, h1)
    gains.append((b1, b0))
    g = (min(b1, 3.0620) - b0) if math.isfinite(b0) else 0
    if best is None or g > best[0]:
        best = (g, int(d1["id"][k]), b0, b1, ok1, [(j, l0[j], h0[j], l1[j], h1[j]) for j in range(len(l0)) if ok1 and ((l1[j] > l0[j] + 1e-7) or (h1[j] < h0[j] - 1e-7))][:12])
G = np.array(gains)
fin = np.isfinite(G[:, 1])
print(f"boxes {n_cmp}; proofs empty the converged box on {n_empty}; tighten it on {n_tight}; fix a binary on {n_binfix}")
print(f"node LP bound: raised >1e-4 on {int(((np.minimum(G[:,0],3.0620)-G[:,1])[fin] > 1e-4).sum())} of {int(fin.sum())} finite boxes; "
      f"mean gain (capped at incumbent) {np.mean((np.minimum(G[:,0],3.0620)-G[:,1])[fin]):.4f}")
print("best example: node", best[1], "bound without", best[2], "with", best[3], "nonempty", best[4])
for t in best[5]: print("   col %d: [%.5g, %.5g] -> [%.5g, %.5g]" % t)
print(f"executed comparisons: {n_cmp}")
sys.exit(0 if n_cmp else 1)
