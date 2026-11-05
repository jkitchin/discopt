"""#2142: is the tree's post-propagation box a fixpoint? Re-run the kernel's own
propagate_spec_py on it (no cutoff / the node's cutoff; 15 / 1000 rounds) and solve the
kernel LP on the result. Prints comparison counts (§6)."""
import math, os, sys
import numpy as np
import discopt._rust as R
assert R.__file__.startswith(os.environ["DISCOPT_TREE"]) and hasattr(R, "solve_spatial_node_py")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
d = dict(np.load(sys.argv[1]))
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in d if not k.startswith("rec_")}
pre = dict(np.load(sys.argv[2]))  # the propagation capture: rec_pre_lo/hi, rec_cutoff
t2 = dict(np.load(sys.argv[3]))   # t2 default npz: scip per box (ids, sc)
scip = dict(zip(t2["ids"].astype(int), t2["sc"]))
E = np.zeros(0); EI = np.zeros(0, dtype=np.int64)
def rec(k, b):
    p = d["rec_%s_ptr" % k]; return d["rec_%s_cat" % k][p[b]:p[b + 1]]
def ours(lo, hi):
    a = dict(kw, lo=np.ascontiguousarray(lo), hi=np.ascontiguousarray(hi), cut_ptr=np.zeros(1, dtype=np.int64), cut_cols=EI, cut_coeffs=E, cut_rhs=E)
    r = R.solve_spatial_node_py(**a)
    if not (r["status"] == "infeasible" or (r["status"] == "optimal" and math.isfinite(r["bound"]))):
        r = R.solve_spatial_node_py(**a, cold_dual_start=True)
    return math.inf if r["status"] == "infeasible" else (r["bound"] if r["status"] == "optimal" else -math.inf)
rows = []; n_cmp = 0
for b in range(len(d["rec_id"])):
    if not (math.isfinite(d["rec_bound"][b]) and d["rec_status"][b] == "optimal"):
        continue
    K_lo, K_hi = rec("lo", b), rec("hi", b)
    cut = float(pre["rec_cutoff"][b]); cut = cut if math.isfinite(cut) else None
    # sanity: the tree box IS propagate(pre, cutoff, 15)
    lv, a_lo, a_hi = R.propagate_spec_py(**kw, lo=pre["rec_pre_lo"][b], hi=pre["rec_pre_hi"][b], cutoff=cut, rounds=15)
    same = np.array_equal(np.asarray(a_lo), K_lo) and np.array_equal(np.asarray(a_hi), K_hi)
    out = {}
    for tag, (lo0, hi0, c, rnd) in {"K+15": (K_lo, K_hi, None, 15), "K+15cut": (K_lo, K_hi, cut, 15),
                                    "pre+1000cut": (pre["rec_pre_lo"][b], pre["rec_pre_hi"][b], cut, 1000)}.items():
        lv, lo1, hi1 = R.propagate_spec_py(**kw, lo=lo0, hi=hi0, cutoff=c, rounds=rnd)
        lo1, hi1 = np.asarray(lo1), np.asarray(hi1)
        rel = np.max(np.where(K_hi - K_lo > 0, ((lo1 - K_lo) + (K_hi - hi1)) / np.maximum(K_hi - K_lo, 1e-300), 0))
        out[tag] = (ours(lo1, hi1) if lv else math.inf, rel)
    n_cmp += 1
    rows.append((int(d["rec_id"][b]), same, ours(K_lo, K_hi), out, scip[int(d["rec_id"][b])]))
print("tree box == propagate(pre, cutoff, 15):", sum(r[1] for r in rows), "/", len(rows))
bK = np.array([r[2] for r in rows]); sc = np.array([r[4] for r in rows])
sel = np.isfinite(sc) & (sc - bK > 1e-4)
for tag in ("K+15", "K+15cut", "pre+1000cut"):
    bt = np.array([r[3][tag][0] for r in rows]); rl = np.array([r[3][tag][1] for r in rows])
    cap = np.minimum(bt, sc)
    print(f"{tag:12s}: boxes shrunk {int((rl > 1e-6).sum())}, max relative width shrink {rl.max():.3f}; bound gain >1e-4 on {int((bt - bK > 1e-4).sum())}; "
          f"closes {np.sum(cap[sel]-bK[sel]):.4f}/{np.sum(sc[sel]-bK[sel]):.4f} = {np.sum(cap[sel]-bK[sel])/np.sum(sc[sel]-bK[sel]):.1%} of the SCIP gap on {int(sel.sum())} boxes")
print("executed comparisons:", n_cmp); sys.exit(0 if n_cmp else 1)
