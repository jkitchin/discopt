"""#2142 item 3, attribution after the tangent kill: SCIP's fixed-box edge vanishes when
its root propagation is off (t2 arms). Is it SCIP's propagation of the ALREADY-propagated
box, fed to the same LP?

Per captured box (post-kernel-propagation box K, the box our node LP was solved on):
  S'   = SCIP root propagation of K (scip_prop BASE_PARAMS: no LP, no cutoff) -- the
         p2 harness, started from K instead of the pre-propagation box;
  K2   = the kernel's own fixpoint re-run on K (propagate_spec_py, 15 rounds, no cutoff);
  ours(K), ours(S' ∩ K), ours(K2): the kernel's node LP bound on each box;
  scip = SCIP root dual bound on K (LP + enforcement, separators off; the p5 arm).
Prints comparison counts (§6); nothing caught (§7).
"""
import math, os, sys
import numpy as np
import discopt._rust as R

assert R.__file__.startswith(os.environ["DISCOPT_TREE"]), R.__file__
assert hasattr(R, "solve_spatial_node_py")
sys.path.insert(0, os.environ["PROPDIR"])
import scip_prop  # noqa: E402

d = dict(np.load(sys.argv[1]))
SPEC = [k for k in d if not k.startswith("rec_")]
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC}
n = int(d["n_cols"])
sprop = scip_prop.ScipProp(kw, {"numerics/boundstreps": 1e-9})
slp = scip_prop.ScipProp(kw, {"lp/solvefreq": 1, "numerics/boundstreps": 1e-9})
E = np.zeros(0)
EI = np.zeros(0, dtype=np.int64)


def rec(k, b):
    p = d["rec_%s_ptr" % k]
    return d["rec_%s_cat" % k][p[b]:p[b + 1]]


def ours(lo, hi):
    a = dict(kw, lo=np.ascontiguousarray(lo), hi=np.ascontiguousarray(hi), cut_ptr=np.zeros(1, dtype=np.int64),
             cut_cols=EI, cut_coeffs=E, cut_rhs=E)
    r = R.solve_spatial_node_py(**a)
    if not (r["status"] == "infeasible" or (r["status"] == "optimal" and math.isfinite(r["bound"]))):
        r = R.solve_spatial_node_py(**a, cold_dual_start=True)
    return math.inf if r["status"] == "infeasible" else (r["bound"] if r["status"] == "optimal" else -math.inf)


def tighter(a_lo, a_hi, b_lo, b_hi):
    """columns where box a is tighter than b by > 1e-6 relative to b's width"""
    w = np.maximum(b_hi - b_lo, 1e-12)
    return int(np.sum(((a_lo - b_lo) > 1e-6 * np.maximum(w, 1)) | ((b_hi - a_hi) > 1e-6 * np.maximum(w, 1))))


rows = []
n_cmp = 0
for b in range(len(d["rec_id"])):
    if not (math.isfinite(d["rec_bound"][b]) and d["rec_status"][b] == "optimal"):
        continue
    K_lo, K_hi = rec("lo", b), rec("hi", b)
    st, S_lo, S_hi = sprop.propagate(K_lo, K_hi, None)
    if st == "infeasible":
        bS, ncolS = math.inf, -1
    else:
        S_lo = np.maximum(S_lo, K_lo)
        S_hi = np.minimum(S_hi, K_hi)
        ncolS = tighter(S_lo, S_hi, K_lo, K_hi)
        bS = ours(np.where(np.isfinite(S_lo), S_lo, -1e20), np.where(np.isfinite(S_hi), S_hi, 1e20))
    live, K2_lo, K2_hi = R.propagate_spec_py(**kw, lo=K_lo, hi=K_hi, cutoff=None, rounds=15)
    ncolK2 = tighter(np.asarray(K2_lo), np.asarray(K2_hi), K_lo, K_hi)
    bK = ours(K_lo, K_hi)
    slp.m.freeTransform()
    sst, _, _ = slp.propagate(K_lo, K_hi, None)
    sc = slp.m.getDualbound() if sst != "infeasible" else math.inf
    n_cmp += 1
    rows.append((int(d["rec_id"][b]), bK, bS, sc, ncolS, ncolK2, live))

ids, bK, bS, sc, ncS, ncK2, live = (np.array(c, dtype=float) for c in zip(*rows))
print(f"boxes {len(rows)}; SCIP propagation of K proves empty: {int((ncS < 0).sum())}; "
      f"tightens >=1 column of K: {int((ncS > 0).sum())} (columns total {int(ncS[ncS > 0].sum())})")
print(f"kernel fixpoint re-run on K tightens >=1 column: {int((ncK2 > 0).sum())}; proves empty: {int((live == 0).sum())}")
sel = np.isfinite(sc) & (sc - bK > 1e-4)
print(f"SCIP(LP) tighter than ours(K) by >1e-4: {int(sel.sum())}; SCIP infeasible: {int((~np.isfinite(sc)).sum())}")
capped = np.minimum(bS, sc)
num, den = float(np.sum(capped[sel] - bK[sel])), float(np.sum(sc[sel] - bK[sel]))
print(f"our LP on S'∩K closes {num:.5f} / {den:.5f} = {num / den:.1%} of the SCIP-vs-ours gap on those boxes")
print(f"  after: SCIP still tighter by >1e-4 on {int((sc[sel] - bS[sel] > 1e-4).sum())}; ours(S') tighter than SCIP on "
      f"{int((np.isfinite(sc) & (bS - sc > 1e-4)).sum())} boxes overall")
for q in sorted(range(len(rows)), key=lambda q: -(sc[q] - bK[q]) if np.isfinite(sc[q]) else 0)[:10]:
    print(f"  node {int(ids[q]):6d} ours(K) {bK[q]:.4f} ours(S') {bS[q]:.4f} scip {sc[q]:.4f} S'-cols {int(ncS[q])}")
print(f"executed comparisons: {n_cmp}")
sys.exit(0 if n_cmp else 1)
