"""#2142 item 3 -- on each captured cstr big-M node box (pre-propagation box + cutoff),
compare the kernel's fixpoint (propagate_spec_py) against (a) the numpy reference FBBT
run to convergence and (b) SCIP root propagation. Prints comparison counts (§6).

  PYTHONPATH=<wt>/python python -u p2_compare.py BOXES.npz [--no-scip] [--scip-param k=v ...]"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WT = os.environ["DISCOPT_TREE"]  # the tree under test (CLAUDE.md §8)
import discopt._rust as R  # noqa: E402

assert R.__file__.startswith(WT) and hasattr(R, "propagate_spec_py"), R.__file__
import fbbt_ref  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("boxes")
ap.add_argument("--no-scip", action="store_true")
ap.add_argument("--scip-param", nargs="*", default=[])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--dump", default="")
args = ap.parse_args()
d = dict(np.load(args.boxes))
SPEC_KEYS = ("n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
             "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
             "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
             "obbt_candidates").split()
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC_KEYS}
rounds = int(d["prop_rounds"])
n = kw["n_cols"]; n_orig = kw["n_orig"]
spec = fbbt_ref.Spec(kw)
scip_p = None
if not args.no_scip:
    import scip_prop
    params = {}
    for kv in args.scip_param:
        k, v = kv.split("=", 1)
        vv = {"true": True, "false": False}.get(v.lower())
        params[k] = vv if vv is not None else (float(v) if "." in v else int(v))
    scip_p = scip_prop.ScipProp(kw, params)


def kern(lo, hi, cut, r):
    return R.propagate_spec_py(**kw, lo=np.ascontiguousarray(lo), hi=np.ascontiguousarray(hi), cutoff=cut, rounds=r)


TOL = 1e-6


def looser(Alo, Ahi, Blo, Bhi):
    """Per-column mask: box A is looser than box B by more than TOL (relative)."""
    lo_l = Alo < Blo - TOL * (1 + np.abs(Blo))
    hi_l = Ahi > Bhi + TOL * (1 + np.abs(Bhi))
    lo_l &= np.isfinite(Blo); hi_l &= np.isfinite(Bhi)
    return lo_l | hi_l


def width(lo, hi):
    return np.where(np.isfinite(hi - lo), hi - lo, np.inf)


nb = len(d["rec_id"])
if args.limit:
    nb = min(nb, args.limit)
cnt = dict(boxes=0, replay_identical=0, replay_mismatch=0, kernel_empty=0, ref_empty=0, scip_empty=0,
           k15_vs_kinf_looser_boxes=0, k_vs_ref_looser_boxes=0, k_vs_ref_looser_cols=0,
           ref_vs_k_looser_boxes=0, ref_vs_k_looser_cols=0, k_vs_scip_looser_boxes=0, k_vs_scip_looser_cols=0,
           scip_vs_k_looser_boxes=0, scip_vs_k_looser_cols=0, ref_vs_scip_looser_boxes=0, ref_vs_scip_looser_cols=0,
           col_comparisons=0, empty_disagree=0)
col_k_ref = np.zeros(n, int); col_k_scip = np.zeros(n, int); col_scip_k = np.zeros(n, int)
ratios_ref, ratios_scip = [], []
ref_rounds = []
dump = []
t0 = time.perf_counter()
for b in range(nb):
    plo, phi, cut = d["rec_pre_lo"][b], d["rec_pre_hi"][b], float(d["rec_cutoff"][b])
    cut = cut if math.isfinite(cut) else None
    live15, klo, khi = kern(plo, phi, cut, rounds)
    cnt["boxes"] += 1
    if live15 and np.array_equal(klo, d["rec_lo"][b]) and np.array_equal(khi, d["rec_hi"][b]):
        cnt["replay_identical"] += 1
    else:
        cnt["replay_mismatch"] += 1
    liveI, kilo, kihi = kern(plo, phi, cut, 100000)
    if live15 and liveI and looser(klo, khi, kilo, kihi).any():
        cnt["k15_vs_kinf_looser_boxes"] += 1
    rlive, rlo, rhi, rr = fbbt_ref.fbbt(spec, plo, phi, cut)
    ref_rounds.append(rr)
    rec = {"id": int(d["rec_id"][b]), "k_live": bool(liveI), "ref_live": bool(rlive)}
    if not liveI:
        cnt["kernel_empty"] += 1
    if not rlive:
        cnt["ref_empty"] += 1
    if liveI and rlive:
        kl = np.where(kilo <= -1e19, -np.inf, kilo); kh = np.where(kihi >= 1e19, np.inf, kihi)
        m1 = looser(kl, kh, rlo, rhi); m2 = looser(rlo, rhi, kl, kh)
        cnt["col_comparisons"] += n
        cnt["k_vs_ref_looser_cols"] += int(m1.sum()); cnt["ref_vs_k_looser_cols"] += int(m2.sum())
        cnt["k_vs_ref_looser_boxes"] += int(m1.any()); cnt["ref_vs_k_looser_boxes"] += int(m2.any())
        col_k_ref += m1
        for j in np.flatnonzero(m1):
            wk, wr = width(kl, kh)[j], width(rlo, rhi)[j]
            ratios_ref.append(wr / wk if wk > 0 and np.isfinite(wk) else np.nan)
        rec["k_ref_cols"] = [(int(j), float(kl[j]), float(kh[j]), float(rlo[j]), float(rhi[j])) for j in np.flatnonzero(m1)]
    if scip_p is not None:
        st, slo, shi = scip_p.propagate(plo, phi, cut)
        rec["scip_status"] = st
        if slo is None:
            cnt["scip_empty"] += 1
        if (slo is None) != (not liveI):
            cnt["empty_disagree"] += 1
        if liveI and slo is not None:
            kl = np.where(kilo <= -1e19, -np.inf, kilo); kh = np.where(kihi >= 1e19, np.inf, kihi)
            m1 = looser(kl, kh, slo, shi); m2 = looser(slo, shi, kl, kh)
            cnt["k_vs_scip_looser_cols"] += int(m1.sum()); cnt["scip_vs_k_looser_cols"] += int(m2.sum())
            cnt["k_vs_scip_looser_boxes"] += int(m1.any()); cnt["scip_vs_k_looser_boxes"] += int(m2.any())
            col_k_scip += m1; col_scip_k += m2
            for j in np.flatnonzero(m1):
                wk, ws = width(kl, kh)[j], width(slo, shi)[j]
                ratios_scip.append(ws / wk if wk > 0 and np.isfinite(wk) else np.nan)
            rec["k_scip_cols"] = [(int(j), float(kl[j]), float(kh[j]), float(slo[j]), float(shi[j])) for j in np.flatnonzero(m1)]
            rec["scip_k_cols"] = [(int(j), float(kl[j]), float(kh[j]), float(slo[j]), float(shi[j])) for j in np.flatnonzero(m2)]
            if rlive:
                m3 = looser(rlo, rhi, slo, shi)
                cnt["ref_vs_scip_looser_cols"] += int(m3.sum()); cnt["ref_vs_scip_looser_boxes"] += int(m3.any())
    dump.append(rec)
    if (b + 1) % 25 == 0:
        print(f"[{b+1}/{nb}] {time.perf_counter()-t0:.1f}s {json.dumps(cnt)}", flush=True)
print("FINAL", json.dumps(cnt))
print("ref rounds: median", float(np.median(ref_rounds)), "max", int(max(ref_rounds)))
if ratios_ref:
    a = np.array(ratios_ref); print("kernel-vs-ref looser cols: width ratio ref/kernel median", np.nanmedian(a), "min", np.nanmin(a))
if ratios_scip:
    a = np.array(ratios_scip); print("kernel-vs-scip looser cols: width ratio scip/kernel median", np.nanmedian(a), "min", np.nanmin(a))
cls = lambda j: "aux" if j >= n_orig else ("bin" if kw["integrality"][j] else "cont")
print("cols where kernel looser than ref (col, count, class):", [(int(j), int(col_k_ref[j]), cls(j)) for j in np.argsort(-col_k_ref)[:15] if col_k_ref[j]])
print("cols where kernel looser than SCIP:", [(int(j), int(col_k_scip[j]), cls(j)) for j in np.argsort(-col_k_scip)[:15] if col_k_scip[j]])
print("cols where SCIP looser than kernel:", [(int(j), int(col_scip_k[j]), cls(j)) for j in np.argsort(-col_scip_k)[:15] if col_scip_k[j]])
if args.dump:
    json.dump(dump, open(args.dump, "w"))
n_cmp = cnt["boxes"]
print(f"executed comparisons: boxes={n_cmp} columns={cnt['col_comparisons']}")
sys.exit(0 if n_cmp > 0 and cnt["col_comparisons"] > 0 else 1)
