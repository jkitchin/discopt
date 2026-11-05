"""#2142 branching variable, entry experiment: offline candidate scoring rules vs SCIP's
replayed decisions on the 195 boxes of s3_noobbt.json.

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u v1_rules.py CAP.npz S3.json [OUT.json] [--scip-x]

With --scip-x (S3.json from v2_scip_lp.py) the rules are scored at SCIP's own last root LP
point instead of the kernel's, which separates rule differences from LP-point differences.

For each box SCIP branched on, the kernel's own node LP is re-solved cold on the box
(`solve_spatial_node_py`, the tree's node solve) and each candidate rule is scored at
that LP point. A rule "agrees" when its top column equals SCIP's. Because SCIP picks
uniformly at random among candidates scoring >= highscorefactor (0.9) x the best
(cons_nonlinear.c selectBranchingCandidate), two further measures are reported:
`inset` (SCIP's column is in the rule's >=0.9 set) and `expect` (the expected agreement
of a rule that also picks at random in that set: mean of 1/|set| over in-set boxes).
Prints an executed-assertion count (CLAUDE.md section 6); nothing is caught (section 7).
"""
import json, math, os, sys
from collections import Counter
import numpy as np
import discopt._rust as R

assert R.__file__.startswith(os.environ["DISCOPT_TREE"]), R.__file__
assert hasattr(R, "solve_spatial_node_py")
cap = dict(np.load(sys.argv[1]))
SCIPX = "--scip-x" in sys.argv
sys.argv = [a for a in sys.argv if a != "--scip-x"]
rows = [r for r in json.load(open(sys.argv[2])) if "scip_col" in r]
SPEC = ("n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
        "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
        "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
        "obbt_candidates").split()
kw = {k: (int(cap[k]) if cap[k].ndim == 0 else cap[k]) for k in SPEC}
n = kw["n_cols"]
integ = cap["integrality"].astype(bool)
E = np.zeros(0, dtype=np.int64)
F = np.zeros(0)
idx = {int(s): k for k, s in enumerate(cap["s_id"])}
n_assert = 0

# terms: (kind, aux col, operand cols with coefficients, f(x) evaluator)
terms = []
for k in range(len(cap["term_kind"])):
    kind = int(cap["term_kind"][k])
    assert kind == 1 and int(cap["term_p"][k]) == 2, "cstr spec: squares only"
    i, w = int(cap["term_i"][k]), int(cap["term_out"][k])
    terms.append(("sq", w, [i], lambda x, i=i: x[i] ** 2))
for k in range(len(cap["blf_w"])):
    a0, a1 = int(cap["blf_a_ptr"][k]), int(cap["blf_a_ptr"][k + 1])
    b0, b1 = int(cap["blf_b_ptr"][k]), int(cap["blf_b_ptr"][k + 1])
    ac, aa, ak = cap["blf_a_cols"][a0:a1].astype(int), cap["blf_a_coeffs"][a0:a1], float(cap["blf_a_const"][k])
    bc, bb, bk = cap["blf_b_cols"][b0:b1].astype(int), cap["blf_b_coeffs"][b0:b1], float(cap["blf_b_const"][k])
    cols = sorted(set(ac.tolist()) | set(bc.tolist()))
    terms.append(("blf", int(cap["blf_w"][k]), cols,
                  lambda x, ac=ac, aa=aa, ak=ak, bc=bc, bb=bb, bk=bk: (ak + aa @ x[ac]) * (bk + bb @ x[bc])))


def scores(x, lo, hi, split, rel, convex_skip, tol=0.0):
    v = np.zeros(n)
    for kind, w, cols, f in terms:
        fx = f(x)
        gap = x[w] - fx
        if abs(gap) <= tol:
            continue
        if convex_skip and kind == "sq" and gap < 0:
            continue  # aux below a convex square: separable by a tangent, SCIP cuts instead
        s = abs(gap) / (max(1.0, abs(fx)) if rel else 1.0)
        cs = [c for c in cols if hi[c] - lo[c] > 1e-9]
        if not cs:
            continue
        if split == "even":
            wt = np.ones(len(cs))
        elif split == "mid":
            wt = np.array([max(0.05, min(x[c] - lo[c], hi[c] - x[c]) / (hi[c] - lo[c])) for c in cs])
        elif split == "width":
            wt = np.array([hi[c] - lo[c] for c in cs])
        elif split == "relwidth":
            wt = np.array([(hi[c] - lo[c]) / (cap["global_hi"][c] - cap["global_lo"][c]) for c in cs])
        for c, ww in zip(cs, wt):
            v[c] += s * ww / wt.sum()
    return v


RULES = {
    "even_abs": dict(split="even", rel=False, convex_skip=False),  # ours, pscost weight 0
    "mid_abs": dict(split="mid", rel=False, convex_skip=False),
    "mid_rel": dict(split="mid", rel=True, convex_skip=False),
    "mid_rel_cvx": dict(split="mid", rel=True, convex_skip=True),  # SCIP as read from source
    "mid_abs_cvx": dict(split="mid", rel=False, convex_skip=True),
    "even_rel_cvx": dict(split="even", rel=True, convex_skip=True),
    "even_abs_cvx": dict(split="even", rel=False, convex_skip=True),
    "width_rel_cvx": dict(split="width", rel=True, convex_skip=True),
    "relw_rel_cvx": dict(split="relwidth", rel=True, convex_skip=True),
}
stat = {k: Counter() for k in RULES}
recon = Counter()
out = []
for r in rows:
    s = idx[r["id"]]
    lo, hi = cap["s_lo"][s], cap["s_hi"][s]
    res = R.solve_spatial_node_py(**kw, lo=lo, hi=hi, cut_ptr=np.zeros(1, dtype=np.int64), cut_cols=E,
                                  cut_coeffs=F, cut_rhs=F)
    if res["status"] != "optimal" or not math.isfinite(res["bound"]):
        res = R.solve_spatial_node_py(**kw, lo=lo, hi=hi, cut_ptr=np.zeros(1, dtype=np.int64), cut_cols=E,
                                      cut_coeffs=F, cut_rhs=F, cold_dual_start=True)
    recon[res["status"]] += 1
    if res["status"] != "optimal":
        continue
    x = np.asarray(res["x"])[:n]
    if SCIPX:
        assert r["scip_x"] is not None
        x = np.asarray(r["scip_x"])[:n]
    recon["bound_match"] += abs(res["bound"] - cap["s_bound"][s]) <= 1e-6 * max(1, abs(cap["s_bound"][s]))
    sc = r["scip_col"]
    row = {"id": r["id"], "scip": sc, "ours_tree": r["our_col"]}
    for name, cfg in RULES.items():
        v = scores(x, lo, hi, **cfg)
        top = int(np.argmax(v))
        hs = set(np.nonzero(v >= 0.9 * v.max())[0].tolist()) if v.max() > 0 else set()
        st = stat[name]
        st["n"] += 1
        st["top"] += top == sc
        st["inset"] += sc in hs
        st["expect"] += (1.0 / len(hs)) if sc in hs else 0.0
        st["setsize"] += len(hs)
        st["tree_top"] += top == r["our_col"]
        row[name] = top
        n_assert += 1
    out.append(row)

print("LP re-solve:", dict(recon))
print(f"{'rule':16s} {'n':>4s} {'top1':>6s} {'inset':>6s} {'expect':>7s} {'|set|':>6s} {'=ourTree':>8s}")
for name, st in stat.items():
    N = st["n"]
    print(f"{name:16s} {N:4d} {st['top']/N:6.3f} {st['inset']/N:6.3f} {st['expect']/N:7.3f} "
          f"{st['setsize']/N:6.2f} {st['tree_top']/N:8.3f}")
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"))
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
