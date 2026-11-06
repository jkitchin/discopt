"""#2142 §114 step 2 (independent of SCIP): witness search on the 94 captured hull boxes.
Every lifted column (term / BLF output) is a function of the others, so the free variables are the
84-79 = 5 non-output lifted columns plus the 236 original columns. Binaries are fixed to the box's
integral range (boxes with free binaries: 8 random assignments). The lifted columns are computed by
FORWARD evaluation in topological order, and scipy least_squares (bounded to the box) minimises
the fixed-row violations plus lifted-column box violations (relative for |bound| > 1), optionally
plus (c^T x - cutoff)+. A start counts as a witness only if the assembled point passes the
independent numpy check specio.residuals <= 1e-6 AND lies in the box (1e-6 abs + 1e-9 rel).
A found witness refutes "empty"; no witness is NOT a proof of emptiness (local method).
  python -u c8_lsq_witness.py BOXES.npz [n_starts]"""
import math, os, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specio  # noqa: E402

d, kw = specio.load(sys.argv[1])
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
n = kw["n_cols"]
integ = np.asarray(kw["integrality"]).astype(bool)
ptr, cols, co, rhs = kw["fixed_row_ptr"], kw["fixed_cols"], kw["fixed_coeffs"], kw["fixed_rhs"]
import scipy.sparse as ss
A = ss.csr_matrix((co, cols, ptr), shape=(len(rhs), n))
# ops: (out, deps, fn)
ops = []
for k in range(len(kw["term_kind"])):
    kind, i, j, w, p = (int(kw[q][k]) for q in ("term_kind", "term_i", "term_j", "term_out", "term_p"))
    a, b = float(kw["term_coeff"][k]), float(kw["term_cst"][k])
    f = {0: lambda x, i=i, j=j: x[i] * x[j], 1: lambda x, i=i, p=p: x[i] ** p,
         2: lambda x, i=i, a=a, b=b: (a * x[i] + b) ** 2, 3: lambda x, i=i, a=a, b=b: math.sqrt(max(a * x[i] + b, 0.0)),
         4: lambda x, i=i, a=a, b=b, p=p: (a * x[i] + b) ** p}[kind]
    ops.append((w, {i, j} if kind == 0 else {i}, f))
for k in range(len(kw["blf_w"])):
    sa = slice(int(kw["blf_a_ptr"][k]), int(kw["blf_a_ptr"][k + 1]))
    sb = slice(int(kw["blf_b_ptr"][k]), int(kw["blf_b_ptr"][k + 1]))
    ac, aco, acs = kw["blf_a_cols"][sa], kw["blf_a_coeffs"][sa], float(kw["blf_a_const"][k])
    bc, bco, bcs = kw["blf_b_cols"][sb], kw["blf_b_coeffs"][sb], float(kw["blf_b_const"][k])
    ops.append((int(kw["blf_w"][k]), set(ac.tolist()) | set(bc.tolist()),
                lambda x, ac=ac, aco=aco, acs=acs, bc=bc, bco=bco, bcs=bcs: (acs + aco @ x[ac]) * (bcs + bco @ x[bc])))
outs = {o for o, _, _ in ops}
order, done = [], set(range(n)) - outs
while len(order) < len(ops):
    prog = False
    for op in ops:
        if op[0] not in done and op[1] <= done:
            order.append(op); done.add(op[0]); prog = True
    assert prog, "cyclic lifted definitions"
free = np.array(sorted(set(range(n)) - outs))
outc = np.array([o for o, _, _ in order])


def assemble(z, base):
    x = base.copy()
    x[free] = z
    for o, _, f in order:
        x[o] = f(x)
    return x


def search(lo, hi, cut, rng):
    lo = lo.copy(); hi = hi.copy()
    best = None
    fl, fh = lo[free], hi[free]
    fixed = fh - fl <= 0
    for s in range(NS):
        base = np.zeros(n)
        z0 = np.where(np.isfinite(fl) & (fl > -1e19), fl, -10.0) + rng.random(len(free)) * (
            np.minimum(np.where(fh < 1e19, fh, 10.0), np.where(fl > -1e19, fl, -10.0) + 20.0)
            - np.where(fl > -1e19, fl, -10.0))
        z0 = np.clip(z0, fl, fh)
        mov = ~fixed
        zf = z0.copy()

        def res(zm):
            z = zf.copy(); z[mov] = zm
            x = assemble(z, base)
            r1 = np.maximum(A @ x - rhs, 0.0)
            v = x[outc]; l, h = lo[outc], hi[outc]
            r2 = np.maximum(l - v, 0) / np.maximum(1, np.abs(l)) + np.maximum(v - h, 0) / np.maximum(1, np.abs(h))
            r3 = [max(float(kw["c"] @ x) - cut, 0.0)] if cut is not None else []
            return np.concatenate([r1, r2, r3])
        lb, ub = fl[mov], fh[mov]
        lb = np.where(lb > -1e19, lb, -np.inf); ub = np.where(ub < 1e19, ub, np.inf)
        ub = np.maximum(ub, lb + 1e-300)
        sol = least_squares(res, np.clip(z0[mov], lb, ub), bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400)
        z = zf.copy(); z[mov] = sol.x
        x = assemble(z, base)
        r = specio.residuals(kw, x)
        tol = 1e-6 + 1e-9 * np.maximum(np.abs(np.where(np.isfinite(lo), lo, 0)), np.abs(np.where(np.isfinite(hi), hi, 0)))
        inbox = bool(np.all(x >= lo - tol) and np.all(x <= hi + tol))
        okc = cut is None or float(kw["c"] @ x) <= cut + 1e-6
        ok = r["max"] <= 1e-6 and inbox and okc
        if best is None or (ok and not best[0]) or (not best[0] and r["max"] < best[1]):
            best = (ok, float(r["max"]), float(kw["c"] @ x), inbox)
        if ok:
            break
    return best


rng = np.random.default_rng(0)
nb = len(d["rec_id"])
W = Wc = 0
for b in range(nb):
    lo0, hi0 = d["rec_pre_lo"][b].copy(), d["rec_pre_hi"][b].copy()
    ilo, ihi = np.ceil(lo0[integ] - 1e-6), np.floor(hi0[integ] + 1e-6)
    assigns = [ilo] if np.all(ilo == ihi) else [np.where(ilo == ihi, ilo, rng.integers(0, 2, len(ilo))) for _ in range(8)]
    res_nc = res_c = None
    cut = float(d["rec_cutoff"][b]); cut = cut if math.isfinite(cut) else None
    for asg in assigns:
        lo, hi = lo0.copy(), hi0.copy()
        lo[integ] = asg; hi[integ] = asg
        r = search(lo, hi, None, rng)
        if res_nc is None or (r[0] and not res_nc[0]) or (not res_nc[0] and r[1] < res_nc[1]):
            res_nc = r
        if res_nc[0]:
            break
    W += res_nc[0]
    print(f"box {b} id {int(d['rec_id'][b])}: witness(no cutoff)={res_nc[0]} resid={res_nc[1]:.2e} obj={res_nc[2]:.4f} inbox={res_nc[3]}", flush=True)
print(f"verified witnesses (no cutoff): {W}/{nb}")
print(f"executed searches: {nb}")
sys.exit(0 if nb else 1)
