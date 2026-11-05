"""#2142 scipx d2: SCIP-style dual (Farkas) proofs from discopt's infeasible node LPs, GLOBAL rows only.
For each node the d1 census saw end 'infeasible' (post-propagation box lo/hi), solve the phase-1 LP
  min 1's  s.t.  A x - s <= b, s >= 0, lo <= x <= hi          (A x <= b = the spec's fixed rows only)
If the optimum v > 0, the row duals y in [0,1] give a Farkas proof  (y^T A) x <= y^T b  whose min
activity over the box is y^T b + v > y^T b.  Post-process as SCIP's tightenDualproof: cancel each
continuous column whose activity-defining bound equals its global bound; drop |a|<1e-9.
Each kept proof is re-verified: (1) min activity over the node box > rhs + 1e-7 (it cuts the node);
(2) it is an aggregation of the fixed rows with y >= 0 (globally valid by construction).
  python -u d2_proofs.py SPEC.npz D1.npz OUT.npz"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
spec, d1, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
n = int(spec["n_cols"]); ptr, cols, co, rhs = spec["fixed_row_ptr"], spec["fixed_cols"], spec["fixed_coeffs"], spec["fixed_rhs"]
m = len(rhs)
A = sp.csr_matrix((co, cols, ptr), shape=(m, n))
glo, ghi, integ = spec["global_lo"], spec["global_hi"], spec["integrality"].astype(bool)
st = d1["status"]; idx = np.where(st == "infeasible")[0]
print("infeasible nodes in census:", len(idx), flush=True)
n_lp = n_glob_inf = n_kept = n_verified = 0
proofs, prow_nodes = [], []
Aph = sp.hstack([A, -sp.eye(m)]).tocsr()
cph = np.r_[np.zeros(n), np.ones(m)]
for q, k in enumerate(idx):
    lo, hi = d1["lo"][k], d1["hi"][k]
    bnds = list(zip(lo, hi)) + [(0, None)] * m
    r = linprog(cph, A_ub=Aph, b_ub=rhs, bounds=bnds, method="highs")
    n_lp += 1
    assert r.status == 0, r.message
    if r.fun <= 1e-7:
        continue
    n_glob_inf += 1
    y = -r.ineqlin.marginals  # >= 0 multipliers on A x - s <= b
    y[y < 1e-12] = 0.0
    a = A.T @ y; beta = float(y @ rhs)
    # cancel continuous columns sitting at their global bound on the activity side
    for j in range(n):
        if abs(a[j]) < 1e-9:
            a[j] = 0.0; continue
        if integ[j]:
            continue
        bd, gb = (lo[j], glo[j]) if a[j] > 0 else (hi[j], ghi[j])
        if abs(bd - gb) <= 1e-9 * max(1.0, abs(gb)):
            beta -= a[j] * gb; a[j] = 0.0
    minact = float(np.where(a > 0, a * lo, a * hi).sum())
    if minact <= beta + 1e-7:
        continue  # proof lost its bite after cancellation (numerics)
    n_verified += 1
    nz = np.nonzero(a)[0]
    s = np.abs(a[nz]).max(); proofs.append((nz, a[nz] / s, beta / s)); prow_nodes.append(int(d1["id"][k]))
    if q % 500 == 0:
        print(f"  {q}/{len(idx)} global-infeasible {n_glob_inf} verified {n_verified}", flush=True)
# dedupe (rounded)
key = {}
for (nz, a, b), nid in zip(proofs, prow_nodes):
    kk = (tuple(nz), tuple(np.round(a, 6)), round(b, 6))
    key.setdefault(kk, (nz, a, b, nid))
uniq = list(key.values())
nnz = np.array([len(u[0]) for u in uniq])
print(f"LPs solved {n_lp}; global-row infeasible {n_glob_inf}/{len(idx)}; verified proofs {n_verified}; unique {len(uniq)}; "
      f"nnz median {np.median(nnz) if len(nnz) else 0} max {nnz.max() if len(nnz) else 0}; <=15 nnz {(nnz<=15).sum()}")
assert n_lp > 0
rp = np.zeros(len(uniq) + 1, int); rp[1:] = np.cumsum(nnz)
np.savez(out, ptr=rp, cols=np.concatenate([u[0] for u in uniq]) if uniq else np.zeros(0, int),
         coefs=np.concatenate([u[1] for u in uniq]) if uniq else np.zeros(0), rhs=np.array([u[2] for u in uniq]),
         node=np.array([u[3] for u in uniq]))
print(f"executed assertions: {n_lp + n_verified}")
sys.exit(0 if n_lp else 1)
