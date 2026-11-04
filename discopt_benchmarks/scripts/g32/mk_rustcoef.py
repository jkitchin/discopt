"""mtp_raw.pkl -> 8-sweep FBBT (Rust presolve cap) + Rust coef pass on <= rows -> mtp_rcoef.pkl"""
import pickle, sys, numpy as np, scipy.sparse as sp, importlib.util
import discopt, discopt._rust as R; print(discopt.__file__)
d = pickle.load(open("mtp_raw.pkl", "rb")); I = np.asarray(d["integrality"]) > 0; n = len(d["c"])
lb = np.array([-np.inf if b[0] is None else b[0] for b in d["bounds"]]); ub = np.array([np.inf if b[1] is None else b[1] for b in d["bounds"]])
Aub = d["A_ub"].tocsr(); bub = np.asarray(d["b_ub"], float); Aeq = d["A_eq"].tocsr(); beq = np.asarray(d["b_eq"], float)
rows = [(Aub, bub), (Aeq, beq), (-Aeq, -beq)]
def fbbt(lb, ub, sweeps):
    lb, ub = lb.copy(), ub.copy()
    for _ in range(sweeps):
        ch = 0
        for A, b in rows:
            for i in range(A.shape[0]):
                s, e = A.indptr[i], A.indptr[i+1]; cols = A.indices[s:e]; a = A.data[s:e]
                lo = np.where(a > 0, a * lb[cols], a * ub[cols]); ninf = np.isinf(lo).sum()
                if ninf > 1: continue
                tot = lo[np.isfinite(lo)].sum()
                for k, j in enumerate(cols):
                    if ninf == 1 and np.isfinite(lo[k]): continue
                    bnd = (b[i] - (tot - (lo[k] if np.isfinite(lo[k]) else 0.0))) / a[k]
                    if a[k] > 0:
                        if I[j]: bnd = np.floor(bnd + 1e-9)
                        if bnd < ub[j] - 1e-9 * (1 + abs(bnd)): ub[j] = bnd; ch += 1
                    else:
                        if I[j]: bnd = np.ceil(bnd - 1e-9)
                        if bnd > lb[j] + 1e-9 * (1 + abs(bnd)): lb[j] = bnd; ch += 1
        if ch == 0: break
    return lb, ub
SW = int(sys.argv[1]); lb2, ub2 = fbbt(lb, ub, SW)
C = Aub.tocsc(); BIG = 1e20; mm = Aub.shape[0]
S = sp.hstack([C, sp.identity(mm, format="csc")]).tocsc(); S.sort_indices()
L = np.r_[np.where(np.isinf(lb2), -BIG, lb2), np.zeros(mm)]; U = np.r_[np.where(np.isinf(ub2), BIG, ub2), np.full(mm, BIG)]
vals, b2, nr, nc, su, sp_ = R.milp_coef_tighten_csc_py(mm, n + mm, n, S.indptr.astype(np.int64), S.indices.astype(np.int64), S.data.astype(float), bub, L, U, np.nonzero(I)[0].astype(np.int64))
vals = np.asarray(vals)[:C.nnz]; assert np.all(S.indptr[:n+1] == C.indptr)
print("sweeps", SW, "rows", nr, "coefs", nc, "skip", su, sp_)
C2 = sp.csc_matrix((np.asarray(vals), C.indices, C.indptr), shape=C.shape)
d2 = dict(d); d2["A_ub"] = C2.tocsr(); d2["b_ub"] = np.asarray(b2)
d2["bounds"] = [(None if np.isinf(a) else a, None if np.isinf(b) else b) for a, b in zip(lb2, ub2)]
pickle.dump(d2, open(sys.argv[2], "wb"))
