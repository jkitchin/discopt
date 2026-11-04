"""Do the rows the replica tightens contain any column with |bound| >= 1e19 (default box)?"""
import pickle, sys, numpy as np, scipy.sparse as sp
INF = 1e20
checked = 0
for f in sys.argv[1:]:
    d = pickle.load(open(f, 'rb')); ns, n, m = d['n_struct'], d['n'], d['m']
    A = sp.csc_matrix((d['vals'], d['row_idx'], d['col_ptr']), shape=(m, n)); lb, ub = d['lb'], d['ub']
    isint = np.zeros(n, bool); isint[d['integer_cols']] = True
    Lg = A[:, ns:].tocsc(); R = A[:, :ns].tocsr(); hit = tot = 0; worst = 0.0
    for i in range(m):
        s, e = Lg.indptr[i], Lg.indptr[i+1]
        if not (e - s == 1 and Lg.indices[s] == i and Lg.data[s] == 1.0 and lb[ns+i] == 0 and ub[ns+i] >= INF): continue
        s, e = R.indptr[i], R.indptr[i+1]; cols = R.indices[s:e]; a = R.data[s:e]
        bnd = np.where(a > 0, ub[cols], lb[cols])
        if (np.abs(bnd) >= INF).any() or not isint[cols].any(): continue
        Mx = (a * bnd).sum(); dd = Mx - d['b'][i]
        if dd <= 1e-6: continue
        if not any(isint[j] and abs(aj) > dd for aj, j in zip(a, cols)): continue
        tot += 1; checked += 1
        if (np.abs(bnd) >= 1e19).any(): hit += 1; worst = max(worst, dd)
    print(f"{f:45s} tightened_rows {tot:5d} with_default_box_col {hit:5d} max_d_there {worst:.3g}")
assert checked > 0; print("PROBE_DONE", checked)
