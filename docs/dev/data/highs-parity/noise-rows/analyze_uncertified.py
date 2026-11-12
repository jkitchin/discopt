import json, sys, glob, numpy as np, scipy.sparse as sp, highspy
INF = 1e20
for f in sorted(glob.glob(sys.argv[1] + "/uncertified_*.json")):
    d = json.load(open(f)); m, n = d["m"], d["n"]
    A = sp.csc_matrix((d["vals"], d["row_idx"], d["col_ptr"]), shape=(m, n))
    l = np.array(d["l"]); u = np.array(d["u"]); b = np.array(d["b"]); c = np.array(d["c"]); y = np.array(d["rho"])
    lo = np.where(l <= -INF, -np.inf, l); up = np.where(u >= INF, np.inf, u)
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = m
    lp.col_cost_ = c; lp.col_lower_ = lo; lp.col_upper_ = up; lp.row_lower_ = b; lp.row_upper_ = b
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = d["col_ptr"]; lp.a_matrix_.index_ = d["row_idx"]; lp.a_matrix_.value_ = d["vals"]
    h.passModel(lp); h.run(); st = h.modelStatusToString(h.getModelStatus())
    res = []
    for s in (1, -1):
        ys = s * y; by = b @ ys; aty = A.T @ ys
        box = np.where(aty > 0, aty * up, np.where(aty < 0, aty * lo, 0.0))
        nopen = int(np.sum(~np.isfinite(box)))
        g0 = by - np.sum(box[np.isfinite(box)])
        res.append((s, round(float(by), 6), nopen, float(g0), float(np.abs(b * ys).sum()), float(np.abs(box[np.isfinite(box)]).sum())))
    print(f.split("/")[-1], f"m={m} n={n} HiGHS={st}", "max|y|=%.3g" % np.abs(y).max())
    for r in res: print("    sign %+d  b.y=%s open_cols=%d g0=%.3e  sum|b y|=%.3g sum|box|=%.3g" % r)
