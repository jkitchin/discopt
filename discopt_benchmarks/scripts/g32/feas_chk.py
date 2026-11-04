import highspy, numpy as np, scipy.sparse as sp, sys
def load(p):
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(p); return h
h = load("mtp.mps"); h.setOptionValue("mip_rel_gap", 1e-9); h.setOptionValue("time_limit", 120.0); h.run()
x = np.array(h.getSolution().col_value); print("ref obj", h.getInfo().objective_function_value)
nchk = 0
for f in sys.argv[1:]:
    lp = load(f).getLp(); A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(lp.num_row_, lp.num_col_))
    r = A @ x; tol = 1e-6
    vr = np.maximum(np.array(lp.row_lower_) - r, r - np.array(lp.row_upper_)).max()
    vc = np.maximum(np.array(lp.col_lower_) - x, x - np.array(lp.col_upper_)).max()
    print(f, "max row viol", vr, "max col viol", vc, "obj", float(np.dot(lp.col_cost_, x))); nchk += lp.num_row_ + lp.num_col_
    assert vr <= tol and vc <= tol
print("checks", nchk); assert nchk > 0
