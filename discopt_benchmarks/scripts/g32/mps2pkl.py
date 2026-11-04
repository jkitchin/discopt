"""mps -> dict pkl in tab_cmir_entry format (A_ub x<=b_ub, A_eq x==b_eq)."""
import sys, pickle, numpy as np, scipy.sparse as sp, highspy
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(sys.argv[1]); lp = h.getLp()
n, m = lp.num_col_, lp.num_row_
A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(m, n)).tocsr()
rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_); INF = highspy.kHighsInf
eq = (rl == ru); up = (~eq) & (ru < INF); lo = (~eq) & (rl > -INF)
A_ub = sp.vstack([A[up], -A[lo]]).tocsr(); b_ub = np.r_[ru[up], -rl[lo]]
cl, cu = np.array(lp.col_lower_), np.array(lp.col_upper_)
bounds = [(None if a <= -INF else a, None if b >= INF else b) for a, b in zip(cl, cu)]
integ = [1 if t == highspy.HighsVarType.kInteger else 0 for t in lp.integrality_] or [0] * n
pickle.dump(dict(c=np.array(lp.col_cost_), A_ub=A_ub, b_ub=b_ub, A_eq=A[eq], b_eq=rl[eq], bounds=bounds, integrality=integ), open(sys.argv[2], "wb"))
print(sys.argv[1], n, m, "int", sum(integ), "ub rows", A_ub.shape[0], "eq", int(eq.sum()))
