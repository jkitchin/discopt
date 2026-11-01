"""Solve an MPS MILP through discopt's default route; compare with SCIP's optimum."""
import sys, warnings
import numpy as np, highspy, scipy.sparse as sp
import discopt.modeling as dm
warnings.simplefilter("ignore")
path, truth = sys.argv[1], float(sys.argv[2])
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(path); lp = h.getLp()
n, mr = lp.num_col_, lp.num_row_
A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(mr, n)).tocsr()
integ = list(lp.integrality_) if len(lp.integrality_) else [highspy.HighsVarType.kContinuous] * n
m = dm.Model("mps")
INF = highspy.kHighsInf
xs = []
for j in range(n):
    lo = lp.col_lower_[j]; hi = lp.col_upper_[j]
    lo = None if lo <= -INF else lo; hi = None if hi >= INF else hi
    if integ[j] == highspy.HighsVarType.kInteger:
        xs.append(m.integer(f"x{j}", lb=lo if lo is not None else -1e6, ub=hi if hi is not None else 1e6))
    else:
        xs.append(m.continuous(f"x{j}", lb=lo if lo is not None else -1e19, ub=hi if hi is not None else 1e19))
perm = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0).permutation(mr) if len(sys.argv) > 3 else range(mr)
for i in perm:
    row = A.getrow(i); expr = sum(float(v) * xs[j] for j, v in zip(row.indices, row.data))
    if isinstance(expr, int): continue
    if lp.row_upper_[i] < INF: m.subject_to(expr <= float(lp.row_upper_[i]))
    if lp.row_lower_[i] > -INF: m.subject_to(expr >= float(lp.row_lower_[i]))
m.minimize(sum(float(c) * xs[j] for j, c in enumerate(lp.col_cost_) if c != 0) + float(lp.offset_))
r = m.solve(time_limit=120)
ok = r.objective is not None and abs(r.objective - truth) <= 1e-6 + 1e-4 * abs(truth)
bound_ok = r.bound is None or r.bound <= truth + 1e-6 + 1e-6 * abs(truth)
print(f"discopt status={r.status} certified={r.gap_certified} obj={r.objective} bound={r.bound} route={getattr(r, 'algorithm_route', None)}")
print(f"truth (SCIP) {truth}: objective {'OK' if ok else 'WRONG'}, bound {'OK' if bound_ok else 'ABOVE OPTIMUM'}")
