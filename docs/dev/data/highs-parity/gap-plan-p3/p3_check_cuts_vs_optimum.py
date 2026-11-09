"""Check dumped root cuts (coeffs . x >= rhs, depth 0) against HiGHS's optimum of the dumped WORKING problem."""
import json, sys, highspy, numpy as np
d = json.load(open(sys.argv[1])); inf = highspy.kHighsInf
cl = lambda v: [(-inf if x <= -1e29 else inf if x >= 1e29 else x) for x in v]
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = d["n"], d["m"]
lp.col_cost_ = np.array(d["c"]); lp.col_lower_ = np.array(cl(d["l"])); lp.col_upper_ = np.array(cl(d["u"]))
lp.row_lower_ = np.array(d["b"]); lp.row_upper_ = np.array(d["b"])
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = np.array(d["colptr"], dtype=np.int32); lp.a_matrix_.index_ = np.array(d["rowidx"], dtype=np.int32); lp.a_matrix_.value_ = np.array(d["vals"])
it = [highspy.HighsVarType.kContinuous] * d["n"]
for j in d["ints"]: it[j] = highspy.HighsVarType.kInteger
lp.integrality_ = it; h.passModel(lp); h.run()
x = np.array(h.getSolution().col_value); n = len(x)
print("HiGHS on working problem:", h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value, "n", n)
checked = viol = skipped = 0; worst = 0.0
for line in open(sys.argv[2]):
    c = json.loads(line)
    if c["depth"] != 0 or any(j >= n for j, _ in c["nz"]): skipped += 1; continue
    act = sum(v * x[j] for j, v in c["nz"]); checked += 1; g = c["rhs"] - act
    if g > 1e-6 * (1 + abs(c["rhs"])): viol += 1; worst = max(worst, g)
print(f"checked={checked} violated_by_optimum={viol} worst={worst:.3e} skipped(slack refs or nested)={skipped}")
sys.exit(0 if checked and not viol else 1)
