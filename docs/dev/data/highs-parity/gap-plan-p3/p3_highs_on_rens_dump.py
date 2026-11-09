"""Solve a dumped discopt sub-problem (A x = b, l <= x <= u, ints) with HiGHS; print the solution-source log."""
import json, sys, numpy as np, highspy
d = json.load(open(sys.argv[1]))
h = highspy.Highs(); h.setOptionValue("output_flag", True); h.setOptionValue("log_to_console", False)
h.setOptionValue("log_file", sys.argv[2]); h.setOptionValue("time_limit", float(sys.argv[3]) if len(sys.argv) > 3 else 30.0)
h.setOptionValue("threads", 1)
lp = highspy.HighsLp(); m, n = d["m"], d["n"]
inf = highspy.kHighsInf
cl = lambda v: [(-inf if x <= -1e29 else inf if x >= 1e29 else x) for x in v]
lp.num_col_, lp.num_row_ = n, m
lp.col_cost_ = np.array(d["c"]); lp.col_lower_ = np.array(cl(d["l"])); lp.col_upper_ = np.array(cl(d["u"]))
lp.row_lower_ = np.array(d["b"]); lp.row_upper_ = np.array(d["b"])
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = np.array(d["colptr"], dtype=np.int32); lp.a_matrix_.index_ = np.array(d["rowidx"], dtype=np.int32)
lp.a_matrix_.value_ = np.array(d["vals"])
integ = [highspy.HighsVarType.kContinuous] * n
for j in d["ints"]: integ[j] = highspy.HighsVarType.kInteger
lp.integrality_ = integ
h.passModel(lp); h.run()
print(h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value, "nodes", h.getInfo().mip_node_count)
