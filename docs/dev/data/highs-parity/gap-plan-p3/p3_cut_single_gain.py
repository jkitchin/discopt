"""Per-cut bound gain: each round-1 cut added ALONE to the LP relaxation. Reports the gain
distribution per side and, for discopt, whether the selected cuts are its strongest."""
import json, sys, numpy as np, highspy
mps, hs_dump, ds_cand, ds_sel = sys.argv[1:5]
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(mps)
n = h.getLp().num_col_
h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)); h.run()
base = h.getInfo().objective_function_value; inf = highspy.kHighsInf
def gain(lo, hi, nz):
    g = highspy.Highs(); g.setOptionValue("output_flag", False); g.passModel(h.getLp())
    idx = np.array([j for j, _ in nz], dtype=np.int32); val = np.array([v for _, v in nz])
    g.addRow(lo, hi, len(idx), idx, val); g.run()
    return g.getInfo().objective_function_value - base
hs = [x for x in map(json.loads, open(hs_dump)) if x.get("round") == 1]
dc = [x for x in map(json.loads, open(ds_cand)) if x.get("round") == 1]
dsl = [x for x in map(json.loads, open(ds_sel)) if x.get("round") == 1]
key = lambda x: (round(x["rhs"], 9), tuple((j, round(v, 9)) for j, v in x["nz"]))
sel = {key(x) for x in dsl}
hg = sorted((gain(-inf, x["ub"], x["nz"]) for x in hs), reverse=True)
dg = sorted(((gain(x["rhs"], inf, x["nz"]), key(x) in sel, len(x["nz"])) for x in dc), reverse=True)
print(f"LP {base:.1f}; HiGHS round-1 cuts {len(hs)}, discopt candidates {len(dc)} (selected {len(sel)})")
print("HiGHS single-cut gains (top 10):", [round(v) for v in hg[:10]], " sum", round(sum(hg)))
print("discopt single-cut gains (top 10):", [round(v[0]) for v in dg[:10]], " sum", round(sum(v[0] for v in dg)))
print("discopt top-15 by gain, selected?:", [(round(v[0]), 'S' if v[1] else '-') for v in dg[:15]])
print("discopt selected cuts' gains:", sorted([round(v[0]) for v in dg if v[1]], reverse=True))
print("count compared:", len(hg) + len(dg))
