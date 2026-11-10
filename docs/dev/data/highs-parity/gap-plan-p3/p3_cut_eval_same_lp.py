"""Add each solver's root cuts, up to round k, to the SAME LP relaxation (HiGHS LP) and
report the bound. HiGHS cuts: a.x <= ub over MPS columns (presolve off). discopt cuts:
a.x >= rhs over its structural columns, whose order is checked against the MPS objective.
Prints the number of cuts evaluated per side (CLAUDE.md §6)."""
import json, sys, numpy as np, highspy
mps, hs_dump, ds_dump = sys.argv[1], sys.argv[2], sys.argv[3]
rounds = [int(r) for r in sys.argv[4].split(",")]
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(mps)
lp = h.getLp(); n = lp.num_col_
h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8))
h.run(); base = h.getInfo().objective_function_value
hs = [json.loads(l) for l in open(hs_dump)]
ds_lines = [json.loads(l) for l in open(ds_dump)]
hdr = ds_lines[0]; ds = []
for x in ds_lines[1:]:
    if "c" in x:  # a later top-level driver call: keep the first call's root only
        break
    ds.append(x)
c = np.array(lp.col_cost_)
dc = np.array(hdr["c"])
ok_map = hdr["ns"] == n and np.allclose(np.abs(dc), np.abs(c), rtol=1e-9, atol=1e-12)
print(f"MPS cols {n}, discopt ns {hdr['ns']}, objective columns match: {ok_map}")
if not ok_map:
    sys.exit("column spaces differ; refusing to compare")
def bound_with(rows):
    g = highspy.Highs(); g.setOptionValue("output_flag", False); g.passModel(h.getLp())
    for lo, hi, nz in rows:
        idx = np.array([j for j, _ in nz], dtype=np.int32); val = np.array([v for _, v in nz])
        g.addRow(lo, hi, len(idx), idx, val)
    g.run(); return g.getInfo().objective_function_value, g.modelStatusToString(g.getModelStatus())
inf = highspy.kHighsInf
print(f"LP (no cuts): {base:.6f}")
for k in rounds:
    hr = [(-inf, x["ub"], x["nz"]) for x in hs if x["round"] <= k]
    skipped = sum(1 for x in ds if x["round"] <= k and any(j >= n for j, _ in x["nz"]))
    dr = [(x["rhs"], inf, x["nz"]) for x in ds if x["round"] <= k and all(j < n for j, _ in x["nz"])]
    hb, hst = bound_with(hr); db, dst = bound_with(dr)
    print(f"rounds<= {k:3}: HiGHS {len(hr):4} cuts -> {hb:.6f} ({hst}) | discopt {len(dr):4} cuts (+{skipped} over slacks, skipped) -> {db:.6f} ({dst})")
