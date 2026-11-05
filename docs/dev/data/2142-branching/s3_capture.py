"""#2142 item 3 step 3 -- capture (box, cutoff, our branching decision) at a sample of cstr
big-M nodes from the default-path deterministic tree, plus the spec the kernel got.

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u s3_capture.py MAX_NODES N_SAMPLE OUT.npz

Decisions are recovered as in s1_splits.py (a child's pre-propagation box is its parent's
post-propagation box with one coordinate overridden). Prints an executed-assertion count."""
import os, sys, time
import numpy as np
assert "DISCOPT_LP_DUAL_REBUILD" not in os.environ
WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R
assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT)
n_assert = 1
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
max_nodes, n_sample, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SPEC_KEYS = ("n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
             "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
             "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
             "obbt_candidates").split()
calls = []
def _w(*a, **k):
    store = {}
    def obs(rec):
        store[rec["id"]] = (rec["parent"], np.asarray(rec["lo"]).copy(), np.asarray(rec["hi"]).copy(),
                            np.asarray(rec["pre_lo"]).copy(), np.asarray(rec["pre_hi"]).copy(),
                            rec["cutoff"], rec["status"], rec["bound"])
    k["node_observer"] = obs
    calls.append({"spec": {kk: k[kk] for kk in SPEC_KEYS}, "recs": store})
    return _orig(*a, **k)
_orig = R.solve_spatial_tree_py
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
res = m.solve(gdp_method="big-m", time_limit=7200, deterministic=True, max_nodes=max_nodes)
print("solve", res.status, res.bound, res.node_count, flush=True)
main = max(calls, key=lambda c: len(c["recs"]))
recs = main["recs"]
dec = {}
for nid, r in recs.items():
    par = r[0]
    if par is None:
        continue
    P = recs[par]
    d = [(c, 0) for c in np.nonzero(r[3] != P[1])[0]] + [(c, 1) for c in np.nonzero(r[4] != P[2])[0]]
    assert len(d) == 1
    c, side = d[0]
    v = r[3][c] if side == 0 else r[4][c]
    dec.setdefault(par, (int(c), float(v), side))
n_assert += 1
pars = sorted(dec)
print("branched parents with a recovered decision:", len(pars))
idx = np.linspace(0, len(pars) - 1, n_sample).round().astype(int)
pick = [pars[i] for i in sorted(set(idx))]
integ = np.asarray(main["spec"]["integrality"]).astype(bool)
save = {k: np.asarray(v) for k, v in main["spec"].items()}
save["s_id"] = np.array(pick)
save["s_lo"] = np.stack([recs[p][1] for p in pick]); save["s_hi"] = np.stack([recs[p][2] for p in pick])
save["s_cutoff"] = np.array([np.nan if recs[p][5] is None else recs[p][5] for p in pick])
save["s_bound"] = np.array([recs[p][7] for p in pick], dtype=float)
save["s_col"] = np.array([dec[p][0] for p in pick])
# integer branch: children carry floor / ceil; record the point floor+0.5
save["s_val"] = np.array([(np.floor(dec[p][1]) + 0.5 if dec[p][2] == 1 else np.ceil(dec[p][1]) - 0.5) if integ[dec[p][0]] else dec[p][1] for p in pick])
np.savez(out, **save)
print("saved", out, len(pick)); n_assert += 1
print(f"executed assertions: {n_assert}")
