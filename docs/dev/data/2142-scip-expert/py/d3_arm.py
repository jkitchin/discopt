"""#2142 scipx d3: discopt cstr big-M tree with extra GLOBALLY VALID linear rows appended to the kernel
spec's fixed rows (Farkas dual proofs = nonnegative aggregations of the spec's own fixed rows; no solver
code change). Emulates SCIP's dual-proof constraints being available to propagation.
  python -u d3_arm.py MAX_NODES PROOFS.npz|none MAXNODEID SPEC.npz"""
import sys, time, os
n_perm = 0
import numpy as np
import discopt, discopt._rust as R
assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
max_nodes, pf, maxid, specf = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
cap = np.load(specf)
n_assert = 0
rows = []
if pf != "none":
    P = np.load(pf)
    for k in range(len(P["rhs"])):
        if P["node"][k] <= maxid:
            rows.append((P["cols"][P["ptr"][k]:P["ptr"][k + 1]], P["coefs"][P["ptr"][k]:P["ptr"][k + 1]], float(P["rhs"][k])))
print("appending", len(rows), "proof rows", flush=True)
calls = []
_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    global n_assert
    assert not a
    if rows:
        assert int(k["n_cols"]) == int(cap["n_cols"]) and np.array_equal(np.asarray(k["fixed_rhs"]), cap["fixed_rhs"]), "spec differs from the capture"
        n_assert += 1
        ptr = list(np.asarray(k["fixed_row_ptr"])); cols = list(np.asarray(k["fixed_cols"])); co = list(np.asarray(k["fixed_coeffs"])); rhs = list(np.asarray(k["fixed_rhs"]))
        for c_, a_, b_ in rows:
            cols += list(map(int, c_)); co += list(map(float, a_)); rhs.append(b_); ptr.append(len(cols))
        k["fixed_row_ptr"] = np.array(ptr, dtype=np.asarray(k["fixed_row_ptr"]).dtype); k["fixed_cols"] = np.array(cols, dtype=np.asarray(k["fixed_cols"]).dtype)
        k["fixed_coeffs"] = np.array(co); k["fixed_rhs"] = np.array(rhs)
    seed = int(os.environ.get("PERM_SEED", "0"))
    if seed:
        global n_perm
        ptr = np.asarray(k["fixed_row_ptr"]); cols = np.asarray(k["fixed_cols"]); co = np.asarray(k["fixed_coeffs"]); rhs = np.asarray(k["fixed_rhs"])
        perm = np.random.default_rng(seed).permutation(len(rhs))
        nc, ncf, np_ = [], [], [0]
        for q in perm:
            nc.append(cols[ptr[q]:ptr[q + 1]]); ncf.append(co[ptr[q]:ptr[q + 1]]); np_.append(np_[-1] + ptr[q + 1] - ptr[q])
        k["fixed_row_ptr"] = np.array(np_, dtype=ptr.dtype); k["fixed_cols"] = np.concatenate(nc).astype(cols.dtype)
        k["fixed_coeffs"] = np.concatenate(ncf); k["fixed_rhs"] = rhs[perm].copy()
        n_perm += 1
    r = _orig(*a, **k)
    calls.append((k.get("max_nodes"), r["bound"], r["node_count"], r.get("n_undecided")))
    return r
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="big-m", time_limit=7200, deterministic=True, max_nodes=max_nodes)
print(f"PERM_SEED={os.environ.get('PERM_SEED','0')} n_perm={n_perm} ARM proofs={pf} maxid={maxid} nrows={len(rows)} MAX_NODES={max_nodes} status={res.status} obj={res.objective} "
      f"bound={res.bound!r} nodes={res.node_count} wall={time.perf_counter()-t0:.1f}s kernel_calls={calls}", flush=True)
assert calls; n_assert += 1
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
