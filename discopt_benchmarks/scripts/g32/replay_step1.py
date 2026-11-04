"""#1917 G3.2 step 1 differential panel: replay captured Rust-MILP inputs with
DISCOPT_RUST_MILP_COEF_TIGHTEN=0 vs =1 (interleaved per instance), against a HiGHS reference
on the same standard form. Emits one JSON line per instance. usage: replay.py T out.jsonl pkl..."""
import os, sys, json, time, pickle
os.environ["DISCOPT_PROFILE"] = "1"
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import discopt._rust as R
assert b"MilpSeedRefused" in open(R.__file__, "rb").read(), "marker absent"
import numpy as np, scipy.sparse as sp, highspy
T = float(sys.argv[1]); out = open(sys.argv[2], "a")
INF = 1e20
def highs_ref(d):
    n, m = d["n"], d["m"]
    A = sp.csc_matrix((d["vals"], d["row_idx"], d["col_ptr"]), shape=(m, n))
    lb = np.where(d["lb"] <= -INF, -highspy.kHighsInf, d["lb"]); ub = np.where(d["ub"] >= INF, highspy.kHighsInf, d["ub"])
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = m; lp.col_cost_ = d["c"]; lp.col_lower_ = lb; lp.col_upper_ = ub
    lp.row_lower_ = d["b"]; lp.row_upper_ = d["b"]; lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    integ = [highspy.HighsVarType.kContinuous] * n
    for j in d["integer_cols"]: integ[int(j)] = highspy.HighsVarType.kInteger
    lp.integrality_ = integ
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", 120.0); h.setOptionValue("mip_rel_gap", 1e-9); h.setOptionValue("mip_abs_gap", 1e-9)
    h.passModel(lp); h.run()
    st = h.modelStatusToString(h.getModelStatus()); info = h.getInfo()
    oc = d.get("obj_const", 0.0)
    return st, info.objective_function_value + oc, info.mip_dual_bound + oc
def verify(d, x):
    """Independent check of the returned structural point against the ORIGINAL rows."""
    ns, n, m = d["n_struct"], d["n"], d["m"]
    A = sp.csc_matrix((d["vals"], d["row_idx"], d["col_ptr"]), shape=(m, n))
    As, Al = A[:, :ns], A[:, ns:].tocsc()
    xs = np.asarray(x)[:ns]
    worst = 0.0
    lo, hi = d["lb"][:ns], d["ub"][:ns]
    worst = max(worst, np.max(np.maximum(lo - xs, 0), initial=0), np.max(np.maximum(xs - hi, 0), initial=0))
    I = np.asarray(d["integer_cols"], int); I = I[I < ns]
    worst = max(worst, np.max(np.abs(xs[I] - np.round(xs[I])), initial=0))
    r = d["b"] - As @ xs  # must equal Al s for s in the logical box
    # each logical column: one entry; slack s_i = r_i / a
    smin = np.zeros(m); smax = np.zeros(m)
    for k in range(n - ns):
        s, e = Al.indptr[k], Al.indptr[k + 1]
        assert e - s == 1, "non-standard logical structure"
        i, a = Al.indices[s], Al.data[s]
        l, u = d["lb"][ns + k], d["ub"][ns + k]
        v = sorted([a * l, a * u]); smin[i] += v[0]; smax[i] += v[1]
    viol = np.maximum(smin - r, 0) + np.maximum(r - smax, 0)
    scale = 1.0 + np.abs(d["b"])
    return max(worst, float(np.max(viol / scale, initial=0)))
KEYS = ("MilpCoefTightenRuns", "MilpCoefTightenRows", "MilpCoefTightenCoefs")
nrows = 0
for p in sys.argv[3:]:
    d = pickle.load(open(p, "rb")); d.pop("had_debug_hook", None)
    name = os.path.basename(p)[:-4]
    t0 = time.perf_counter(); hst, hobj, hbd = highs_ref(d); ht = time.perf_counter() - t0
    row = {"inst": name, "m": d["m"], "ns": d["n_struct"], "ints": len(d["integer_cols"]),
           "highs": [hst, hobj, hbd, round(ht, 2)], "load1": os.getloadavg()[0]}
    kw = dict(d); kw["time_limit_s"] = T; kw["max_nodes"] = 10**9  # full run, not the captured probe budget
    if kw.get("initial_incumbent") is not None: kw["initial_incumbent"] = np.ascontiguousarray(kw["initial_incumbent"])
    for arm in ("0", "1"):
        os.environ["DISCOPT_RUST_MILP_COEF_TIGHTEN"] = arm
        rk = dict(kw); rk["max_nodes"] = 2; rk["time_limit_s"] = min(T, 60.0)
        rr = R.solve_milp_csc_py(**rk); row[f"root{arm}"] = rr[3]
        R.profile_reset_py()
        t0 = time.perf_counter()
        st, x, obj, bd, nodes, its = R.solve_milp_csc_py(**kw)
        w = time.perf_counter() - t0
        pc = R.profile_counters_py(); ctr = [pc.get(k, 0) for k in KEYS]
        v = verify(d, x) if np.isfinite(obj) and len(x) else None
        row[f"arm{arm}"] = {"status": st, "obj": obj, "bound": bd, "nodes": nodes, "wall": round(w, 2), "ctr": ctr, "viol": v}
        if arm == "0": assert ctr == [0, 0, 0], ctr
        else: assert ctr[0] >= 1, ctr
    # cert checks
    ref_ub = hobj if np.isfinite(hobj) else np.inf
    bad = []
    for arm in ("0", "1"):
        a = row[f"arm{arm}"]; tol = 1e-6 * max(1.0, abs(ref_ub)) if np.isfinite(ref_ub) else 0
        if np.isfinite(ref_ub) and a["bound"] > ref_ub + tol: bad.append(f"arm{arm} bound {a['bound']} > highs primal {ref_ub}")
        if np.isfinite(ref_ub) and row[f"root{arm}"] > ref_ub + tol: bad.append(f"root{arm} {row[f'root{arm}']} > highs primal {ref_ub}")
        if a["viol"] is not None and a["viol"] > 1e-6: bad.append(f"arm{arm} incumbent viol {a['viol']}")
        if hst == "Optimal" and a["status"] == "optimal" and abs(a["obj"] - hobj) > 1e-4 * max(1, abs(hobj)) + 1e-6:
            bad.append(f"arm{arm} optimal obj {a['obj']} vs highs {hobj}")
        if hst == "Infeasible" and np.isfinite(a["obj"]): bad.append(f"arm{arm} found point on infeasible")
    row["bad"] = bad
    out.write(json.dumps(row) + "\n"); out.flush(); nrows += 1
    a0, a1 = row["arm0"], row["arm1"]
    print(f"{name:45s} H {hst[:8]:8s} {hobj:12.5g} | root {row['root0']:12.6g} -> {row['root1']:12.6g} | OFF {a0['status'][:10]:10s} b {a0['bound']:12.6g} n {a0['nodes']:7d} {a0['wall']:6.2f}s | ON {a1['status'][:10]:10s} b {a1['bound']:12.6g} n {a1['nodes']:7d} {a1['wall']:6.2f}s ctr {a1['ctr']} {'BAD '+str(bad) if bad else ''}", flush=True)
print(f"PROBE_DONE rows={nrows}"); sys.exit(0 if nrows else 1)
