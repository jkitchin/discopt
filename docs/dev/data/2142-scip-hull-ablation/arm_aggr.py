"""#2142 SCIP-hull-ablation entry experiment: discopt's cstr homog-hull tree with the kernel
spec AGGREGATED the way SCIP's presolve aggregates it (doubleton equalities -> one column),
derived from the spec alone (aggr_derive.py -> aggr.npz). No solver code change: a spy on
solve_spatial_tree_py rewrites the spec and expands every point handed back.

  DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python \
      python -u arm_aggr.py {aggr|base} MAX_NODES

Asserts the loaded extension is this tree's build (CLAUDE.md §8), that every expanded
incumbent satisfies the ORIGINAL spec, and prints an executed-assertion count (§6)."""
import collections, os, sys, time
import numpy as np
import discopt, discopt._rust as R
assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
n_assert = 1
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2142-exact-hull"))
import exact_hull_proto as P  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

arm, max_nodes = sys.argv[1], int(sys.argv[2])
P.install("homog")
AG = np.load(os.path.join(HERE, "aggr.npz"))
agg = {int(j): (int(k), float(a), float(b)) for j, k, a, b in zip(AG["j"], AG["k"], AG["a"], AG["b"])}
STATS = collections.Counter()


def rewrite(k):
    """Return (new kwargs, expand) where expand maps a kernel point back to all columns."""
    n = int(k["n_cols"]); n_orig = int(k["n_orig"])
    c = np.array(k["c"], dtype=float); integ = np.array(k["integrality"]).astype(np.int64)
    lo = np.array(k["global_lo"], dtype=float); hi = np.array(k["global_hi"], dtype=float)
    tk, ti, tj, tout = (np.array(k[x]).astype(np.int64) for x in ("term_kind", "term_i", "term_j", "term_out"))
    tp = np.array(k["term_p"]).astype(np.int64); tco = np.array(k["term_coeff"], dtype=float); tcs = np.array(k["term_cst"], dtype=float)
    bw = np.array(k["blf_w"]).astype(np.int64)
    # which aggregations the spec encoding can express
    ok = {}
    for j, (kk, a, b) in agg.items():
        ident = (a == 1.0 and b == 0.0)
        if c[j] != 0.0 and b != 0.0: STATS["skip_obj"] += 1; continue
        bad = False
        for t in range(len(tk)):
            if tout[t] == j and not ident: bad = True
            if ti[t] == j and tk[t] in (0, 1) and not ident: bad = True
            if tj[t] == j and tk[t] == 0 and not ident: bad = True
        if j in set(bw.tolist()) and not ident: bad = True
        if bad: STATS["skip_term"] += 1; continue
        ok[j] = (kk, a, b)
    STATS["aggregated"] = len(ok)
    assert all(kk not in ok for kk, _, _ in ok.values()), "representative is itself aggregated"
    # objective
    for j, (kk, a, b) in ok.items():
        c[kk] += a * c[j]; c[j] = 0.0
    # bounds of j -> representative
    for j, (kk, a, b) in ok.items():
        l1, l2 = (lo[j] - b) / a, (hi[j] - b) / a
        lo[kk] = max(lo[kk], min(l1, l2)); hi[kk] = min(hi[kk], max(l1, l2))
        if integ[j] and not integ[kk]: integ[kk] = 1
        lo[j] = hi[j] = 0.0; integ[j] = 0
        assert lo[kk] <= hi[kk] + 1e-9, (j, kk, lo[kk], hi[kk])
    # fixed rows
    ptr, cols, coefs, rhs = (np.array(k[x]) for x in ("fixed_row_ptr", "fixed_cols", "fixed_coeffs", "fixed_rhs"))
    nptr, ncols, ncoefs, nrhs = [0], [], [], []
    for r in range(len(rhs)):
        row, b = {}, float(rhs[r])
        for q in range(int(ptr[r]), int(ptr[r + 1])):
            j, a = int(cols[q]), float(coefs[q])
            if j in ok:
                kk, s, t = ok[j]; b -= a * t; row[kk] = row.get(kk, 0.0) + a * s
            else:
                row[j] = row.get(j, 0.0) + a
        row = {j: a for j, a in row.items() if abs(a) > 1e-13}
        if not row:
            assert b >= -1e-9, ("row became 0 <= negative", r, b); STATS["rows_dropped"] += 1; continue
        ncols += list(row); ncoefs += list(row.values()); nrhs.append(b); nptr.append(len(ncols))
    k["fixed_row_ptr"] = np.array(nptr, dtype=ptr.dtype); k["fixed_cols"] = np.array(ncols, dtype=cols.dtype)
    k["fixed_coeffs"] = np.array(ncoefs, dtype=float); k["fixed_rhs"] = np.array(nrhs, dtype=float)
    # terms (kind 1 inputs/outputs only under identity; kinds 2/3/4 affine inputs)
    for t in range(len(tk)):
        for arr in (ti, tj, tout):
            if int(arr[t]) in ok:
                kk, a, b = ok[int(arr[t])]
                if arr is ti and tk[t] in (2, 3, 4):
                    tcs[t] += tco[t] * b; tco[t] *= a
                arr[t] = kk
    k["term_i"], k["term_j"], k["term_out"], k["term_coeff"], k["term_cst"] = ti, tj, tout, tco, tcs
    # BLF forms
    for side in ("a", "b"):
        p_, c_, co_, cst_ = (np.array(k[f"blf_{side}_{x}"]) for x in ("ptr", "cols", "coeffs", "const"))
        np_, nc_, nco_, ncst_ = [0], [], [], []
        for t in range(len(cst_)):
            form, cst = {}, float(cst_[t])
            for q in range(int(p_[t]), int(p_[t + 1])):
                j, a = int(c_[q]), float(co_[q])
                if j in ok:
                    kk, s, b = ok[j]; cst += a * b; form[kk] = form.get(kk, 0.0) + a * s
                else:
                    form[j] = form.get(j, 0.0) + a
            form = {j: a for j, a in form.items() if abs(a) > 1e-13}
            nc_ += list(form); nco_ += list(form.values()); ncst_.append(cst); np_.append(len(nc_))
        k[f"blf_{side}_ptr"] = np.array(np_, dtype=p_.dtype); k[f"blf_{side}_cols"] = np.array(nc_, dtype=c_.dtype)
        k[f"blf_{side}_coeffs"] = np.array(nco_, dtype=float); k[f"blf_{side}_const"] = np.array(ncst_, dtype=float)
    for t in range(len(bw)):
        if int(bw[t]) in ok: bw[t] = ok[int(bw[t])][0]
    k["blf_w"] = bw
    k["c"] = c; k["integrality"] = integ; k["global_lo"] = lo; k["global_hi"] = hi
    k["obbt_candidates"] = np.array([j for j in np.array(k["obbt_candidates"]).tolist() if int(j) not in ok], dtype=np.array(k["obbt_candidates"]).dtype)

    def expand(x):
        x = np.array(x, dtype=float)
        if x.size != n: return x
        for j, (kk, a, b) in ok.items(): x[j] = a * x[kk] + b
        return x
    return k, expand


def check_original(spec0, x, tol=1e-6):
    """Max violation of the ORIGINAL spec's fixed rows and term/BLF definitions at x."""
    ptr, cols, coefs, rhs = (np.array(spec0[k]) for k in ("fixed_row_ptr", "fixed_cols", "fixed_coeffs", "fixed_rhs"))
    v = 0.0
    for r in range(len(rhs)):
        v = max(v, float(sum(coefs[q] * x[cols[q]] for q in range(ptr[r], ptr[r + 1])) - rhs[r]))
    tk, ti, tout, tp = (np.array(spec0[k]) for k in ("term_kind", "term_i", "term_out", "term_p"))
    for t in range(len(tk)):
        assert tk[t] == 1, tk[t]
        v = max(v, abs(x[tout[t]] - x[ti[t]] ** tp[t]))
    for side in ("a", "b"):
        pass
    bw = np.array(spec0["blf_w"])
    for t in range(len(bw)):
        A = spec0["blf_a_const"][t] + sum(spec0["blf_a_coeffs"][q] * x[spec0["blf_a_cols"][q]] for q in range(spec0["blf_a_ptr"][t], spec0["blf_a_ptr"][t + 1]))
        B = spec0["blf_b_const"][t] + sum(spec0["blf_b_coeffs"][q] * x[spec0["blf_b_cols"][q]] for q in range(spec0["blf_b_ptr"][t], spec0["blf_b_ptr"][t + 1]))
        v = max(v, abs(x[bw[t]] - A * B))
    return v


calls, recs = [], []
_o = R.solve_spatial_tree_py


def obs(rec):
    recs.append((rec["id"], rec.get("bound"), str(rec.get("status"))))


def _w(**k):
    global n_assert
    if int(k.get("max_nodes", 0)) > 1 and "node_observer" not in k:
        k["node_observer"] = obs
    spec0 = {x: np.array(k[x]) for x in k if isinstance(k[x], np.ndarray)}
    expand = None
    if arm == "aggr":
        k, expand = rewrite(dict(k))
        for name in ("incumbent_value", "primal_hook"):
            if name in k and k[name] is not None:
                f = k[name]
                k[name] = (lambda f: (lambda x, *a: f(expand(x), *a)))(f)
    t = time.perf_counter()
    r = _o(**k)
    if expand is not None and r.get("incumbent_x") is not None and np.asarray(r["incumbent_x"]).size:
        x = expand(r["incumbent_x"])
        viol = check_original(spec0, x)
        assert viol <= 1e-6, f"expanded incumbent violates the original spec by {viol}"; n_assert += 1
        r["incumbent_x"] = x
        STATS["incumbent_checked"] += 1
    calls.append(dict(max_nodes=k.get("max_nodes"), status=r["status"], bound=r["bound"], nodes=r["node_count"],
                      undecided=r.get("n_undecided"), n_cols=int(k["n_cols"]), n_rows=int(len(k["fixed_rhs"])), wall=round(time.perf_counter() - t, 1)))
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="hull", time_limit=7200, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(f"ARM={arm} MAX_NODES={max_nodes} status={res.status} obj={res.objective} bound={res.bound!r} nodes={res.node_count} "
      f"gap_certified={getattr(res, 'gap_certified', None)} wall={wall:.1f}s", flush=True)
print("rewrite stats:", dict(STATS))
for c_ in calls: print("  kernel:", c_, flush=True)
if recs:
    print("node status histogram:", dict(collections.Counter(r[2] for r in recs)))
ctr = dict(R.profile_counters_py()) if os.environ.get("DISCOPT_PROFILE") == "1" else {}
print("counters:", {k: v for k, v in ctr.items() if k in ("LpVerdictNumerical", "LpInfeasUncertified", "LpVerdictInfeasible", "LpVerdictOptimal", "FarkasRejectMargin", "DualProofLearned", "DualProofTightenings", "DualProofFathoms")})
assert calls, "the kernel never ran"; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
if res.objective is not None:
    assert res.objective >= 3.0620073 - 1e-5, "incumbent below the oracle"; n_assert += 1
assert P.STATS["rows"] == 5, P.STATS; n_assert += 1
print(f"RESULT arm={arm} nodes={max_nodes} bound={res.bound!r} status={res.status} wall={wall:.1f} numerical={ctr.get('LpVerdictNumerical')}")
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
