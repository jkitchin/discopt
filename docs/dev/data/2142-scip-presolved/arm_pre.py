"""#2142: discopt's cstr homog-hull tree on SCIP's PRESOLVED model (translate.py). No solver
code change: a spy on solve_spatial_tree_py asserts the spec it receives is the captured one
and replaces it with the translation; every point handed back is expanded.

  DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python \
      python -u arm_pre.py {pre|base} MAX_NODES [TIME_LIMIT] [det|nodet]

Asserts the loaded tree (§8), that the intercepted spec equals the captured npz, that every
expanded incumbent satisfies the ORIGINAL spec; prints an executed-assertion count (§6)."""
import collections, os, sys, time
import numpy as np
os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = "1"
import discopt, discopt._rust as R
assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
n_assert = 1
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import translate as T  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

arm, max_nodes = sys.argv[1], int(sys.argv[2])
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0
det = (sys.argv[4] if len(sys.argv) > 4 else "det") == "det"
CAP = dict(np.load(T.SPEC))
STATS = collections.Counter()
calls = []
_o = R.solve_spatial_tree_py


def _w(**k):
    global n_assert
    expand = None
    if int(k["n_cols"]) == int(CAP["n_cols"]):
        for key, v in CAP.items():
            if key in ("global_lo", "global_hi") or key.startswith("meta_"):
                continue  # a probe call may hand a tightened box; the rest must match
            assert np.array_equal(np.asarray(k[key]), v), f"intercepted spec differs in {key}"
        n_assert += 1
        STATS["spec_matched"] += 1
        if arm == "pre":
            spec0 = {x: (int(k[x]) if x in ("n_cols", "n_orig") else np.asarray(k[x])) for x in CAP if not x.startswith("meta_")}
            spec, expand, compress, info, _ = T.build_spec(spec0)
            STATS.update({"rewritten": 1})
            if os.environ.get("BOX_DISCOPT_FBBT") == "1":
                kw0 = {x: (int(spec0[x]) if x in ("n_cols", "n_orig") else (np.asarray(spec0[x]).astype(np.int64) if np.asarray(spec0[x]).dtype.kind in "iub" else np.asarray(spec0[x], float))) for x in spec0}
                live, flo, fhi = R.propagate_spec_py(**kw0, lo=kw0["global_lo"].copy(), hi=kw0["global_hi"].copy(), cutoff=None, rounds=15)
                assert live
                n0 = int(spec0["n_cols"]); act = [j for j in range(n0) if spec["global_hi"][j] > spec["global_lo"][j] or j in ()]
                lo_, hi_ = spec["global_lo"].copy(), spec["global_hi"].copy()
                for j in range(n0):
                    if not (lo_[j] == 0.0 and hi_[j] == 0.0 and (spec0["global_lo"][j] != 0.0 or spec0["global_hi"][j] != 0.0)):
                        lo_[j], hi_[j] = flo[j], fhi[j]
                STATS["box_replaced"] = int(np.sum((lo_ != spec["global_lo"]) | (hi_ != spec["global_hi"])))
                spec["global_lo"], spec["global_hi"] = lo_, hi_
            if os.environ.get("ADD_MAPPED_ROWS") == "1":
                E = np.load(os.path.join(HERE, "rows_mapped.npz"))
                off = len(spec["fixed_cols"])
                spec["fixed_row_ptr"] = np.concatenate([spec["fixed_row_ptr"], off + E["ptr"][1:]]).astype(spec["fixed_row_ptr"].dtype)
                spec["fixed_cols"] = np.concatenate([spec["fixed_cols"], E["cols"]]).astype(spec["fixed_cols"].dtype)
                spec["fixed_coeffs"] = np.concatenate([spec["fixed_coeffs"], E["coefs"]])
                spec["fixed_rhs"] = np.concatenate([spec["fixed_rhs"], E["rhs"]])
                STATS["mapped_rows_added"] = int(len(E["rhs"]))
            for x in spec:
                k[x] = spec[x]
            for name in ("incumbent_value", "primal_hook"):
                if name in k and k[name] is not None:
                    f = k[name]
                    k[name] = (lambda f: (lambda x, *a: f(expand(x), *a)))(f)
    t = time.perf_counter()
    r = _o(**k)
    if expand is not None and r.get("incumbent_x") is not None and np.asarray(r["incumbent_x"]).size:
        y = expand(r["incumbent_x"])
        viol = T.residual({x: (int(CAP[x]) if CAP[x].ndim == 0 else CAP[x]) for x in CAP}, y)
        assert viol <= 1e-5, f"expanded incumbent violates the original spec by {viol}"; n_assert += 1
        r["incumbent_x"] = y
    calls.append(dict(max_nodes=k.get("max_nodes"), status=r["status"], bound=r["bound"], nodes=r["node_count"],
                      n_cols=int(k["n_cols"]), n_rows=int(len(k["fixed_rhs"])), wall=round(time.perf_counter() - t, 1)))
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
kw = dict(gdp_method="hull", time_limit=tl)
if det:
    kw.update(deterministic=True, max_nodes=max_nodes)
res = m.solve(**kw)
wall = time.perf_counter() - t0
print(f"ARM={arm} MAX_NODES={max_nodes} TL={tl} DET={det} status={res.status} obj={res.objective} bound={res.bound!r} "
      f"nodes={res.node_count} gap_certified={getattr(res, 'gap_certified', None)} wall={wall:.1f}s", flush=True)
print("stats:", dict(STATS))
for c_ in calls:
    print("  kernel:", c_, flush=True)
ctr = dict(R.profile_counters_py()) if os.environ.get("DISCOPT_PROFILE") == "1" else {}
print("counters:", {k: v for k, v in ctr.items() if k in ("LpVerdictNumerical", "LpInfeasUncertified", "DualProofLearned", "DualProofFathoms")})
assert STATS["spec_matched"] >= 1, "the captured spec never reached the kernel"; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
if res.objective is not None:
    assert res.objective >= 3.0620073 - 1e-5, "incumbent below the oracle"; n_assert += 1
print(f"RESULT arm={arm} nodes={max_nodes} bound={res.bound!r} status={res.status} wall={wall:.1f}")
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
