"""#2142 item 3, tangent entry experiment, step 1: capture cstr big-M node LPs.

Re-runs the deterministic 20k-node capture of `2142-propagation/p1_capture.py` and, for
every STRIDE-th node, stores the node LP exactly as the kernel assembled it (the
node_observer's `col_ptr/row_idx/vals/b/l/u/c`, standard form `[A | I] z = b`) next to
the box and the node bound.  Asserts the boxes equal the propagation study's capture
(REF npz), so both studies talk about the same nodes.

  [PROP_ROUNDS=R] DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u t1_capture_lp.py OUT.npz REF.npz MAX_NODES STRIDE   (REF '-' skips the box check)
"""
import os, sys, time
import numpy as np

WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R  # noqa: E402

assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT), (discopt.__file__, R.__file__)
assert hasattr(R, "propagate_spec_py"), "marker absent: not a #2142 build"
n_assert = 2
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

out, ref, max_nodes, stride = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SPEC_KEYS = ("n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
             "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
             "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
             "obbt_candidates").split()
LPK = ("col_ptr", "row_idx", "vals", "b", "l", "u", "c", "lo", "hi")
specs, recs = [], []


def obs(rec):
    if rec["id"] % stride == 0:
        r = {"call": len(specs) - 1, "id": rec["id"], "m": rec["m"], "n": rec["n"],
             "bound": rec["bound"], "obj": rec["obj"], "status": rec["status"], "cutoff": rec["cutoff"]}
        for k in LPK:
            r[k] = np.asarray(rec[k]).copy()
        recs.append(r)


_orig = R.solve_spatial_tree_py


def _wrapped(*a, **k):
    assert not a
    k["node_observer"] = obs
    if os.environ.get("PROP_ROUNDS"):
        k["propagation_rounds"] = int(os.environ["PROP_ROUNDS"])
    specs.append({kk: k[kk] for kk in SPEC_KEYS})
    return _orig(*a, **k)


R.solve_spatial_tree_py = _wrapped
model = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = model.solve(gdp_method="big-m", time_limit=1200, deterministic=True, max_nodes=max_nodes)
print("solve:", res.status, res.objective, res.bound, res.node_count,
      round(time.perf_counter() - t0, 1), "s; kernel calls", len(specs), "kept", len(recs), flush=True)
main = max(range(len(specs)), key=lambda q: sum(r["call"] == q for r in recs))
recs = [r for r in recs if r["call"] == main]
assert recs
n_assert += 1
ids = np.array([r["id"] for r in recs], dtype=float)
if ref != "-":
    refd = np.load(ref)
    assert np.array_equal(ids, refd["rec_id"]), "node ids differ from the propagation capture"
    for b, r in enumerate(recs):
        assert np.array_equal(r["lo"], refd["rec_lo"][b]) and np.array_equal(r["hi"], refd["rec_hi"][b])
        n_assert += 2
save = {k: np.asarray(v) for k, v in specs[main].items()}
save["rec_id"] = ids
for k in ("bound", "obj", "cutoff"):
    save["rec_" + k] = np.array([np.nan if r[k] is None else r[k] for r in recs], dtype=float)
save["rec_status"] = np.array([r["status"] for r in recs])
save["rec_m"] = np.array([r["m"] for r in recs])
save["rec_n"] = np.array([r["n"] for r in recs])
for k in LPK:
    arrs = [r[k] for r in recs]
    save["rec_%s_ptr" % k] = np.cumsum([0] + [len(a) for a in arrs])
    save["rec_%s_cat" % k] = np.concatenate(arrs)
np.savez(out, **save)
print("saved", out, "boxes", len(recs))
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert > 0 else 1)
