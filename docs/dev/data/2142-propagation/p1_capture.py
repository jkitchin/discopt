"""#2142 item 3 -- capture cstr big-M node boxes (pre- and post-propagation + the cutoff
propagation ran under) through the kernel's node_observer, on the WORKTREE build.

  PYTHONPATH=<wt>/python python -u p1_capture.py OUT.npz MAX_NODES STRIDE

Saves the spec the kernel was actually called with plus every STRIDE-th observed node.
Prints an executed-assertion count (CLAUDE.md §6); observer errors propagate (§7)."""
import os, sys, time
import numpy as np

# The off20k_* logs were captured with DISCOPT_LP_DUAL_REBUILD=0 (retired); rebuild10k_* with =1.
WT = os.environ["DISCOPT_TREE"]  # the tree under test (CLAUDE.md §8)
import discopt, discopt._rust as R  # noqa: E402

assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT), (discopt.__file__, R.__file__)
assert hasattr(R, "propagate_spec_py"), "marker absent: not the #2142 propagation build"
n_assert = 2
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

out, max_nodes, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SPEC_KEYS = ("n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
             "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
             "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
             "obbt_candidates").split()
specs, recs = [], []
n_seen = [0]


def obs(rec):
    n_seen[0] += 1
    if rec["id"] % stride == 0 and "pre_lo" in rec:
        recs.append({"call": len(specs) - 1, **{k: (np.asarray(rec[k]).copy() if k in ("lo", "hi", "pre_lo", "pre_hi") else rec[k])
                     for k in ("id", "parent", "lo", "hi", "pre_lo", "pre_hi", "cutoff", "bound", "status")}})


_orig = R.solve_spatial_tree_py


def _wrapped(*a, **k):
    assert not a, "positional kernel call: adapt the capture"
    k["node_observer"] = obs
    specs.append({kk: k[kk] for kk in SPEC_KEYS})
    specs[-1]["_cfg"] = {kk: k[kk] for kk in k if kk not in SPEC_KEYS and not callable(k[kk]) and not hasattr(k[kk], "shape")}
    return _orig(*a, **k)


R.solve_spatial_tree_py = _wrapped
model = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = model.solve(gdp_method="big-m", time_limit=600, deterministic=True, max_nodes=max_nodes)
print("solve:", res.status, res.objective, res.bound, res.node_count, round(time.perf_counter() - t0, 1),
      "s; kernel calls", len(specs), "observed", n_seen[0], "kept", len(recs), flush=True)
assert len(recs) > 0
n_assert += 1
for q, s_ in enumerate(specs):
    print("kernel call", q, "n_cols", s_["n_cols"], "records", sum(r["call"] == q for r in recs), "cfg:", s_["_cfg"])
main = max(range(len(specs)), key=lambda q: sum(r["call"] == q for r in recs))
recs = [r for r in recs if r["call"] == main]
sp = {k: np.asarray(v) for k, v in specs[main].items() if k != "_cfg"}
save = dict(sp)
for k in ("id", "cutoff", "bound"):
    save["rec_" + k] = np.array([np.nan if r[k] is None else r[k] for r in recs], dtype=float)
for k in ("lo", "hi", "pre_lo", "pre_hi"):
    save["rec_" + k] = np.stack([r[k] for r in recs])
save["prop_rounds"] = np.array(specs[main]["_cfg"].get("propagation_rounds", 15))
np.savez(out, **save)
print("saved", out, "boxes", len(recs), "cutoffs finite", int(np.isfinite(save["rec_cutoff"]).sum()))
n_assert += 1
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert > 0 else 1)
