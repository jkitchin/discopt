"""#2142 item 3 step 1 -- where does the cstr big-M tree split? (bound-neutral observer)

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u s1_splits.py MAX_NODES OUT.json

Reconstructs every branching decision from the node_observer stream: a child's
pre-propagation box is its parent's post-propagation box with ONE coordinate
overridden (QNode::into_box), so (col, value) is recovered exactly. A split is seen
iff at least one child was popped and solved; the coverage is printed.
Relative position = (p - lo)/(hi - lo) on the parent's post-propagation box; relative
width = (hi - lo) / (root post-propagation width). Prints an executed-assertion count."""
import json, os, sys, time
import numpy as np
assert "DISCOPT_LP_DUAL_REBUILD" not in os.environ
WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R
assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT), (discopt.__file__, R.__file__)
assert hasattr(R, "propagate_spec_py"), "marker absent"
n_assert = 1
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
max_nodes, out = int(sys.argv[1]), sys.argv[2]
calls = []  # per kernel call: dict(cfg, integrality, recs)

def make_obs(store):
    def obs(rec):
        store[rec["id"]] = (rec["parent"], np.asarray(rec["lo"]).copy(), np.asarray(rec["hi"]).copy(),
                            None if "pre_lo" not in rec else np.asarray(rec["pre_lo"]).copy(),
                            None if "pre_hi" not in rec else np.asarray(rec["pre_hi"]).copy(), rec["status"])
    return obs

_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    assert not a
    store = {}
    k["node_observer"] = make_obs(store)
    cfg = {kk: v for kk, v in k.items() if isinstance(v, (int, float, bool, str)) or v is None}
    calls.append({"cfg": cfg, "integrality": np.asarray(k["integrality"]).copy(), "n_orig": k["n_orig"], "recs": store})
    return _orig(*a, **k)
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="big-m", time_limit=1800, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(f"solve status={res.status} obj={res.objective} bound={res.bound!r} nodes={res.node_count} wall={wall:.1f}s kernel_calls={len(calls)}", flush=True)
assert calls; n_assert += 1
main = max(calls, key=lambda c: len(c["recs"]))
print("kernel cfg (scalars):", {k: v for k, v in main["cfg"].items() if k in ("spatial_score", "spatial_midpull", "spatial_pscost_weight", "scale_free_branching", "pseudocost_branching", "propagation_rounds", "max_nodes")})
recs, integ = main["recs"], main["integrality"]
root = recs[1]
root_w = root[2] - root[1]
splits = {}  # (parent, col) -> (value, parent lo, parent hi)
n_child, n_multi, n_zero = 0, 0, 0
for nid, (par, lo, hi, plo, phi, st) in recs.items():
    if par is None:
        continue
    if plo is None:
        raise SystemExit("no pre box: propagation off?")
    P = recs[par]
    d_lo = np.nonzero(plo != P[1])[0]; d_hi = np.nonzero(phi != P[2])[0]
    diff = [(c, "lo") for c in d_lo] + [(c, "hi") for c in d_hi]
    n_child += 1
    if len(diff) != 1:
        if len(diff) == 0:
            n_zero += 1
        else:
            n_multi += 1
        continue
    c, side = diff[0]
    v = plo[c] if side == "lo" else phi[c]
    splits.setdefault((par, int(c)), (float(v), float(P[1][c]), float(P[2][c]), side))
n_assert += 1
assert n_multi == 0, f"{n_multi} children differ from their parent in >1 coordinate: model of into_box wrong"
n_assert += 1
n_parents_branched = len({p for (p, _) in splits})
n_nodes = len(recs)
print(f"observed nodes {n_nodes}; children {n_child}; zero-diff children {n_zero}; splits recovered {len(splits)}")
rows = []
for (par, c), (v, lo, hi, side) in splits.items():
    is_int = bool(integ[c])
    w = hi - lo
    pos = (v - lo) / w if w > 0 else float("nan")
    if is_int:
        # integer child bound is floor or ceil; the branch point was floor+0.5
        pos = ((np.floor(v) + 0.5 if side == "hi" else np.ceil(v) - 0.5) - lo) / w if w > 0 else float("nan")
    rw = w / root_w[c] if np.isfinite(root_w[c]) and root_w[c] > 0 else float("nan")
    rows.append({"parent": par, "col": c, "int": is_int, "orig": bool(c < main["n_orig"]), "pos": pos, "relw": rw, "w": w})
sp = [r for r in rows if not r["int"]]
it = [r for r in rows if r["int"]]
print(f"splits: spatial {len(sp)}, integer {len(it)}")
assert len(sp) > 0; n_assert += 1
pos = np.array([r["pos"] for r in sp]); edge = np.minimum(pos, 1 - pos)
relw = np.array([r["relw"] for r in sp])
hist = np.histogram(pos, bins=np.linspace(0, 1, 21))[0]
summary = {
    "max_nodes": max_nodes, "bound": res.bound, "nodes": res.node_count,
    "n_spatial_splits": len(sp), "n_integer_splits": len(it), "children_observed": n_child,
    "frac_edge_1pct": float(np.mean(edge < 0.01)), "frac_edge_5pct": float(np.mean(edge < 0.05)),
    "frac_edge_10pct": float(np.mean(edge < 0.10)), "frac_edge_20pct": float(np.mean(edge < 0.20)),
    "frac_edge_lt_0375": float(np.mean(edge < 0.375 - 1e-9)),
    "pos_hist_20bins": hist.tolist(),
    "relw_quantiles": {q: float(np.nanquantile(relw, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    "relw_hist_decades": np.histogram(np.log10(np.clip(relw[np.isfinite(relw)], 1e-12, None)), bins=list(range(-12, 2)))[0].tolist(),
    "cols_split_top": sorted(((int(c), int(n)) for c, n in zip(*np.unique([r["col"] for r in sp], return_counts=True))), key=lambda t: -t[1])[:15],
    "orig_share": float(np.mean([r["orig"] for r in sp])),
}
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "rows": rows}, open(out, "w"))
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
