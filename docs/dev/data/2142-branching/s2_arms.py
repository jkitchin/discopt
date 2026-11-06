"""#2142 item 3 step 2 -- cstr big-M tree arms over the kernel's spatial split-point knobs.

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u s2_arms.py ARM MAX_NODES

Kwargs are injected into _rust.solve_spatial_tree_py (deterministic node budget)."""
import os, sys, time
assert "DISCOPT_LP_DUAL_REBUILD" not in os.environ
assert "DISCOPT_SPATIAL_BRANCH_CLAMP" not in os.environ
WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R
assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT), R.__file__
assert "spatial_clamp" in R.solve_spatial_tree_py.__text_signature__, "marker absent: not the #2142 branch-point build"
n_assert = 2
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
ARMS = {
    "base": {},
    "clamp": {"spatial_clamp": 0.2},
    "trig": {"spatial_midpull_reldomtrig": 0.5},
    "scip": {"spatial_clamp": 0.2, "spatial_midpull_reldomtrig": 0.5},
    "scip_sf": {"spatial_clamp": 0.2, "spatial_midpull_reldomtrig": 0.5, "scale_free_branching": True},
    "lp_clamp": {"spatial_midpull": 0.0, "spatial_clamp": 0.2},
    "sf": {"scale_free_branching": True},
}
arm, max_nodes = sys.argv[1], int(sys.argv[2])
inj = ARMS[arm]
calls = []
_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    assert not a
    k.update(inj)
    calls.append({kk: k.get(kk) for kk in ("spatial_midpull", "spatial_clamp", "spatial_midpull_reldomtrig", "scale_free_branching", "spatial_score", "max_nodes")})
    return _orig(*a, **k)
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="big-m", time_limit=7200, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(f"ARM={arm} MAX_NODES={max_nodes} status={res.status} obj={res.objective} bound={res.bound!r} "
      f"gap_certified={getattr(res, 'gap_certified', None)} nodes={res.node_count} wall={wall:.1f}s calls={calls}", flush=True)
assert calls; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6 * 3.07, "bound above the oracle"
n_assert += 1
print(f"executed assertions: {n_assert}")
