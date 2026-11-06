"""#2142 branching variable (performance-plan 108) -- cstr big-M tree arms for the
kernel's violation measure.

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u v3_arms.py ARM MAX_NODES

Kwargs are injected into _rust.solve_spatial_tree_py on the default path
(deterministic node budget). Prints an executed-assertion count."""
import os, sys, time
assert "DISCOPT_SPATIAL_VIOL_SCIP" not in os.environ
os.environ["DISCOPT_PROFILE"] = "1"
WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R
assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT), R.__file__
assert "spatial_viol_scip" in R.solve_spatial_tree_py.__text_signature__, "marker absent"
n_assert = 2
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
ARMS = {
    "base": {},
    "viol": {"spatial_viol_scip": True},
    "viol_nopc": {"spatial_viol_scip": True, "spatial_pscost_weight": 0.0},
    "nopc": {"spatial_pscost_weight": 0.0},
}
arm, max_nodes = sys.argv[1], int(sys.argv[2])
inj = ARMS[arm]
calls = []
_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    assert not a
    k.update(inj)
    calls.append({kk: k.get(kk) for kk in ("spatial_score", "spatial_pscost_weight", "spatial_midpull", "spatial_viol_scip", "max_nodes")})
    return _orig(*a, **k)
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="big-m", time_limit=7200, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
prof = R.profile_counters_py()
ctr = {k: v for k, v in dict(prof).items() if "SpatialViol" in str(k)}
print(f"ARM={arm} MAX_NODES={max_nodes} status={res.status} obj={res.objective} bound={res.bound!r} "
      f"gap_certified={getattr(res, 'gap_certified', None)} nodes={res.node_count} wall={wall:.1f}s ctr={ctr} calls={calls}", flush=True)
assert calls; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6 * 3.07, "bound above the oracle"
n_assert += 1
print(f"executed assertions: {n_assert}")
