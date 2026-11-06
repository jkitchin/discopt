"""#2142 hull-mixed entry, part 1: GDPlib cstr bound at a deterministic node budget.

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u e1_arms.py ARM MAX_NODES
ARM: hull (flag OFF), mixed (hull, DISCOPT_HULL_MIXED_BIGM=1), bigm (flag OFF).
The flag is set in this process's environment before discopt is imported.
Prints an executed-assertion count (CLAUDE.md §6)."""
import os, sys, time
arm, max_nodes = sys.argv[1], int(sys.argv[2])
assert "DISCOPT_HULL_MIXED_BIGM" not in os.environ
os.environ["DISCOPT_HULL_MIXED_BIGM"] = "1" if arm == "mixed" else "0"
os.environ.setdefault("PYTHONHASHSEED", "0")
WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R
import discopt._relax.gdp_reformulate as G
assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT), (discopt.__file__, R.__file__)
assert hasattr(G, "_hull_mixed_big_m_rows"), "marker absent"
n_assert = 2
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
method = "big-m" if arm == "bigm" else "hull"
calls = []
_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    calls.append(k.get("max_nodes"))
    return _orig(*a, **k)
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
G._HULL_MIXED_STATS.update(big_m=0, perspective=0)
t0 = time.perf_counter()
res = m.solve(gdp_method=method, time_limit=7200, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(f"ARM={arm} MAX_NODES={max_nodes} status={res.status} obj={res.objective} bound={res.bound!r} "
      f"gap_certified={getattr(res, 'gap_certified', None)} nodes={res.node_count} wall={wall:.1f}s "
      f"mixed_stats={G._HULL_MIXED_STATS} kernel_calls={calls}", flush=True)
if arm == "mixed":
    assert G._HULL_MIXED_STATS["big_m"] > 0, "mixed branch never fired"
else:
    assert G._HULL_MIXED_STATS["big_m"] == 0
n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6 * 3.07, "bound above the oracle"
n_assert += 1
print(f"executed assertions: {n_assert}")
