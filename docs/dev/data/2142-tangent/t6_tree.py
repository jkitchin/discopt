"""#2142: cstr big-M tree arm at a deterministic node budget with the kernel's
`propagation_rounds` overridden (injected by wrapping _rust.solve_spatial_tree_py).
DISCOPT_LP_DUAL_REBUILD must be absent (retired).
  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python python -u t6_tree.py ROUNDS MAX_NODES"""
import os, sys, time
assert "DISCOPT_LP_DUAL_REBUILD" not in os.environ
WT = os.environ["DISCOPT_TREE"]
import discopt, discopt._rust as R
assert discopt.__file__.startswith(WT) and R.__file__.startswith(WT) and hasattr(R, "solve_spatial_node_py"), R.__file__
from discopt.pyomo.gdp import convert_pyomo_gdp
import gdplib.cstr as cstr
rounds, max_nodes = int(sys.argv[1]), int(sys.argv[2])
calls = []
_orig = R.solve_spatial_tree_py
def _w(*a, **k):
    assert not a
    calls.append(k.get("propagation_rounds", "default(15)"))
    k["propagation_rounds"] = rounds
    return _orig(*a, **k)
R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method="big-m", time_limit=1800, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(f"ROUNDS={rounds} MAX_NODES={max_nodes} status={res.status} obj={res.objective} bound={res.bound!r} "
      f"nodes={res.node_count} wall={wall:.1f}s kernel_calls={len(calls)} passed={calls}", flush=True)
assert calls, "the kernel was never called: the arm measured nothing"
print("executed assertions: 2")
