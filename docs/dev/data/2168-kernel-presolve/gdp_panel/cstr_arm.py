"""#2168 full-solve entry: GDPlib cstr under gdp_method="hull", DISCOPT_KERNEL_PRESOLVE OFF/ON.

  PYTHONHASHSEED=0 PYTHONPATH=<worktree>/python python -u cstr_arm.py on|off MAX_NODES TL det|nodet

Asserts the loaded discopt is the worktree's and carries the presolve (CLAUDE.md §8), records
every kernel call, checks the oracle bounds, prints an executed-assertion count (§6)."""
import os
import sys
import time

arm, max_nodes, tl, det = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "det"
os.environ["DISCOPT_KERNEL_PRESOLVE"] = "1" if arm == "on" else "0"
import discopt  # noqa: E402
import discopt._rust as R  # noqa: E402
from discopt import solver as S  # noqa: E402

WT = os.environ["DISCOPT_WT"]
assert discopt.__file__.startswith(WT), discopt.__file__
assert R.__file__.startswith(WT), R.__file__
assert hasattr(S, "KERNEL_PRESOLVE_STATS")
n_assert = 3
import gdplib.cstr as cstr  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

calls = []
_o = R.solve_spatial_tree_py


def _w(**k):
    t = time.perf_counter()
    r = _o(**k)
    calls.append(dict(max_nodes=k.get("max_nodes"), status=r["status"], bound=r["bound"],
                      nodes=r["node_count"], wall=round(time.perf_counter() - t, 1)))
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
kw = dict(gdp_method="hull", time_limit=tl)
if det:
    kw["deterministic"] = True
if max_nodes > 0:
    kw["max_nodes"] = max_nodes
t0 = time.perf_counter()
res = m.solve(**kw)
wall = time.perf_counter() - t0
for c in calls:
    print("  kernel:", c, flush=True)
kps = dict(S.KERNEL_PRESOLVE_STATS)
print("presolve stats:", kps)
assert calls, "the kernel never ran"; n_assert += 1
if arm == "on":
    assert kps.get("applied", 0) >= 1, kps; n_assert += 1
else:
    assert kps.get("calls", 0) == 0, kps; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
if res.objective is not None:
    assert res.objective >= 3.0620073 - 1e-5, "incumbent below the oracle"; n_assert += 1
print(f"RESULT arm={arm} det={det} max_nodes={max_nodes} tl={tl} status={res.status} "
      f"obj={res.objective} bound={res.bound!r} nodes={res.node_count} "
      f"gap_certified={getattr(res, 'gap_certified', None)} wall={wall:.1f}", flush=True)
print(f"executed assertions: {n_assert}", flush=True)
sys.stdout.flush()
os._exit(0 if n_assert else 1)
