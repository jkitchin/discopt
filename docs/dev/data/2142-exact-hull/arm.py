"""#2142 exact-hull entry experiment: one deterministic GDPlib cstr solve per arm.

  DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python \
      python -u arm.py ARM MAX_NODES [TIME_LIMIT] [det|nodet]

ARM: hull (shipped eps-perspective) | bigm (shipped default) | ratio | homog | lifted (the
prototype lowerings of exact_hull_proto.py, installed by monkeypatch before the solve) |
flag (the shipped implementation: hull with DISCOPT_HULL_EXACT_PERSPECTIVE=1, no monkeypatch).
Records every kernel call (status, bound, nodes, undecided, wall), the largest |global bound|
of the spatial spec the kernel was handed (the §114 magnitude), the node-status histogram of
the main tree call, and the LP-verdict counters. Asserts the loaded extension is this tree's
build (CLAUDE.md §8) and prints an executed-assertion count (§6)."""
import collections
import os
import sys
import time

import numpy as np

import discopt
import discopt._rust as R

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert b"DISCOPT_INT_ROUND_EXACT" not in open(R.__file__, "rb").read(), "stale _rust (§114 flag)"
n_assert = 2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_hull_proto as P  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

arm = sys.argv[1]
max_nodes = int(sys.argv[2])
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0
det = (sys.argv[4] if len(sys.argv) > 4 else "det") == "det"
method = {"hull": "hull", "bigm": "big-m"}.get(arm, "hull")
if arm in ("ratio", "homog", "lifted", "both"):
    P.install(arm)
if arm == "flag":
    os.environ["DISCOPT_HULL_EXACT_PERSPECTIVE"] = "1"
else:
    os.environ.pop("DISCOPT_HULL_EXACT_PERSPECTIVE", None)
import discopt._relax.gdp_reformulate as G  # noqa: E402

calls, recs = [], []
spec_mag = [0.0]
_o = R.solve_spatial_tree_py


def obs(rec):
    recs.append((rec["id"], rec.get("bound"), str(rec.get("status"))))


def _w(**k):
    lo, hi = np.asarray(k["global_lo"]), np.asarray(k["global_hi"])
    fin = np.isfinite(lo) & np.isfinite(hi)
    spec_mag[0] = max(spec_mag[0], float(np.max(np.abs(np.concatenate([lo[fin], hi[fin]])))))
    if int(k.get("max_nodes", 0)) > 1 and "node_observer" not in k:
        k["node_observer"] = obs
    t = time.perf_counter()
    r = _o(**k)
    calls.append(dict(max_nodes=k.get("max_nodes"), status=r["status"], bound=r["bound"],
                      nodes=r["node_count"], undecided=r.get("n_undecided"),
                      n_cols=int(k["n_cols"]), wall=round(time.perf_counter() - t, 1)))
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
kw = dict(gdp_method=method, time_limit=tl)
if det:
    kw.update(deterministic=True, max_nodes=max_nodes)
res = m.solve(**kw)
wall = time.perf_counter() - t0
print(f"ARM={arm} METHOD={method} DET={det} MAX_NODES={max_nodes} TL={tl} status={res.status} "
      f"obj={res.objective} bound={res.bound!r} nodes={res.node_count} "
      f"gap_certified={getattr(res, 'gap_certified', None)} wall={wall:.1f}s", flush=True)
print(f"proto stats: {P.STATS}")
print(f"spec max |global bound|: {spec_mag[0]:.3e}")
for c in calls:
    print("  kernel:", c, flush=True)
if recs:
    st = collections.Counter(r[2] for r in recs)
    print("node status histogram:", dict(st))
ctr = dict(R.profile_counters_py()) if os.environ.get("DISCOPT_PROFILE") == "1" else {}
print("counters:", {k: v for k, v in ctr.items() if k in (
    "LpVerdictNumerical", "LpInfeasUncertified", "LpVerdictInfeasible", "LpVerdictOptimal",
    "FarkasRejectMargin", "DualProofLearned", "DualProofTightenings", "DualProofFathoms")})
assert calls, "the kernel never ran"; n_assert += 1
assert res.bound is None or res.bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
if arm in ("homog", "lifted", "both"):
    assert P.STATS["rows"] == 5, P.STATS; n_assert += 1
if arm == "flag":
    print(f"exact-hull stats: {G._HULL_EXACT_STATS}")
    assert G._HULL_EXACT_STATS["homogenized"] == 5, G._HULL_EXACT_STATS; n_assert += 1
else:
    assert sum(G._HULL_EXACT_STATS.values()) == 0, G._HULL_EXACT_STATS; n_assert += 1
if res.objective is not None:
    assert res.objective >= 3.0620073 - 1e-5, "incumbent below the oracle"; n_assert += 1
print(f"RESULT arm={arm} nodes={max_nodes} bound={res.bound!r} status={res.status} "
      f"wall={wall:.1f} spec_mag={spec_mag[0]:.3e} numerical={ctr.get('LpVerdictNumerical')}")
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
