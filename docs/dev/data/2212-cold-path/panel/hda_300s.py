import os, sys, time, json
from discopt.modeling.core import from_nl
from discopt._rust import profile_counters_py, profile_reset_py

profile_reset_py()
t = time.time()
r = from_nl("python/tests/data/minlplib_nl/hda.nl").solve(
    deterministic=True, max_nodes=int(sys.argv[1]), time_limit=300
)
w = time.time() - t
c = profile_counters_py()
keys = [
    k
    for k in c
    if k.startswith(
        (
            "LogicalDualCold",
            "LpVerdict",
            "DualColdFallbacks",
            "DualWarmSolves",
            "WarmVerdict",
            "Phase1Pivots",
        )
    )
]
print(
    json.dumps(
        {
            "arm": os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "0"),
            "nodes": r.node_count,
            "status": str(r.status),
            "bound": r.bound,
            "wall": round(w, 1),
            **{k: c[k] for k in keys if c[k]},
        }
    )
)
