"""#2182 task 2, H2 entry experiment: ski-rental the root share of the GDP config
constructor (root call gets min(_GDP_CONFIG_DEFER_ROOT_S, time already spent)).

One solve per process: python -u h2_ab.py <instance> <method> <arm OFF|ON>
The driver (h2_drive.sh) interleaves arms. ON patches discopt.solver's module
global, which both call sites look up at call time. Prints one JSON row.
"""

import json
import sys
import time
import warnings

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import discopt  # noqa: E402
import discopt.solver as S  # noqa: E402
from benchmarks import pyomo_gdp_examples as px  # noqa: E402

from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

assert discopt.__file__.startswith("/home/user/discopt/python/discopt"), discopt.__file__
name, method, arm = sys.argv[1:4]
assert arm in ("OFF", "ON")
fired = {"calls": 0, "shrunk": 0, "root_s": []}
_orig = S._gdp_config_root_until


def h2_root_until(full_deadline, now, solve_start):
    fired["calls"] += 1
    base = _orig(full_deadline, now, solve_start)
    if not S._gdp_config_defer_enabled() or base >= full_deadline:
        fired["root_s"].append(round(base - now, 3))
        return base
    out = min(base, now + max(0.0, now - solve_start))
    if out < base:
        fired["shrunk"] += 1
    fired["root_s"].append(round(out - now, 3))
    return out


if arm == "ON":
    S._gdp_config_root_until = h2_root_until
else:

    def counting(full_deadline, now, solve_start):
        fired["calls"] += 1
        base = _orig(full_deadline, now, solve_start)
        fired["root_s"].append(round(base - now, 3))
        return base

    S._gdp_config_root_until = counting

pm = px.builder(name)() if name.startswith(px.PREFIX) else __import__("gdp_baseline")._spec(name).builder()
conv = convert_pyomo_gdp(pm)
t = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = conv.model.solve(gdp_method=method, time_limit=300)
wall = time.perf_counter() - t
print(
    json.dumps(
        {
            "instance": name,
            "method": method,
            "arm": arm,
            "status": r.status,
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "obj": r.objective,
            "bound": r.bound,
            "nodes": r.node_count,
            "wall": round(wall, 3),
            "cfg_calls": fired["calls"],
            "cfg_shrunk": fired["shrunk"],
            "cfg_root_s": fired["root_s"],
        }
    ),
    flush=True,
)
