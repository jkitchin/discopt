# ruff: noqa -- frozen #1731 panel record, kept as run.
"""One solve of one instance; prints a single JSON line. argv: lib path T"""

import faulthandler
import json
import os
import sys
import time

import numpy as np

lib, path, T = sys.argv[1], sys.argv[2], float(sys.argv[3])
faulthandler.dump_traceback_later(T + 240, exit=True)

import discopt.solver as S  # noqa: E402
from discopt.modeling.core import ObjectiveSense  # noqa: E402
from discopt.warm_start import check_feasibility  # noqa: E402

assert S.__file__.startswith(os.environ["PANEL_TREE"]), S.__file__
assert hasattr(S, "_objective_lattice")  # marker: the #1731 code is loaded

if lib == "minlplib":
    from discopt.modeling.core import from_nl

    m = from_nl(path)
else:
    from discopt.interfaces.qplib import from_qplib

    m = from_qplib(path)

flag_on = S._tuning().spatial_obj_lattice
_orig_round = S._round_bounds_to_lattice
lat_stats = dict(calls=0, changed=0)


def _counting_round(lbs, g, off):
    out = _orig_round(lbs, g, off)
    lat_stats["calls"] += 1
    a = np.asarray(lbs, dtype=float)
    lat_stats["changed"] += int(np.sum(np.isfinite(a) & (out != a)))
    return out


S._round_bounds_to_lattice = _counting_round
t = time.time()
r = m.solve(time_limit=T)
wall = time.time() - t
feas = None
viol = []
if r.x is not None:
    xf = np.concatenate([np.ravel(np.asarray(r.x[v.name], dtype=float)) for v in m._variables])
    feas, viol = check_feasibility(m, xf, tol=1e-5)
print(
    "RESULT "
    + json.dumps(
        dict(
            flag=flag_on,
            sense="max" if m._objective.sense == ObjectiveSense.MAXIMIZE else "min",
            status=r.status,
            objective=r.objective,
            bound=r.bound,
            cert=bool(getattr(r, "gap_certified", False)),
            nodes=int(r.node_count),
            wall=wall,
            route=str(getattr(r, "algorithm_route", "")),
            feas=feas,
            lat_calls=lat_stats["calls"],
            lat_changed=lat_stats["changed"],
            viol=viol[:3],
        )
    ),
    flush=True,
)
