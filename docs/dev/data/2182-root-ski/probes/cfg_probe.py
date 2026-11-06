"""#2182 H1 probe: does the root GDP config constructor keep spending after its last
improvement?

python -u cfg_probe.py <instance> <method>

Wraps primal_heuristics.one_hot_config_subnlp (imported by name inside solve_model
at call time, so patching the module attribute takes effect) and, while inside
it, primal_heuristics.subnlp, timestamping every feasible point. Prints one JSON
row and an executed-assertion count; exits non-zero if the constructor never ran.
"""

import json
import sys
import time
import warnings

sys.path.insert(0, "/home/user/discopt/discopt_benchmarks")
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import discopt  # noqa: E402
import discopt._relax.primal_heuristics as ph  # noqa: E402
from benchmarks import pyomo_gdp_examples as px  # noqa: E402

from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402

import os as _os  # noqa: E402

# §8: the tree under test; PROBE_TREE names it when it is not the main checkout.
assert discopt.__file__.startswith(
    _os.environ.get("PROBE_TREE", "/home/user/discopt/python")
), discopt.__file__
name, method = sys.argv[1], sys.argv[2]

calls = []  # one dict per one_hot_config_subnlp call
_inside = [None]

_orig_cfg = ph.one_hot_config_subnlp
_orig_sub = ph.subnlp


import numpy as _np


def sub_wrap(*a, **k):
    out = _orig_sub(*a, **k)
    c = _inside[0]
    if c is not None:
        c["n_sub"] += 1
        mask = ph._get_integer_mask(a[0])
        key = tuple(_np.round(_np.asarray(a[1], dtype=float)[mask]).astype(int).tolist())
        c.setdefault("keys", []).append(key)
        if out is not None:
            c["pts"].append((time.perf_counter() - c["t0"], float(out[1])))
    return out


def cfg_wrap(*a, **k):
    c = {"t0": time.perf_counter(), "n_sub": 0, "pts": []}
    dl = k.get("deadline")
    c["grant"] = None if dl is None else dl - c["t0"]
    _m = a[0]
    _im = ph._get_integer_mask(_m)
    _lb, _ub = ph._get_variable_bounds(_m)
    _g, _dec = ph._prune_decided_groups(
        ph._scan_one_hot_rows(_m, _im, int(_im.size)), *ph._pinned_bounds(_m, _lb, _ub)
    )
    _cov = {j for g in _g for j in g} | set(_dec)
    _res = [j for j in _np.nonzero(_im)[0].tolist() if j not in _cov and _lb[j] >= -1e-9 and _ub[j] <= 1 + 1e-9]
    _other_int = int(_np.count_nonzero(_im)) - len(_cov) - len(_res)
    _tot = 1
    for g in _g:
        _tot *= len(g)
    c["n_groups"] = len(_g)
    c["total_cfg"] = _tot * (2 ** len(_res)) if _other_int <= 0 else None
    _inside[0] = c
    try:
        res = _orig_cfg(*a, **k)
    finally:
        _inside[0] = None
    c["spent"] = time.perf_counter() - c["t0"]
    c["n_results"] = len(res)
    c["best_returned"] = min((float(o) for _, o in res), default=None)
    calls.append(c)
    return res


_orig_dive = ph.one_hot_config_dive


def dive_wrap(*a, **k):
    c = _inside[0]
    t = time.perf_counter()
    try:
        return _orig_dive(*a, **k)
    finally:
        if c is not None:
            c["dive_start"] = t - c["t0"]
            c["dive_spent"] = time.perf_counter() - t
            c["n_sub_before_dive"] = c.get("n_sub_snapshot", None)


ph.one_hot_config_dive = dive_wrap
ph.one_hot_config_subnlp = cfg_wrap
ph.subnlp = sub_wrap

pm = px.builder(name)() if name.startswith(px.PREFIX) else __import__("gdp_baseline")._spec(name).builder()
conv = convert_pyomo_gdp(pm)
t = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = conv.model.solve(gdp_method=method, time_limit=300)
wall = time.perf_counter() - t
rows = []
for c in calls:
    best, t_best = float("inf"), None
    for tt, o in c["pts"]:
        if t_best is None or o < best - 1e-9 * max(1.0, abs(best)):
            best, t_best = o, tt
    rows.append(
        {
            "grant": None if c["grant"] is None else round(c["grant"], 3),
            "spent": round(c["spent"], 3),
            "n_sub": c["n_sub"],
            "n_feas": len(c["pts"]),
            "t_last_improve": None if t_best is None else round(t_best, 3),
            "best_seen": None if t_best is None else best,
            "dive_start": None if c.get("dive_start") is None else round(c["dive_start"], 3),
            "dive_spent": None if c.get("dive_spent") is None else round(c["dive_spent"], 3),
            "n_distinct_cfg": len(set(c.get("keys", []))),
            "n_groups": c["n_groups"],
            "total_cfg": c["total_cfg"],
            "n_int": len(c["keys"][0]) if c.get("keys") else None,
            "dive_pts": sum(1 for tt, _ in c["pts"] if c.get("dive_start") is not None and tt >= c["dive_start"]),
        }
    )
out = {
    "instance": name,
    "method": method,
    "status": r.status,
    "obj": r.objective,
    "root_bound": r.root_bound,
    "nodes": r.node_count,
    "wall": round(wall, 3),
    "cfg_calls": rows,
}
print(json.dumps(out), flush=True)
n_assert = 0
assert calls, "constructor never ran"
n_assert += 1
for c, row in zip(calls, rows):
    assert (row["t_last_improve"] is None) == (len(c["pts"]) == 0), row
    n_assert += 1
print(f"executed-assertions: {n_assert}", flush=True)
