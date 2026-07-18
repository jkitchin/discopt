#!/usr/bin/env python3
r"""Attribute the #282 Workstream-A panel cert-clean violations (nvs13, tanksize).

Both flagged instances are INERT at the root (root byte-identical ON/OFF), so the
flag cannot be the structural cause. This probe re-runs each OFF x2 and ON x2 in
isolation to separate a genuine flag effect from run-to-run nondeterminism (the
concurrent-load artifact correction #2 warned about), and independently checks the
feasibility of EVERY incumbent (OFF and ON) with the exact violation magnitude.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_NL = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_FLAG = "DISCOPT_ROOT_LP_PROBE_TIGHT"


def feasible(nl_path, x):
    """Return (ok, worst_var_viol, worst_con_viol) for incumbent x vs the raw .nl."""
    m = from_nl(str(nl_path))
    repr_ = m._nl_repr
    x_flat = np.empty(repr_.n_vars, dtype=np.float64)
    off = 0
    worst_v = 0.0
    for v in m._variables:
        chunk = np.asarray(x[v.name], dtype=np.float64).reshape(-1)
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        worst_v = max(worst_v, float(np.max(np.maximum(lb - chunk, 0.0), initial=0.0)))
        worst_v = max(worst_v, float(np.max(np.maximum(chunk - ub, 0.0), initial=0.0)))
        x_flat[off : off + v.size] = chunk
        off += v.size
    worst_c = 0.0
    for i in range(repr_.n_constraints):
        body = float(repr_.evaluate_constraint(i, x_flat))
        rhs = float(repr_.constraint_rhs(i))
        s = repr_.constraint_sense(i)
        if s in ("<=", "L"):
            worst_c = max(worst_c, body - rhs)
        elif s in (">=", "G"):
            worst_c = max(worst_c, rhs - body)
        elif s in ("==", "E"):
            worst_c = max(worst_c, abs(body - rhs))
    return worst_v, worst_c


def run(name, flag_on, tl=25.0):
    if flag_on:
        os.environ[_FLAG] = "1"
    else:
        os.environ.pop(_FLAG, None)
    m = from_nl(str(_NL / f"{name}.nl"))
    res = m.solve(time_limit=tl, gap_tolerance=1e-4)
    wv, wc = (None, None)
    if getattr(res, "x", None) is not None:
        wv, wc = feasible(_NL / f"{name}.nl", res.x)
    return res, wv, wc


def main():
    for name in ("nvs13", "tanksize"):
        print(f"\n===== {name} =====")
        for tag, on in (("OFF#1", False), ("OFF#2", False), ("ON#1", True), ("ON#2", True)):
            res, wv, wc = run(name, on)
            print(
                f"  {tag}: status={res.status} obj={res.objective} "
                f"root={getattr(res, 'root_bound', None)} nodes={res.node_count} "
                f"| worst_var_viol={wv} worst_con_viol={wc}"
            )


if __name__ == "__main__":
    main()
