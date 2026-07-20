#!/usr/bin/env python
"""Issue #800 / T0 — canonical baseline, config A (production / unseeded).

Pins the wall / node_count / first-incumbent-latency anchor every later #800 task
compares against, on the full convex panel {rsyn0805m/0810m/0815m/0820m/0830m,
syn05m/10m/15m/20m/40m}. This is the PRODUCTION/UNSEEDED path — it mirrors
`try_convex_solve` exactly (same spec build, same budget, `initial_incumbent=None`,
same #779 incumbent verification) but captures the raw kernel dict so it can record
`first_incumbent_secs`/`first_incumbent_node` (added to the kernel result in T0).

Cert-clean is asserted (T0 kills only on a non-cert-clean run): the dual bound
never sits below the MINLPLib oracle optimum, a certified objective matches the
oracle (no false optimal), and every used incumbent is #779-verified feasible
against the pristine model.

Config B (seeded nodes-to-certify vs the prototype 67/60/46/55) is the separate
anchor `issue798_k2_tree_gate.py`.

Usage:  python issue800_t0_baseline.py [budget_seconds]   (default 120, the
        production DISCOPT_CONVEX_KERNEL_BUDGET default)
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"
os.environ["DISCOPT_CONVEX_KERNEL"] = "1"

import numpy as np  # noqa: E402

import discopt.modeling as dm  # noqa: E402
from discopt.solvers._convex_kernel import (  # noqa: E402
    _incumbent_is_feasible,
    _unflatten,
    build_convex_spec,
    solve_convex_tree,
)

PANEL = [
    "rsyn0805m",
    "rsyn0810m",
    "rsyn0815m",
    "rsyn0820m",
    "rsyn0830m",
    "syn05m",
    "syn10m",
    "syn15m",
    "syn20m",
    "syn40m",
]

SNAP = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")


def load_oracle() -> dict[str, float]:
    oracle: dict[str, float] = {}
    for line in open(SOLU):
        p = line.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            try:
                oracle[p[1]] = float(p[2])
            except ValueError:
                pass
    return oracle


def main() -> bool:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    oracle = load_oracle()
    print(
        f"# T0 config A (production/unseeded), budget={budget:g}s\n"
        f"{'instance':12s} {'status':10s} {'wall_s':>8s} {'nodes':>7s} "
        f"{'inc_node':>8s} {'inc_lat_s':>9s} {'bound':>12s} {'opt':>12s} "
        f"{'used':>5s}  verdict"
    )
    routed = incorrect = 0
    rows = []
    for name in PANEL:
        f = SNAP + name + ".nl"
        if not os.path.exists(f):
            print(f"{name:12s} MISSING")
            continue
        m = dm.from_nl(f)
        spec = build_convex_spec(m)
        if spec is None:
            print(f"{name:12s} declined")
            continue
        routed += 1
        opt = oracle.get(name)
        t0 = time.perf_counter()
        r = solve_convex_tree(spec, time_limit_s=budget, initial_incumbent=None)
        wall = time.perf_counter() - t0

        status = r["status"]
        bound = float(r["bound"])
        inc = r["incumbent"]
        inc_node = r.get("first_incumbent_node")
        inc_lat = r.get("first_incumbent_secs")

        # Reproduce try_convex_solve's "use the result" decision + #779 verify.
        used = False
        obj = float("nan")
        if status == "optimal" and inc is not None:
            inc_x = np.asarray(r["incumbent_x"], float)
            if inc_x.size:
                _xd, x_flat = _unflatten(m, inc_x)
                if _incumbent_is_feasible(m, x_flat):
                    used = True
                    obj = float(inc)

        # Cert-clean assertions (T0 kills only if a run is NOT cert-clean).
        sound_bound = opt is None or bound >= opt - 1e-4 * max(1.0, abs(opt))
        false_opt = used and opt is not None and abs(obj - opt) > 1e-2 * max(1.0, abs(opt))
        bad = (not sound_bound) or false_opt
        if bad:
            incorrect += 1
        verdict = "OK" if not bad else "CERT-FAIL"
        if not sound_bound:
            verdict += "(unsound-bound)"
        if false_opt:
            verdict += "(false-optimal)"

        inc_node_s = "-" if inc_node is None else str(inc_node)
        inc_lat_s = "-" if inc_lat is None else f"{inc_lat:.2f}"
        opt_s = "None" if opt is None else f"{opt:12.4f}"
        print(
            f"{name:12s} {status:10s} {wall:8.2f} {r['node_count']:7d} "
            f"{inc_node_s:>8s} {inc_lat_s:>9s} {bound:12.4f} {opt_s:>12s} "
            f"{str(used):>5s}  {verdict}",
            flush=True,
        )
        rows.append((name, status, wall, r["node_count"], inc_node, inc_lat, bound, opt, used))

    ok = incorrect == 0
    print(
        f"\nT0 config A cert-clean: routed={routed} incorrect={incorrect}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
