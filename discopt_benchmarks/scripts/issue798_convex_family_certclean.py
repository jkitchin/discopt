#!/usr/bin/env python
"""Issue #798 / K4 — cert-clean sweep over the convex rsyn/syn family (snapshot).

Routes each instance through the soundness gate + try_convex_solve and checks the
dual bound never sits below the MINLPLib oracle optimum (soundDual) and the
certified objective matches it (no false optimal). Unverifiable incumbents fall
back (sound). Graduation evidence beyond the 4-instance panel + in-repo corpus.
"""

import os
import time

os.environ["DISCOPT_COEF_TIGHTEN"] = "1"
os.environ["DISCOPT_CONVEX_KERNEL"] = "1"
import discopt.modeling as dm
from discopt.solvers._convex_kernel import build_convex_spec, try_convex_solve

snap = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
solu = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")
oracle = {}
for L in open(solu):
    p = L.split()
    if len(p) >= 3 and p[0] in ("=opt=", "=best="):
        try:
            oracle[p[1]] = float(p[2])
        except ValueError:
            pass
cand = [
    "rsyn0805m",
    "rsyn0810m",
    "rsyn0815m",
    "rsyn0820m",
    "rsyn0830m",
    "syn05m",
    "syn10m",
    "syn20m",
    "syn40m",
    "syn15m",
]
routed = incorrect = 0
for name in cand:
    f = snap + name + ".nl"
    if not os.path.exists(f):
        print(name, "missing", flush=True)
        continue
    m = dm.from_nl(f)
    if build_convex_spec(m) is None:
        print(f"{name:10s} declined", flush=True)
        continue
    routed += 1
    opt = oracle.get(name)
    t = time.perf_counter()
    r = try_convex_solve(m, time_limit=45)
    dt = time.perf_counter() - t
    if r is None:
        print(f"{name:10s} fallback(unverified)", flush=True)
        continue
    bad = (
        r.status == "optimal"
        and opt is not None
        and abs(r.objective - opt) > 1e-2 * max(1, abs(opt))
    )
    sound = opt is None or r.bound >= opt - 1e-4 * max(1, abs(opt))
    if bad or not sound:
        incorrect += 1
    print(
        f"{name:10s} {r.status:9s} obj={r.objective:.3f} bound={r.bound:.3f} opt={opt} {dt:.1f}s soundDual={sound} falseOpt={bad}",
        flush=True,
    )
print(
    f"\nCONVEX-FAMILY cert-clean: routed={routed} incorrect={incorrect}  {'PASS' if incorrect == 0 else 'FAIL'}",
    flush=True,
)
