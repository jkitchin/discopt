"""#1152: seek a feasible incumbent for the instances whose bound MOVED in the panel.

A feasible objective is a valid upper bound on the true optimum, so it is the
soundness oracle for a dual bound on an instance ``minlplib.solu`` is not available
for (this environment has no MINLPLib snapshot). Prints one line per instance and a
count; exits non-zero if it probed nothing (§6).
"""

from __future__ import annotations

import os
import sys
import time

from discopt.modeling.core import from_nl

T = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
NAMES = (
    sys.argv[2].split(",")
    if len(sys.argv) > 2
    else ["4stufen", "beuster", "casctanks", "bchoco08", "tanksize"]
)

n = 0
for name in NAMES:
    path = os.path.join("python/tests/data/minlplib_nl", name + ".nl")
    if not os.path.exists(path):
        print(f"{name:14s} NOT VENDORED", flush=True)
        continue
    t0 = time.perf_counter()
    r = from_nl(path).solve(time_limit=T)
    n += 1
    print(
        f"{name:14s} T={T:.0f} wall={time.perf_counter() - t0:7.1f} status={r.status:11s} "
        f"objective={r.objective} bound={r.bound}",
        flush=True,
    )
print(f"# probed={n}")
sys.exit(0 if n else 1)
