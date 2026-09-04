"""#1152 entry measurement: which in-repo instances overrun ``solve(time_limit=T)``?

Prints one line per (instance, T) as it goes (§10) and a final count of executed
comparisons; exits non-zero if nothing was measured (§6).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from discopt.modeling.core import from_nl  # noqa: E402

CORPUS = "python/tests/data/minlplib_nl"
T_LIST = [float(t) for t in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["5"])]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None

names = sorted(f[:-3] for f in os.listdir(CORPUS) if f.endswith(".nl"))
if ONLY:
    names = [n for n in names if n in ONLY]

n_cmp = 0
print(f"# marker=i1152-overrun-sweep  sources={from_nl.__module__}", flush=True)
for name in names:
    for T in T_LIST:
        m = from_nl(os.path.join(CORPUS, name + ".nl"))
        t0 = time.perf_counter()
        r = m.solve(time_limit=T)
        wall = time.perf_counter() - t0
        n_cmp += 1
        print(
            f"{name:24s} T={T:6.1f} wall={wall:8.2f} ratio={wall / T:6.2f} "
            f"status={r.status:12s} nodes={r.node_count:7d} bound={r.bound} obj={r.objective}",
            flush=True,
        )
print(f"# comparisons={n_cmp}", flush=True)
if n_cmp == 0:
    sys.exit(1)
