"""#1152 attribution v3 — who builds the uniform relaxation late in root setup?"""

from __future__ import annotations

import os
import sys
import time
import traceback

import discopt._relax.uniform_relax as UR
from discopt.modeling.core import from_nl

name, T = sys.argv[1], float(sys.argv[2])
path = os.path.join("python/tests/data/minlplib_nl", name + ".nl")
T0 = [0.0]
N = [0]

real = UR.build_uniform_relaxation


def wrapper(*a, **k):
    t0 = time.perf_counter()
    stack = "".join(traceback.format_stack(limit=8)[:-1])
    try:
        return real(*a, **k)
    finally:
        N[0] += 1
        print(
            f"\n=== build_uniform_relaxation start={t0 - T0[0]:.2f} "
            f"dur={time.perf_counter() - t0:.2f} deadline_kw={k.get('build_deadline')}\n{stack}",
            flush=True,
        )


UR.build_uniform_relaxation = wrapper
m = from_nl(path)
T0[0] = time.perf_counter()
r = m.solve(time_limit=T)
print(f"\n{name} wall={time.perf_counter() - T0[0]:.2f} bound={r.bound} nodes={r.node_count}")
print(f"# events={N[0]}")
sys.exit(0 if N[0] else 1)
