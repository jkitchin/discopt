"""#1152: did the root-relaxation fallback run, and with what grant?"""

from __future__ import annotations

import os
import sys
import time

import discopt.solver as S
from discopt.modeling.core import from_nl

name, T = sys.argv[1], float(sys.argv[2])
path = os.path.join("python/tests/data/minlplib_nl", name + ".nl")
CALLS = []
real = S._root_relaxation_lower_bound


def wrapper(model, lb, ub, time_limit, **k):
    t0 = time.perf_counter()
    out = real(model, lb, ub, time_limit, **k)
    CALLS.append((t0 - T0[0], time_limit, time.perf_counter() - t0, out))
    return out


S._root_relaxation_lower_bound = wrapper
T0 = [0.0]
m = from_nl(path)
T0[0] = time.perf_counter()
r = m.solve(time_limit=T)
wall = time.perf_counter() - T0[0]
print(
    f"{name} T={T} wall={wall:.2f} ratio={wall / T:.2f} bound={r.bound} nodes={r.node_count} "
    f"status={r.status}"
)
for start, grant, dur, out in CALLS:
    print(f"  fallback start={start:.2f} grant={grant:.2f} dur={dur:.2f} -> {out}")
print(f"# fallback_calls={len(CALLS)}")
