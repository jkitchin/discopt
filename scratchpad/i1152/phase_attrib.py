"""#1152 attribution: where does a ``solve(time_limit=T)`` overrun go, phase by phase?

Wraps every pre-B&B call that can dominate the root setup and prints a timeline
relative to the solve start. Ends with an executed-event count and exits non-zero
if nothing was recorded (§6). No exception is swallowed (§7).
"""

from __future__ import annotations

import os
import sys
import time

import discopt.solver as S
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt.modeling.core import from_nl

print(f"# marker=i1152-phase-attrib sources={S.__file__}", flush=True)

name = sys.argv[1] if len(sys.argv) > 1 else "casctanks"
T = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
path = os.path.join("python/tests/data/minlplib_nl", name + ".nl")

EVENTS: list[tuple[str, float, float]] = []
T0 = [0.0]


def _wrap(obj, attr, label):
    real = getattr(obj, attr)

    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            EVENTS.append((label, t0 - T0[0], time.perf_counter() - t0))

    setattr(obj, attr, wrapper)
    return real


_wrap(MccormickLPRelaxer, "solve_at_node", "MccormickLPRelaxer.solve_at_node")
_wrap(S, "_root_relaxation_lower_bound", "_root_relaxation_lower_bound")

import discopt._relax.uniform_relax as UR  # noqa: E402

_wrap(UR, "build_uniform_relaxation", "build_uniform_relaxation")

import discopt._relax.milp_relaxation as MR  # noqa: E402

_wrap(MR, "build_milp_relaxation", "build_milp_relaxation")

import discopt._relax.nonlinear_bound_tightening as NBT  # noqa: E402

_wrap(NBT, "tighten_nonlinear_bounds", "tighten_nonlinear_bounds")

m = from_nl(path)
T0[0] = time.perf_counter()
r = m.solve(time_limit=T)
wall = time.perf_counter() - T0[0]

print(
    f"\n{name} T={T} wall={wall:.2f} ratio={wall / T:.2f} status={r.status} "
    f"nodes={r.node_count} bound={r.bound} obj={r.objective}",
    flush=True,
)
print(f"{'phase':38s} {'start':>8s} {'dur':>8s} {'end':>8s}")
for label, start, dur in EVENTS:
    print(f"{label:38s} {start:8.2f} {dur:8.2f} {start + dur:8.2f}")
agg: dict[str, list[float]] = {}
for label, _s, dur in EVENTS:
    agg.setdefault(label, []).append(dur)
print("\n# totals")
for label, ds in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
    print(f"{label:38s} n={len(ds):3d} total={sum(ds):7.2f} max={max(ds):7.2f}")
print(f"# events={len(EVENTS)}", flush=True)
if not EVENTS:
    sys.exit(1)
