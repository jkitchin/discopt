"""#1152 attribution v2 — logs the solver's own root-setup decisions plus phase walls."""

from __future__ import annotations

import logging
import os
import sys
import time

import discopt._relax.milp_relaxation as MR
import discopt._relax.uniform_relax as UR
import discopt.solver as S
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt.modeling.core import from_nl

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


_wrap(MccormickLPRelaxer, "solve_at_node", "solve_at_node")
_wrap(S, "_root_relaxation_lower_bound", "root_fallback")
_wrap(UR, "build_uniform_relaxation", "build_uniform_relaxation")
_wrap(MR, "build_milp_relaxation", "build_milp_relaxation")
_wrap(S, "_declared_box_tightening", "declared_box_tightening")
_wrap(S, "_classify_model_convexity", "classify_convexity")

logging.basicConfig(level=logging.INFO, format="LOG %(name)s %(message)s", stream=sys.stdout)
logging.getLogger("discopt").setLevel(logging.DEBUG)

m = from_nl(path)
T0[0] = time.perf_counter()
r = m.solve(time_limit=T)
wall = time.perf_counter() - T0[0]
print(
    f"\n{name} T={T} wall={wall:.2f} ratio={wall / T:.2f} status={r.status} "
    f"nodes={r.node_count} bound={r.bound} obj={r.objective}",
    flush=True,
)
for label, start, dur in EVENTS:
    print(f"EV {label:34s} {start:8.2f} {dur:8.2f} {start + dur:8.2f}")
print(f"# events={len(EVENTS)}")
sys.exit(0 if EVENTS else 1)
