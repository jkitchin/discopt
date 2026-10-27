"""#2010 probe: one solve of a ``.nl`` instance, recording the ``max_nodes`` each
#1940 lift-race arm received and the nodes it spent.

Usage: ``python -u scripts/issue2010_race_budget_probe.py <file.nl> <time_limit>``
with ``SLICE=<seconds>`` to scale ``_IPX_RACE_SLICE_S`` (the #1959 CI-speed proxy,
used the other way: a longer slice stands in for a faster machine). Prints one
``RESULT`` JSON line; exits non-zero when no race arm ran (CLAUDE.md §6).
"""

from __future__ import annotations

import json
import os
import sys
import time

import discopt.modeling as dm
import discopt.solver as sv

print("solver loaded from", sv.__file__, flush=True)
nl, tl = sys.argv[1], float(sys.argv[2])
arms: list = []
_real = sv.solve_model


def _spy(*args, **kw):
    arm = sv._IPX_RACE_ARM.get()
    t = time.perf_counter()
    r = _real(*args, **kw)
    if arm is not None:
        spent = int(r.node_count or 0)
        arms.append((arm, kw.get("max_nodes"), spent, str(r.status), time.perf_counter() - t))
    return r


sv.solve_model = _spy
sv._IPX_RACE_SLICE_S = float(os.environ.get("SLICE", sv._IPX_RACE_SLICE_S))
t0 = time.perf_counter()
r = dm.from_nl(nl).solve(time_limit=tl, gap_tolerance=1e-4)
rec = dict(
    inst=os.path.basename(nl)[:-3],
    status=str(r.status),
    obj=r.objective,
    bound=r.bound,
    nodes=r.node_count,
    wall=round(time.perf_counter() - t0, 2),
    cert=r.gap_certified,
    arms=arms,
)
print("RESULT", json.dumps(rec), flush=True)
print("EXECUTED race arms:", len(arms), flush=True)
os._exit(0 if arms else 1)
