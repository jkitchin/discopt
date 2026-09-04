"""#1151 null control for Panel A's 10 result differences.

Panel A recorded ``_row_scales`` invocations = 0 in BOTH arms, so the two arms
executed identical code and no difference between them can be caused by this
change. This control demonstrates that directly rather than arguing it: it
re-runs Panel A's 10 differing instances with BOTH arms set to the SAME (new)
form, under the same interleaving. Differences of the same shape here are
run-order/wall-clock variation, which is what Panel A's ``divergent_rows = 0``
already implies.
"""

import sys
import time

import discopt.modeling as dm

INSTANCES = [
    "hda", "nvs17", "nvs19", "nvs20", "nvs23",
    "nvs24", "st_e31", "syn05hfsg", "tanksize", "tspn05",
]
ROOTS = ["python/tests/data/minlplib_nl", "python/tests/data/minlplib"]

import os

compared = 0
differing = 0
print(f"{'instance':<12}{'arm':>5}{'status':>12}{'objective':>20}{'bound':>22}{'nodes':>8}",
      flush=True)
for name in INSTANCES:
    path = next((os.path.join(r, name + ".nl") for r in ROOTS
                 if os.path.exists(os.path.join(r, name + ".nl"))), None)
    if path is None:
        print(f"{name}: not vendored", flush=True)
        continue
    recs = []
    for rep in ("A", "B"):
        t0 = time.perf_counter()
        r = dm.from_nl(path).solve(time_limit=20.0, gap_tolerance=1e-4)
        recs.append((r.status, r.objective, r.bound, int(r.node_count or 0)))
        print(f"{name:<12}{rep:>5}{r.status:>12}"
              f"{(r.objective if r.objective is not None else float('nan')):>20.12g}"
              f"{(r.bound if r.bound is not None else float('nan')):>22.12g}"
              f"{int(r.node_count or 0):>8}   ({time.perf_counter() - t0:.1f}s)", flush=True)
    compared += 1
    if recs[0] != recs[1]:
        differing += 1

print(f"\ninstances re-run with NO code difference between arms: {compared}")
print(f"  of which the two identical-code runs still disagree: {differing}")
if compared == 0:
    sys.exit("NULL CONTROL MEASURED NOTHING")
