"""One (instance, arm) solve, in its own process, printing a JSON line.

Usage: solve_one.py <nl-path> <arm: base|fix> <max_nodes> <time_limit>

CLAUDE.md §8: the arm is verified by a marker in the loaded module source, not by
the file the driver *meant* to copy. §7: nothing is swallowed -- a raise is printed
as a result and the process exits non-zero.
"""

import faulthandler
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

path, arm, max_nodes, tl = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])

import discopt  # noqa: E402
from discopt._relax import nonlinear_bound_tightening as nbt  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

src = open(nbt.__file__).read()
has_marker = "#1415" in src
if arm == "fix" and not has_marker:
    sys.exit(f"ARM MISMATCH: 'fix' arm loaded {nbt.__file__} without the #1415 marker")
if arm == "base" and has_marker:
    sys.exit(f"ARM MISMATCH: 'base' arm loaded {nbt.__file__} WITH the #1415 marker")

faulthandler.enable()
faulthandler.dump_traceback_later(tl + 180.0, exit=True)
t0 = time.perf_counter()
try:
    r = from_nl(path).solve(max_nodes=max_nodes, time_limit=tl)
except Exception as exc:  # noqa: BLE001 - a crash is a result
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
wall = time.perf_counter() - t0

print(
    json.dumps(
        {
            "arm": arm,
            "discopt": discopt.__file__,
            "status": r.status,
            "objective": None if r.objective is None else float(r.objective),
            "bound": None if r.bound is None else float(r.bound),
            "certified": bool(r.gap_certified),
            "nodes": int(r.node_count or 0),
            "wall": wall,
            "backstop": r.status == "time_limit" or wall >= tl * 0.98,
        }
    )
)
