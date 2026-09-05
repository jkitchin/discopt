"""#1165: is hda's ON arm under ``deterministic=True, max_nodes=1`` slow, or hung?

The OFF arm returns in 305 s; a 2400 s wrapper killed the ON arm before it
returned. A timing wrapper that prints on return never fires for a call that does
not return (CLAUDE.md §10), so this uses ``faulthandler.dump_traceback_later`` to
print where the ON arm actually is at fixed intervals, and lets it run to a hard
cap. Catches nothing (§7); prints an executed-solve count (§6).
"""

import faulthandler
import os
import sys
import time

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402

print(f"[§8] discopt.__file__ = {discopt.__file__}", flush=True)
assert "/home/user/discopt/python/discopt/" in discopt.__file__, "wrong tree loaded"

faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True, exit=False)

os.environ["DISCOPT_RELAX_ROW_FILTER"] = "0"
os.environ["DISCOPT_NODE_NUMERICAL_DUAL_BOUND"] = "1"

t0 = time.time()
r = dm.from_nl(os.path.join("python", "tests", "data", "minlplib_nl", "hda.nl")).solve(
    time_limit=100000, max_nodes=1, deterministic=True
)
dt = time.time() - t0
faulthandler.cancel_dump_traceback_later()
print(f"\nON arm returned after {dt:.1f}s status={r.status} nodes={r.node_count} "
      f"bound={r.bound!r}", flush=True)
print("executed solves = 1")
