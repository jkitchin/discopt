"""Repeat the two panel instances that differed, to separate a real bound change
from a wall-clock artifact on a time-limited run."""
import sys, json
import discopt.mpec as mpec
from discopt.modeling.core import from_nl

arm = sys.argv[1]
print(f"[{arm}] marker present: {hasattr(mpec, 'carry_complementarities')}", flush=True)
n = 0
for name in ("bchoco07", "tls2"):
    for rep in range(3):
        m = from_nl(f"python/tests/data/minlplib_nl/{name}.nl")
        r = m.solve(time_limit=10.0)
        n += 1
        print(f"[{arm}] {name} rep{rep} nodes={r.node_count} obj={r.objective} bound={r.bound}", flush=True)
print(f"[{arm}] EXECUTED_SOLVES: {n}")
assert n > 0
