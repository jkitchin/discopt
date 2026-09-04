"""#1039 bucket F: does sonet23v4 still lose its dual bound at time_limit=2.0?
Also measure the wall, since bucket B says root setup overruns its budget --
these two tests pull in opposite directions and the numbers decide."""
import os, sys, time
import discopt
from discopt.modeling.core import from_nl

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__

BENCH = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
path = os.path.join(BENCH, "sonet23v4.nl")
if not os.path.exists(path):
    print(f"NOT VENDORED: {path}")
    sys.exit(1)

n = 0
for tl in (2.0, 8.0, 30.0):
    t0 = time.perf_counter()
    r = from_nl(path).solve(time_limit=tl)
    w = time.perf_counter() - t0
    print(f"tl={tl:5.1f} wall={w:7.1f}s ({w/tl:5.2f}x) status={r.status:10s} "
          f"nodes={r.node_count} bound={r.bound!r} obj={r.objective!r}")
    if r.bound is not None:
        print(f"        sound vs oracle -22747.5: {r.bound <= -22747.5 + 1e-4}", flush=True)
    n += 1
print(f"EXECUTED SOLVES: {n}")
sys.exit(0 if n else 1)
