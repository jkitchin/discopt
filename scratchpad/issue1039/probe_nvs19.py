"""#1039 bucket F: nvs19 returned -1001.2 in 60 s against a known optimum of
-1098.4, though the test's docstring says it "reaches the optimum in ~11s
locally".  Find the budget at which it actually reaches the optimum now, so the
test gets a RE-DERIVED budget backed by a measurement rather than a nudge."""
import sys, time
import discopt
from discopt.modeling import from_nl

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__

NL = "python/tests/data/minlplib/nvs19.nl"
OPT = -1098.4
n = 0
for tl in (30, 60, 120, 240, 480):
    t0 = time.perf_counter()
    r = from_nl(NL).solve(time_limit=tl, gap_tolerance=1e-4)
    w = time.perf_counter() - t0
    reached = r.objective is not None and r.objective <= OPT + 1e-3
    print(f"tl={tl:4d} wall={w:7.1f}s status={r.status:10s} nodes={r.node_count} "
          f"obj={r.objective!r} bound={r.bound!r} reached_opt={reached}", flush=True)
    n += 1
    if reached:
        print(f"--> first budget reaching the optimum: {tl}s (wall {w:.1f}s)")
        break
print(f"EXECUTED SOLVES: {n}")
sys.exit(0 if n else 1)
