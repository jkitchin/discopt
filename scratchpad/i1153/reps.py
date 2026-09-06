"""#1153: is the node-count inversion at 5 s -> 10 s real, or run-to-run noise?

Interleaved repetitions (CLAUDE.md §9), never budget-major. Reports per-rung
node counts, their spread, and the incumbent, plus an executed-rep count.
"""
import os, statistics, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
names = sys.argv[1].split(",")
rungs = [float(x) for x in sys.argv[2].split(",")]
reps = int(sys.argv[3])
data = {(n, t): [] for n in names for t in rungs}
objs = {(n, t): [] for n in names for t in rungs}
n_runs = 0
for rep in range(reps):
    for name in names:
        for tl in rungs:
            r = from_nl(f"python/tests/data/minlplib_nl/{name}.nl").solve(
                time_limit=tl, gap_tolerance=1e-4
            )
            data[(name, tl)].append(int(r.node_count or 0))
            objs[(name, tl)].append(r.objective)
            n_runs += 1
            print(f"rep={rep} {name:16s} tl={tl:5.1f} nodes={r.node_count} "
                  f"obj={r.objective!r}", flush=True)
print()
for name in names:
    for tl in rungs:
        v = data[(name, tl)]
        sd = statistics.pstdev(v) if len(v) > 1 else 0.0
        print(f"{name:16s} tl={tl:5.1f} nodes={v} mean={statistics.mean(v):.1f} sd={sd:.1f} "
              f"objs={objs[(name, tl)]}", flush=True)
print(f"\n# executed runs: {n_runs}", flush=True)
raise SystemExit(1 if n_runs == 0 else 0)
