"""#1153: node throughput and incumbent vs budget, both arms, interleaved.

Interleaves arms WITHIN a repetition (CLAUDE.md §9), reports spread, and prints
an executed-run count. Arm is chosen per run via SolverTuning, so both arms run
in one process on one binary.
"""
import os, statistics, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(solver_tuning, "heuristic_entry_share"), "marker absent — wrong tree"

names = sys.argv[1].split(",")
rungs = [float(x) for x in sys.argv[2].split(",")]
reps = int(sys.argv[3])
acc = {}
n_runs = 0
for rep in range(reps):
    for name in names:
        for tl in rungs:
            for on in (False, True):
                tok = solver_tuning.enter_scope(
                    solver_tuning.SolverTuning(heuristic_entry_share=on)
                )
                try:
                    r = from_nl(f"python/tests/data/minlplib_nl/{name}.nl").solve(
                        time_limit=tl, gap_tolerance=1e-4
                    )
                finally:
                    solver_tuning.reset_current(tok)
                acc.setdefault((name, tl, on), []).append(
                    (int(r.node_count or 0), r.objective, r.bound)
                )
                n_runs += 1
                print(f"rep={rep} {name:16s} tl={tl:6.1f} share={'ON ' if on else 'OFF'} "
                      f"nodes={r.node_count} obj={r.objective!r} bound={r.bound!r}",
                      flush=True)
print()
for name in names:
    for tl in rungs:
        for on in (False, True):
            v = acc.get((name, tl, on), [])
            if not v:
                continue
            nd = [x[0] for x in v]
            sd = statistics.pstdev(nd) if len(nd) > 1 else 0.0
            print(f"{name:16s} tl={tl:6.1f} share={'ON ' if on else 'OFF'} nodes={nd} "
                  f"mean={statistics.mean(nd):7.1f} sd={sd:5.1f} obj={v[-1][1]!r}", flush=True)
print(f"\n# executed runs: {n_runs}", flush=True)
raise SystemExit(1 if n_runs == 0 else 0)
