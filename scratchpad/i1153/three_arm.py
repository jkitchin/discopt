"""#1153: legacy vs FLAT share vs SUCCESS-WEIGHTED share, one container, one run.

The pre-restart and post-restart baselines differ (tspn10@5 was 7/7, is now 3/3),
so cross-container comparison is invalid. All three arms are measured here in one
process, interleaved within each repetition (CLAUDE.md §9).

Arms:
  legacy   flag off                       -> share 1.0 always
  flat     flag on, share applied flatly   -> share = base at every call
  weighted flag on, share ** fruitless     -> the shipped rule
"""
import os, statistics, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
import discopt.solver as S
from discopt import solver_tuning
from discopt.modeling.core import from_nl

_REAL_SHARE = S._heuristic_entry_share

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(solver_tuning, "heuristic_entry_share"), "marker absent — wrong tree"

# The success-weighted rule was measured inert and DELETED from the package
# (plan doc §6.5), so this harness reconstructs it locally instead of importing a
# symbol that no longer exists — a committed probe that cannot run is not
# evidence, and this one is cited as the falsification's evidence.
#
# Both arms are injected by swapping ``solver._heuristic_entry_share``, the seam
# the entry rule reads. ``finder`` scoping is unaffected: the rule still applies
# the share only at the two feasibility-pump entries.
_STATE = {"calls": 0, "found": 0}


def _flat():
    return solver_tuning.heuristic_entry_share()


def _weighted():
    base = solver_tuning.heuristic_entry_share()
    if base >= 1.0:
        return 1.0
    return float(base) ** max(0, _STATE["calls"] - _STATE["found"])


ARMS = {
    "legacy": (False, _flat),
    "flat": (True, _flat),
    "weighted": (True, _weighted),
}

names = sys.argv[1].split(",")
rungs = [float(x) for x in sys.argv[2].split(",")]
reps = int(sys.argv[3])
acc, n_runs = {}, 0
for rep in range(reps):
    for name in names:
        for tl in rungs:
            for arm, (on, fn) in ARMS.items():
                S._heuristic_entry_share = fn
                tok = solver_tuning.enter_scope(
                    solver_tuning.SolverTuning(heuristic_entry_share=on)
                )
                try:
                    r = from_nl(f"python/tests/data/minlplib_nl/{name}.nl").solve(
                        time_limit=tl, gap_tolerance=1e-4
                    )
                finally:
                    solver_tuning.reset_current(tok)
                    S._heuristic_entry_share = _REAL_SHARE
                acc.setdefault((name, tl, arm), []).append((int(r.node_count or 0), r.objective))
                n_runs += 1
                print(f"rep={rep} {name:16s} tl={tl:5.1f} {arm:9s} nodes={r.node_count} "
                      f"obj={r.objective!r}", flush=True)
print()
for name in names:
    for tl in rungs:
        for arm in ARMS:
            v = acc.get((name, tl, arm), [])
            if not v:
                continue
            nd = [x[0] for x in v]
            sd = statistics.pstdev(nd) if len(nd) > 1 else 0.0
            print(f"{name:16s} tl={tl:5.1f} {arm:9s} nodes={nd} mean={statistics.mean(nd):6.1f} "
                  f"sd={sd:4.1f} obj={v[-1][1]!r}", flush=True)
print(f"\n# executed runs: {n_runs}", flush=True)
raise SystemExit(1 if n_runs == 0 else 0)
