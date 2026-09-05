"""#1153: do the saturated carves actually SPEND their grant?

"ON == OFF" has two readings — the cap changed nothing because the stage never
wanted the time, or because the stage never ran at all (CLAUDE.md §6). This
records, per solve: every grant handed to a saturated carve, and the wall each
consuming stage actually spent. It also reports the largest single stage call,
so a solve that burns its budget somewhere else is visible rather than inferred.

Exits non-zero if no grant and no stage call was observed.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
import discopt.solver as S
import discopt._relax.lp_spatial_bb as LSB
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(solver_tuning, "saturate_role2"), "marker absent — wrong tree loaded"

grants = []
_real_sat = solver_tuning.saturate_role2


def _sat(seconds, frac):
    out = _real_sat(seconds, frac)
    grants.append((float(seconds), float(frac), float(out)))
    return out


S._role2_saturate = _sat
LSB.saturate_role2 = _sat

spend = {}


def _timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            n, tot, mx = spend.get(name, (0, 0.0, 0.0))
            spend[name] = (n + 1, tot + dt, max(mx, dt))

    return wrapper


# The consumers of the saturated grants, plus the two stages that dominate an
# unexplained wall: the node/root LP and the per-node OBBT.
import discopt._relax.obbt as OBBT
import discopt._relax.root_reduce as RR
from discopt._relax.mccormick_lp import MccormickLPRelaxer

OBBT.obbt_tighten_root = _timed("obbt_tighten_root", OBBT.obbt_tighten_root)
S.obbt_tighten_root = getattr(S, "obbt_tighten_root", OBBT.obbt_tighten_root)
RR.run_root_fixpoint = _timed("run_root_fixpoint", RR.run_root_fixpoint)
MccormickLPRelaxer.solve_at_node = _timed(
    "MccormickLPRelaxer.solve_at_node", MccormickLPRelaxer.solve_at_node
)

inst, tl = sys.argv[1], float(sys.argv[2])
arms = [a == "on" for a in (sys.argv[3:] or ["off", "on"])]
seen = 0
for on in arms:
    grants.clear()
    spend.clear()
    tok = solver_tuning.enter_scope(solver_tuning.SolverTuning(budget_saturation=on))
    t0 = time.perf_counter()
    try:
        r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    finally:
        solver_tuning.reset_current(tok)
    wall = time.perf_counter() - t0
    print(f"\n=== {os.path.basename(inst)} tl={tl} saturation={'ON' if on else 'OFF'} "
          f"obj={r.objective!r} bound={r.bound!r} nodes={r.node_count} status={r.status} "
          f"wall={wall:.1f}", flush=True)
    print(f"    grants ({len(grants)}):", flush=True)
    for g in grants:
        print(f"      carve {g[0]:8.2f}s (frac={g[1]}) -> granted {g[2]:8.2f}s"
              f"{'  CAPPED' if g[2] < g[0] - 1e-9 else ''}", flush=True)
    print("    stage wall (calls, total, max single):", flush=True)
    for k, (n, tot, mx) in sorted(spend.items(), key=lambda kv: -kv[1][1]):
        print(f"      {k:34s} n={n:6d} total={tot:8.2f}s ({100*tot/wall:5.1f}%) "
              f"max={mx:7.2f}s", flush=True)
    seen += len(grants) + len(spend)
print(f"\n# observed grants+stages: {seen}", flush=True)
raise SystemExit(1 if seen == 0 else 0)
