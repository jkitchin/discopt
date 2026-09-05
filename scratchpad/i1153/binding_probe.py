"""#1153 rule-6/rule-8 check: does the ON arm actually cap anything?

A differential whose ON arm never reaches the code under test measures nothing
and reads as a pass. This wraps every saturation call site, runs one solve per
arm at a budget past the reference, and prints how many calls CAPPED. Exits
non-zero when the ON arm capped nothing.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
import discopt.solver as S
import discopt._relax.lp_spatial_bb as LSB
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(solver_tuning, "saturate_role2"), "marker absent: wrong tree loaded"

calls = []
_real = solver_tuning.saturate_role2


def _wrapped(seconds, frac):
    out = _real(seconds, frac)
    calls.append((float(seconds), float(frac), float(out)))
    return out


S._role2_saturate = _wrapped
LSB.saturate_role2 = _wrapped

inst, tl = sys.argv[1], float(sys.argv[2])
rc = 0
for on in (False, True):
    calls.clear()
    tok = solver_tuning.enter_scope(solver_tuning.SolverTuning(budget_saturation=on))
    try:
        r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    finally:
        solver_tuning.reset_current(tok)
    capped = [c for c in calls if c[2] < c[0] - 1e-12]
    print(f"saturation={'ON ' if on else 'OFF'} tl={tl} calls={len(calls)} capped={len(capped)} "
          f"obj={r.objective!r} bound={r.bound!r} nodes={r.node_count} status={r.status}",
          flush=True)
    for c in capped:
        print(f"    capped: {c[0]:.2f}s -> {c[2]:.2f}s (frac={c[1]})", flush=True)
    if on and not capped:
        print("!! ON arm capped nothing — this differential measures nothing", flush=True)
        rc = 1
raise SystemExit(rc)
