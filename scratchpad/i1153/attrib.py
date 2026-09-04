"""#1153 attribution: where did the budget go, and what produced the incumbent?

Runs one instance at each rung with INFO logging captured, then prints, per rung,
the incumbent and the ordered list of budget-consuming stage messages. Prints an
executed-rung count and exits non-zero when zero rungs ran.
"""
import io, logging, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
from discopt.modeling.core import from_nl
from discopt import solver_tuning

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)

inst = sys.argv[1]
rungs = [float(x) for x in sys.argv[2].split(",")]
sat = len(sys.argv) > 3 and sys.argv[3] == "on"

root = logging.getLogger("discopt")
root.setLevel(logging.INFO)
n = 0
for tl in rungs:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(relativeCreated)8.0fms %(name)s %(message)s"))
    root.addHandler(h)
    tok = solver_tuning.enter_scope(solver_tuning.SolverTuning(budget_saturation=sat))
    t0 = time.perf_counter()
    try:
        r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    finally:
        solver_tuning.reset_current(tok)
        root.removeHandler(h)
    n += 1
    print(f"\n===== tl={tl} sat={sat} obj={r.objective!r} bound={r.bound!r} "
          f"nodes={r.node_count} status={r.status} wall={time.perf_counter()-t0:.2f}", flush=True)
    for line in buf.getvalue().splitlines():
        print("   " + line, flush=True)
print(f"\n# executed rungs: {n}", flush=True)
raise SystemExit(1 if n == 0 else 0)
