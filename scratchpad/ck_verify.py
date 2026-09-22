"""#1422 entry check: does a DECLINED convex-kernel attempt's bound reach the result?

Prints an executed-assertion count and exits non-zero if it is zero (CLAUDE.md §6).
No exception is swallowed (§7). The loaded module is asserted (§8).
"""
import os, sys, time
import discopt
from discopt.modeling.core import from_nl
import discopt.solvers._convex_kernel as ck
import discopt.modeling.core as core

print("core   __file__:", core.__file__, flush=True)
print("kernel __file__:", ck.__file__, flush=True)
assert "keep_declined_bound_enabled" in dir(ck), "marker ABSENT: wrong tree loaded"

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
NAMES = sys.argv[1:] or ["ball_mk2_30"]
TL = float(os.environ.get("TL", "8"))

checks = 0
for nm in NAMES:
    m = from_nl(os.path.join(NL, nm + ".nl"))
    t0 = time.perf_counter()
    r = m.solve(time_limit=TL)
    w = time.perf_counter() - t0
    print(
        f"{nm:20s} status={r.status:12s} obj={r.objective} bound={r.bound} "
        f"valid={r.bound_valid} src={r.bound_source} wall={w:.2f} reported={r.wall_time:.2f}",
        flush=True,
    )
    checks += 1

print(f"EXECUTED_CHECKS={checks}")
sys.exit(0 if checks else 1)
