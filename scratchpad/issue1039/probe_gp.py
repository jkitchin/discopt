"""#1039 bucket E: test_bb_opt_out_skips_gp_fast_path reports objective
1.998683979470214 for `minimize x/y + y/x` over a positive box.  By AM-GM no
feasible point can give less than 2, so this is either a super-optimal incumbent
(a SOUNDNESS defect) or a bound leaking into the objective field.  Recompute the
objective from the reported point with an oracle written outside the system.
CLAUDE.md §6 executed count; §7 nothing swallowed."""
import sys, warnings
import discopt
from discopt import Model

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__, discopt.__file__

POS = dict(lb=1e-3, ub=1e3)
n = 0
for tl in (5.0, 30.0):
    m = Model("balance")
    x = m.continuous("x", **POS)
    y = m.continuous("y", **POS)
    m.minimize(x / y + y / x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(solver="bb", time_limit=tl)
    xv, yv = r.value(x), r.value(y)
    oracle = xv / yv + yv / xv           # independent of discopt arithmetic
    print(f"tl={tl}: status={r.status} objective={r.objective!r} bound={r.bound!r}")
    print(f"   x={xv!r} y={yv!r}  oracle f(x,y)={oracle!r}")
    print(f"   objective - oracle = {r.objective - oracle:.6e}")
    print(f"   oracle - 2.0       = {oracle - 2.0:.6e}   (must be >= 0 by AM-GM)")
    print(f"   in box: {POS['lb'] <= xv <= POS['ub']} {POS['lb'] <= yv <= POS['ub']}",
          flush=True)
    n += 1
print(f"EXECUTED SOLVES: {n}")
sys.exit(0 if n else 1)
