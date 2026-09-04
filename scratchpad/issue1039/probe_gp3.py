"""#1039 bucket E: test the small-denominator amplification hypothesis.

Hypothesis: the reported objective for a model containing a quotient is taken
from a reformulated/auxiliary space where the defining equality (w*y = x) is
enforced to an ABSOLUTE tolerance, so the induced error in w is ~tol/y and blows
up as the denominator shrinks.

Prediction: the objective-vs-oracle delta scales like 1/lb.
Kill criterion: if the delta is flat (or does not shrink) as lb rises, the
amplification hypothesis is FALSE and the cause is elsewhere."""
import sys, warnings
import discopt
from discopt import Model

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__

n = 0
rows = []
for lb in (1e-3, 1e-2, 1e-1, 1.0):
    m = Model("div")
    x = m.continuous("x", lb=lb, ub=1e3)
    y = m.continuous("y", lb=lb, ub=1e3)
    m.minimize(x / y + y / x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(solver="bb", time_limit=10.0)
    xv, yv = float(r.value(x)), float(r.value(y))
    orc = xv / yv + yv / xv
    d = r.objective - orc
    rows.append((lb, min(xv, yv), d))
    print(f"lb={lb:<8g} status={r.status:8s} objective={r.objective!r}")
    print(f"{'':11s} point=({xv!r}, {yv!r}) min={min(xv,yv):.6g}")
    print(f"{'':11s} oracle={orc!r} delta={d:+.6e}  |delta|*min_denom={abs(d)*min(xv,yv):.3e}",
          flush=True)
    n += 1

print("\nlb        min_denom     delta          delta*min_denom (flat => 1/y law)")
for lb, md, d in rows:
    print(f"{lb:<9g} {md:<13.6g} {d:+.6e}  {abs(d)*md:.3e}")
print(f"\nEXECUTED SOLVES: {n}")
sys.exit(0 if n else 1)
