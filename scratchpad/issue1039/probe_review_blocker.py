"""Verify PR #1150 review's blocker claim: does the _is_linear widening produce a
false `optimal` on the GDP hull/auto route?  §8: assert which code is loaded and
that the version marker is present."""
import sys, warnings
import discopt
import discopt.modeling as dm
import discopt._relax.gdp_reformulate as G

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__, discopt.__file__
src = open(G.__file__).read()
print(f"marker SumOverExpression count in gdp_reformulate.py = {src.count('SumOverExpression')}")

TRUE_OPT = -30.0
n = 0
for method in ("auto", "hull", "big-m"):
    m = dm.Model("t")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
    m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]], name="modes")
    m.minimize(-(x[0] + x[1] + x[2]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(gdp_method=method, time_limit=30)
        bad = (r.objective is not None and r.objective > TRUE_OPT + 1e-6)
        print(f"gdp_method={method:7s} status={r.status:10s} obj={r.objective!r} "
              f"bound={r.bound!r}{'   <-- FALSE CERTIFICATE' if bad and r.status=='optimal' else ''}")
    except Exception as e:                     # §7: report, never swallow
        print(f"gdp_method={method:7s} RAISED {type(e).__name__}: {str(e)[:110]}")
    n += 1
print(f"\nEXECUTED CASES: {n}   (true optimum {TRUE_OPT})")
sys.exit(0 if n else 1)
