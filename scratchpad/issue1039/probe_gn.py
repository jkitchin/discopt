"""#1039 bucket E: test_solve_gauss_newton_matches_full_nonlinear_ls asserts
status=='optimal' and gets 'feasible' in ~1s.  Determine WHICH arm, and whether
the solve is stopping early (a completeness miss) or the two arms still agree on
the parameters (which is what the test is actually about)."""
import sys
import numpy as np
import discopt
from discopt import Model
import discopt.modeling as dm

assert "/Users/jkitchin/projects/discopt/python/discopt" in discopt.__file__

rng = np.random.default_rng(0)
t = np.linspace(0, 1, 12)
y = 2.0 * np.exp(1.3 * t) + 1e-3 * rng.standard_normal(12)


def build():
    m = Model("exp")
    p = m.continuous("p", lb=0.1, ub=5)
    q = m.continuous("q", lb=0.1, ub=3)
    expr = (p * dm.exp(q * float(t[0])) - float(y[0])) ** 2
    for i in range(1, len(t)):
        expr = expr + (p * dm.exp(q * float(t[i])) - float(y[i])) ** 2
    m.minimize(expr)
    return m, p, q


def sse(pv, qv):
    return float(np.sum((pv * np.exp(qv * t) - y) ** 2))


n = 0
sols = {}
for gn in (False, True):
    for tl in (120, 600):
        m, p, q = build()
        r = m.solve(gauss_newton=gn, time_limit=tl, skip_convex_check=True)
        pv, qv = float(r.value(p)), float(r.value(q))
        print(f"gn={gn!s:5s} tl={tl:3d} status={r.status:9s} nodes={r.node_count} "
              f"obj={r.objective!r}")
        print(f"{'':18s} bound={r.bound!r} gap={r.gap!r} certified={r.gap_certified}")
        print(f"{'':18s} p={pv!r} q={qv!r} oracle_sse={sse(pv, qv)!r}", flush=True)
        sols[(gn, tl)] = (pv, qv)
        n += 1

print("\nparameter agreement between arms (what the test is really about):")
for tl in (120, 600):
    a, b = sols[(False, tl)], sols[(True, tl)]
    print(f"  tl={tl}: allclose(atol=1e-3) = {np.allclose(a, b, atol=1e-3)}  {a} vs {b}")
print(f"\nEXECUTED SOLVES: {n}")
sys.exit(0 if n else 1)
