"""Single-tree OA with fractional separation, checked against brute force (#1141).

All variables integer in a small box, one convex quadratic constraint that lives
ONLY in the separators. The certified optimum must equal the enumerated optimum,
with and without the node hook. This is the test that would catch a driver bug the
redundant-cut differential cannot: here the node cuts genuinely cut.

Prints an executed-comparison count and exits non-zero on any mismatch (§6).
"""
import itertools
import sys
import numpy as np
from discopt.solvers.milp_simplex import solve_milp_with_lazy_cuts

rng = np.random.default_rng(7)
compared = 0
bad = []

for trial in range(60):
    n = int(rng.integers(2, 5))
    LO, HI = -3, 3
    a = rng.normal(scale=1.5, size=n).round(2)
    r2 = float(round(rng.uniform(2.0, 12.0), 2))
    c = rng.normal(size=n).round(2)
    A = rng.normal(size=(2, n)).round(2)
    b = (np.abs(A) @ np.full(n, 3.0) * rng.uniform(0.5, 1.0, size=2)).round(3)

    def g(x, _a=a, _r2=r2):
        return float(np.sum((np.asarray(x, float) - _a) ** 2) - _r2)

    def cut(x, _a=a, _r2=r2):
        x = np.asarray(x, float)
        if g(x) <= 1e-9:
            return []
        # tangent of the convex g at x: g(x) + 2(x-a).(z-x) <= 0
        grad = 2.0 * (x - _a)
        return [(grad, float(grad @ x - g(x)))]

    # brute force over the integer box, honouring the linear rows too
    best, bestx = None, None
    for pt in itertools.product(range(LO, HI + 1), repeat=n):
        p = np.array(pt, float)
        if np.any(A @ p > b + 1e-9):
            continue
        if g(p) > 1e-9:
            continue
        v = float(c @ p)
        if best is None or v < best:
            best, bestx = v, p

    res = {}
    for arm in ("off", "on"):
        kw = dict(c=c, A_ub=A, b_ub=b, bounds=[(float(LO), float(HI))] * n,
                  integrality=np.ones(n, int), lazy_callback=cut,
                  time_limit=20.0, gap_tolerance=1e-9)
        if arm == "on":
            kw.update(node_callback=cut, node_hook_rounds=3, node_hook_cut_cap=300)
        res[arm] = solve_milp_with_lazy_cuts(**kw)

    compared += 1
    for arm, r in res.items():
        st = dict(r.callback_stats or {})
        if best is None:
            ok = r.status.name in ("INFEASIBLE",) or r.objective is None
            detail = f"expected infeasible, got {r.status.name} obj={r.objective!r}"
        else:
            ok = r.objective is not None and abs(r.objective - best) <= 1e-6 * max(1.0, abs(best))
            detail = f"expected {best!r}, got {r.objective!r} (status {r.status.name})"
        # A dual bound above the true optimum is a false certificate even when the
        # incumbent happens to be right.
        if best is not None and r.bound is not None and r.bound > best + 1e-6 * max(1.0, abs(best)):
            ok = False
            detail += f"; BOUND {r.bound!r} ABOVE true optimum {best!r}"
        if not ok:
            bad.append((trial, arm, detail))
            print(f"trial {trial} arm={arm}: {detail} mipnode={st.get('mipnode_calls')}")

print(f"\nEXECUTED COMPARISONS: {compared * 2}   MISMATCHES: {len(bad)}")
if compared == 0:
    print("PROBE MEASURED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if bad else 0)
