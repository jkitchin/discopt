"""Is test_large_dense_jacobian_no_crash's deadline overrun caused by #940?

The assertion is wall-clock (wall < 48s), so it is exactly the kind of claim
CLAUDE.md §9 says needs an interleaved A/B and a spread, not a single sequential
run. Arms are selected by flipping the module globals the POUNCE backends read,
so both run in one process under identical load, alternating order.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.modeling as dm
import discopt.solvers.lp_pounce as LPP
import discopt.solvers.qp_pounce as QPP

assert LPP._CONSTR_VIOL_TOL == 1e-8
_POST_TOL, _POST_REJECT, _POST_RAY = (
    LPP._CONSTR_VIOL_TOL, LPP._settle_ambiguous_unbounded, QPP._certify_unbounded_ray)


def set_arm(arm):
    if arm == "pre":
        LPP._CONSTR_VIOL_TOL = QPP._CONSTR_VIOL_TOL = 1e-4
        LPP._settle_ambiguous_unbounded = lambda result, c, A, cl, cu, lb, ub, opts: result
        QPP._certify_unbounded_ray = lambda *a, **k: True
    else:
        LPP._CONSTR_VIOL_TOL = QPP._CONSTR_VIOL_TOL = _POST_TOL
        LPP._settle_ambiguous_unbounded = _POST_REJECT
        QPP._certify_unbounded_ray = _POST_RAY


def build():
    n = 1100
    m = dm.Model("big")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(n)]
    bs = [m.binary(f"b{i}") for i in range(0, n, 10)]
    for i in range(n):
        m.subject_to(xs[i] * xs[i] + xs[(i + 1) % n] <= 10.0)
    for k, b in enumerate(bs):
        m.subject_to(xs[k] + 2.0 * b <= 6.0)
    m.minimize(dm.sum([xs[i] for i in range(n)]) - dm.sum(bs))
    return m


REPS = 3
walls = {"pre": [], "post": []}
RUNS = 0
for rep in range(REPS):
    for arm in (("pre", "post") if rep % 2 == 0 else ("post", "pre")):
        set_arm(arm)
        t = time.perf_counter()
        r = build().solve(time_limit=8.0, gap_tolerance=1e-4)
        w = time.perf_counter() - t
        RUNS += 1
        walls[arm].append(w)
        print(f"rep{rep} {arm:4s} wall={w:6.1f}s status={r.status}", flush=True)

print(f"\nRUNS_EXECUTED={RUNS}   (test asserts wall < 48s)")
for arm in ("pre", "post"):
    a = np.array(walls[arm])
    print(f"  {arm:4s}: mean={a.mean():6.1f}s sd={a.std(ddof=1):5.1f}s "
          f"min={a.min():6.1f} max={a.max():6.1f}  over_deadline={(a >= 48).sum()}/{len(a)}")
sys.exit(0 if RUNS else 1)
