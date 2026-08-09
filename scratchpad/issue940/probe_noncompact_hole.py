"""Does the false-UNBOUNDED hole survive OUTSIDE the compact box?

The #940 fix refuses UNBOUNDED only when every bound is finite. If POUNCE's
ambiguous code-3/4 exit can also be hit on a box with one infinite bound whose
true status is OPTIMAL, then discopt still manufactures a false certificate and
the residual defect is discopt's status map, not POUNCE's convergence.

The objective is min c'x with c >= 0 and x >= 0, so no ray can decrease it: the
LP is bounded below BY CONSTRUCTION regardless of any infinite upper bound.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.solvers.lp_pounce as LPP
from discopt.solvers.lp_simplex import solve_lp as simplex
from discopt.solvers import SolveStatus

CHECKS = 0
false_unbounded = []
for seed in range(30):
    for n, m, scale in ((20, 10, 1e7), (40, 20, 1e7), (20, 10, 1e8)):
        rng = np.random.default_rng(seed)
        A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))          # -Ax <= -b  (i.e. Ax >= b)
        b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
        c = np.abs(rng.uniform(1.0, 10.0, size=n))               # c >= 0, minimize
        # ONE variable left unbounded above -> box is not compact.
        bounds = [(0.0, np.inf)] + [(0.0, 10.0 * scale)] * (n - 1)
        ref = simplex(c=c, A_ub=A, b_ub=b, bounds=bounds)
        res = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds)
        CHECKS += 1
        if ref.status == SolveStatus.OPTIMAL and res.status == SolveStatus.UNBOUNDED:
            false_unbounded.append((seed, n, m, scale, ref.objective))

print(f"CHECKS_EXECUTED={CHECKS}")
print(f"false UNBOUNDED on a NON-compact box (true status optimal): {len(false_unbounded)}")
for r in false_unbounded[:6]:
    print(f"   seed={r[0]} n={r[1]} m={r[2]} scale={r[3]:g}  true optimum={r[4]!r}")

# --- model level: does it reach a user-visible certificate? -------------------
import discopt.modeling as dm
rng = np.random.default_rng(0)
n, m, scale = 20, 10, 1e7
A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
c = np.abs(rng.uniform(1.0, 10.0, size=n))
mdl = dm.Model("noncompact")
x = mdl.continuous("x", shape=(n,), lb=0.0, ub=np.inf)
mdl.minimize(dm.sum(lambda j: float(c[j]) * x[j], over=range(n)))
for i in range(m):
    row = A[i]
    mdl.subject_to(dm.sum(lambda j: float(row[j]) * x[j], over=range(n)) <= float(b[i]),
                   name=f"r{i}")
res = mdl.solve(nlp_solver="pounce")
ref = simplex(c=c, A_ub=A, b_ub=b, bounds=[(0.0, np.inf)] * n)
print(f"\nMODEL LEVEL: model.solve() -> {res.status!r} obj={res.objective!r} | "
      f"exact simplex -> {ref.status.name} obj={ref.objective!r}")

sys.exit(0 if CHECKS else 1)
