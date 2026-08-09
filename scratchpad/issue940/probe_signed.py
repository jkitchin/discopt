"""Issue #940: SIGNED objective error of the accepted POUNCE point against the
exact-simplex oracle, both tolerance arms, interleaved in one process.

For a MINIMIZE LP a negative error means the reported objective sits BELOW the
true optimum (super-optimal, i.e. bought with residual infeasibility) — the
#850 failure mode. ``_solve_lp_matrix`` reports ``bound = objective``, so this is
the quantity that decides whether the returned bound is trustworthy.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.solvers.lp_pounce as LPP
from discopt.solvers.lp_simplex import solve_lp as simplex
from discopt.solvers import SolveStatus

assert LPP._CONSTR_VIOL_TOL == 1e-8
COMPARISONS = 0
old_err, new_err = [], []
rng = np.random.default_rng(20940)
for n, m in ((5, 3), (10, 6), (20, 10), (40, 20)):
    for scale in (1e0, 1e2, 1e3, 1e4, 1e5, 1e6):
        for rep in range(3):
            A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
            b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
            c = np.abs(rng.uniform(1.0, 10.0, size=n))
            bounds = [(0.0, 10.0 * scale)] * n
            ref = simplex(c=c, A_ub=A, b_ub=b, bounds=bounds)
            if ref.status != SolveStatus.OPTIMAL:
                continue
            o = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds,
                             options={"constr_viol_tol": 1e-4})
            nw = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds)
            if o.status != SolveStatus.OPTIMAL or nw.status != SolveStatus.OPTIMAL:
                continue
            COMPARISONS += 1
            d = max(1.0, abs(ref.objective))
            old_err.append((o.objective - ref.objective) / d)
            new_err.append((nw.objective - ref.objective) / d)
for label, e in (("pre-fix  (constr_viol_tol=1e-4)", np.array(old_err)),
                 ("post-fix (constr_viol_tol=1e-8)", np.array(new_err))):
    print(f"{label}: signed rel error vs simplex optimum  "
          f"min={e.min():+.3e}  max={e.max():+.3e}  "
          f"mean|e|={np.abs(e).mean():.3e}  n_super_optimal={(e < -1e-9).sum()}")
print(f"COMPARISONS_EXECUTED={COMPARISONS}")
sys.exit(0 if COMPARISONS else 1)
