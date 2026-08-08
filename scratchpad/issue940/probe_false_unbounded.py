"""Does the pre-existing scale-1e7 POUNCE false UNBOUNDED reach a user-visible
certificate? Model level, both tolerance arms, exact-simplex oracle.

_solve_lp_matrix only DEFERS an UNBOUNDED when a declared bound sits in
[1e15, 1e20). These bounds are ~1e8, so POUNCE's verdict is certified as-is.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.modeling as dm
import discopt.solvers.lp_pounce as LPP
import discopt.solvers.lp_simplex as LPS
from discopt.solvers import SolveStatus

CHECKS = 0
scale = 1e7
for rep in (0, 3, 4):
    rng = np.random.default_rng(20940)
    # regenerate the same stream position as probe_unbounded for n=20,m=10,scale=1e7
    n, mrows = 20, 10
    A = -np.abs(rng.uniform(0.5, 5.0, size=(mrows, n)))
    b = -scale * np.abs(rng.uniform(0.5, 2.0, size=mrows)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    for _ in range(rep):
        A = -np.abs(rng.uniform(0.5, 5.0, size=(mrows, n)))
        b = -scale * np.abs(rng.uniform(0.5, 2.0, size=mrows)) * n * 0.25
        c = np.abs(rng.uniform(1.0, 10.0, size=n))
    bounds = [(0.0, 10.0 * scale)] * n
    ref = LPS.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds)
    old = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds, options={"constr_viol_tol": 1e-4})
    new = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds)
    CHECKS += 1
    if True:
        # rebuild as a model and solve end to end through nlp_solver="pounce"
        m = dm.Model("scale1e7")
        x = m.continuous("x", shape=(n,), lb=0.0, ub=10.0 * scale)
        m.minimize(dm.sum(lambda j: float(c[j]) * x[j], over=range(n)))
        for i in range(mrows):
            row = A[i]
            m.subject_to(dm.sum(lambda j: float(row[j]) * x[j], over=range(n))
                         <= float(b[i]), name=f"r{i}")
        res = m.solve(nlp_solver="pounce")
        print(f"rep={rep}: matrix simplex={ref.status.name} pounce@1e-4={old.status.name} "
              f"pounce@{LPP._CONSTR_VIOL_TOL:g}={new.status.name}  ||  model.solve() -> "
              f"{res.status!r} obj={res.objective!r}  (simplex oracle obj={ref.objective!r})")
print(f"CHECKS_EXECUTED={CHECKS}")
sys.exit(0 if CHECKS else 1)
