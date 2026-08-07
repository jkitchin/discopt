import os, sys
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np
from discopt.solvers.qp_pounce import solve_qp
from discopt.solver import _matrix_solution_feasible
from discopt.solvers import SolveStatus
CHECKS = 0
for seed in range(40):
    for scale in (1e2, 1e3, 1e4):
        rng = np.random.default_rng(seed)
        n, m = 10, 6
        B = rng.normal(size=(n,n)); Q = B @ B.T + n*np.eye(n)
        c = -np.abs(rng.uniform(1.,10.,size=n))*scale
        A_ub = -np.abs(rng.uniform(0.5,5.,size=(m,n)))
        b_ub = -scale*np.abs(rng.uniform(0.5,2.,size=m))*n*0.25
        bounds=[(0., 10.*scale)]*n
        # force the PRE-FIX behaviour explicitly via the caller option
        res = solve_qp(Q=Q,c=c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,options={"constr_viol_tol":1e-4})
        CHECKS += 1
        if res.status != SolveStatus.OPTIMAL: continue
        if not _matrix_solution_feasible(res.x,A_ub,b_ub,None,None,bounds):
            print(f"TRIPS pre-fix: seed={seed} scale={scale:g}")
print(f"CHECKS_EXECUTED={CHECKS}")
sys.exit(0 if CHECKS else 1)
