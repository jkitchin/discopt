import os, sys
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np
import discopt.solvers.lp_pounce as LPP
from discopt.solvers.lp_simplex import solve_lp as simplex
from discopt.solvers import SolveStatus
CHECKS=0
hits=[]
for seed in range(60):
    for n, m, scale in ((20,10,1e7),(40,20,1e7),(20,10,1e8)):
        rng = np.random.default_rng(seed)
        A = -np.abs(rng.uniform(0.5,5.,size=(m,n)))
        b = -scale*np.abs(rng.uniform(0.5,2.,size=m))*n*0.25
        c = np.abs(rng.uniform(1.,10.,size=n))
        bounds=[(0.,10.*scale)]*n
        ref = simplex(c=c,A_ub=A,b_ub=b,bounds=bounds)
        old = LPP.solve_lp(c=c,A_ub=A,b_ub=b,bounds=bounds,options={"constr_viol_tol":1e-4})
        new = LPP.solve_lp(c=c,A_ub=A,b_ub=b,bounds=bounds)
        CHECKS+=1
        if ref.status==SolveStatus.OPTIMAL and old.status==SolveStatus.ERROR and new.status==SolveStatus.ERROR:
            hits.append((seed,n,m,scale))
print("both arms hit the impossible-UNBOUNDED path (pre-fix would certify it):")
for h in hits[:10]: print("  ", h)
print(f"CHECKS_EXECUTED={CHECKS}  hits={len(hits)}")
sys.exit(0 if CHECKS else 1)
