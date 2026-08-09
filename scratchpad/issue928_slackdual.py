"""Issue #928 entry experiment, step 3: can a dual-feasible SLACK basis make the
cold deadline-bound LP solvable by the (anytime, monotone-bound) dual simplex?

Construct: slacks basic, structural columns nonbasic at the bound matching the
sign of c_j (c_j>0 -> AT_LOWER, c_j<0 -> AT_UPPER, c_j=0 -> AT_LOWER, free -> 0
handled by nb_value). Report how many columns lack the finite bound dual
feasibility needs, then solve with that in_basis and compare to p*.

Usage: python issue928_slackdual.py <lps.pkl> <lp_index>
"""

import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt  # noqa: E402
from discopt.solvers.milp_simplex import solve_lp_warm_std  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

AT_LOWER, BASIC, AT_UPPER = 0, 1, 2
INF = 1e20

pkl, idx = sys.argv[1], int(sys.argv[2])
with open(pkl, "rb") as fh:
    data = pickle.load(fh)
rec = data["lps"][idx]
c = np.asarray(rec["c"], dtype=np.float64).ravel()
n = c.shape[0]
m = rec["m"]
lb = np.array([lo for lo, _ in rec["bounds"]], dtype=np.float64)
ub = np.array([hi for _, hi in rec["bounds"]], dtype=np.float64)

# Dual-feasibility eligibility of the sign-matched slack basis.
need_lb = c > 0
need_ub = c < 0
viol = (need_lb & (lb <= -INF)) | (need_ub & (ub >= INF))
print(f"LP{idx}: n={n} m={m}; columns violating dual feasibility at slack basis: {viol.sum()}")

col_status = np.where(need_ub, AT_UPPER, AT_LOWER).astype(np.int8)
col_status = np.concatenate([col_status, np.full(m, BASIC, dtype=np.int8)])
basic_vars = np.arange(n, n + m, dtype=np.int64)

args = (rec["c"], rec["A_ub"], rec["b_ub"], rec["bounds"])

t0 = time.perf_counter()
res, _, cert = solve_lp_warm_std(*args, None, return_cert=True, time_limit=None)
w_cold = time.perf_counter() - t0
p_cold = None if res is None else res.objective
print(f"cold primal: wall={w_cold:.3f}s p*={p_cold}")

t0 = time.perf_counter()
res2, _, cert2 = solve_lp_warm_std(
    *args, (col_status, basic_vars), return_cert=True, time_limit=None
)
w_dual = time.perf_counter() - t0
p_dual = None if res2 is None else res2.objective
print(f"slack-dual basis: wall={w_dual:.3f}s p*={p_dual} status={None if res2 is None else res2.status}")

if p_cold is not None and p_dual is not None:
    print(f"objective agreement: |diff|={abs(p_cold - p_dual):.3g}")
    print("comparisons_executed=1")
else:
    print("comparisons_executed=0")
    sys.exit(1)
