"""Rust MILP on an MPS model (default kwargs): root bound (nodes=2) and a timed run. usage: rtight.py <mps> <T>"""
import sys, time
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python"), discopt.__file__
import numpy as np, scipy.sparse as sp, highspy
from discopt._rust import solve_milp_csc_py
from discopt.solvers.milp_simplex import _marshal_std_form
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(sys.argv[1]); lp = h.getLp()
A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_), shape=(lp.num_row_, lp.num_col_)).tocsr()
rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_); INF = 1e30
eq = (rl == ru); up = (~eq) & (ru < INF); lo = (~eq) & (rl > -INF)
A_ub = sp.vstack([A[up], -A[lo]]).tocsr(); b_ub = np.r_[ru[up], -rl[lo]]
cl, cu = np.array(lp.col_lower_), np.array(lp.col_upper_)
bounds = [(None if a <= -INF else a, None if b >= INF else b) for a, b in zip(cl, cu)]
integ = np.array([1 if t == highspy.HighsVarType.kInteger else 0 for t in lp.integrality_])
c = np.array(lp.col_cost_); n = len(c)
std = _marshal_std_form(c, A_ub, b_ub, A[eq].tocsr(), ru[eq], bounds, integ)
print(sys.argv[1].split("/")[-1], "n", n, "m", std.m, "ints", integ.sum(), "offset", lp.offset_, flush=True)
def go(tag, nodes, T):
    t = time.perf_counter()
    st, x, obj, bound, nn, it = solve_milp_csc_py(std.c, std.m, n + std.m, std.col_ptr, std.row_idx, std.vals, std.b, std.lb, std.ub,
        std.int_cols, n, 0.0, nodes, 1e-4, time_limit_s=T)
    print(f"{tag:12s} {st:12s} inc {obj+lp.offset_:14.6f} bound {bound+lp.offset_:14.6f} nodes {nn:7d} {time.perf_counter()-t:6.2f}s", flush=True)
go("root", 2, 60.0); go("run", 10**9, float(sys.argv[2]))
print("PROBE_DONE runs=2")
