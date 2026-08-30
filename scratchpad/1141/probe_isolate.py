"""Is the #1141 false bound in the PLUMBING (requeue/fold) or in the CUTS?

Runs the portfolio master with the node hook wired to a separator whose rows are
valid by construction and cut nothing (nonnegative combinations of the master's
own rows). The requeue/fold path runs exactly as in the real arm; the relaxation
does not change. If the certified optimum still moves, the plumbing is at fault.
"""
import os, sys, pathlib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
import discopt.solvers.milp_simplex as ms
import discopt.solvers.oa as oa

KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)
MODE = sys.argv[1] if len(sys.argv) > 1 else "surrogate"

_orig = ms.solve_milp_with_lazy_cuts
rng = np.random.default_rng(3)
fired = {"n": 0}


def wrapped(**kw):
    A = kw.get("A_ub")
    A = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    b = np.asarray(kw.get("b_ub"), float)

    if MODE == "surrogate":
        def surrogate(x):
            fired["n"] += 1
            lam = rng.random(A.shape[0]) * (rng.random(A.shape[0]) < 0.3)
            if lam.sum() <= 0:
                lam = np.ones(A.shape[0]) / A.shape[0]
            return [(lam @ A, float(lam @ b))]
        kw["node_callback"] = surrogate
        kw["node_hook_rounds"] = 2
        kw["node_hook_cut_cap"] = 500
    return _orig(**kw)


ms.solve_milp_with_lazy_cuts = wrapped
oa.solve_milp_with_lazy_cuts = wrapped
os.environ["DISCOPT_OA_NODE_CUTS"] = "1"
m = portfolio2.build(**KW)
r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
            time_limit=180, gap_tolerance=1e-4)
print(f"MODE={MODE}  obj={r.objective!r} bound={r.bound!r} status={r.status} "
      f"node_cb_calls={fired['n']}")
print("reference (node cuts OFF) optimum is -0.10089619806602235")
