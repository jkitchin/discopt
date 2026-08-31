"""Per-driver-call capture: is the master the hooked run PROVED optimal actually
optimal at the value it reported? (#1141)"""
import os, sys, pathlib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
import discopt.solvers.milp_simplex as ms
import discopt.solvers.oa as oa
from discopt.solvers.milp_simplex import solve_milp

KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)
calls = []
_orig = ms.solve_milp_with_lazy_cuts


def wrapped(**kw):
    rec = {"rows": [], "kw": {k: kw.get(k) for k in
                              ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds", "integrality")}}
    calls.append(rec)
    for key, tag in (("node_callback", "node"), ("lazy_callback", "lazy")):
        cb = kw.get(key)
        if cb is None:
            continue

        def spy(x, _cb=cb, _tag=tag, _rec=rec):
            rows = _cb(x)
            for co, rhs in rows or []:
                _rec["rows"].append((_tag, np.asarray(co, float).copy(), float(rhs)))
            return rows

        kw[key] = spy
    out = _orig(**kw)
    rec["result"] = out
    return out


ms.solve_milp_with_lazy_cuts = wrapped
oa.solve_milp_with_lazy_cuts = wrapped
os.environ["DISCOPT_OA_NODE_CUTS"] = "1"
m = portfolio2.build(**KW)
r = m.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
            time_limit=180, gap_tolerance=1e-4)
print(f"ON: obj={r.objective!r} bound={r.bound!r} status={r.status}")
print(f"driver calls: {len(calls)}")

checks = 0
for i, rec in enumerate(calls):
    kw = rec["kw"]
    res = rec["result"]
    A = kw["A_ub"]
    A = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    b = np.asarray(kw["b_ub"], float)
    N = np.asarray(kw["c"], float).shape[0]
    rows = [(co, rhs) for _t, co, rhs in rec["rows"] if co.shape[0] == N]
    A_all = np.vstack([A] + ([np.vstack([x[0] for x in rows])] if rows else []))
    b_all = np.concatenate([b] + ([np.array([x[1] for x in rows])] if rows else []))
    plain = solve_milp(c=kw["c"], A_ub=A_all, b_ub=b_all, A_eq=kw["A_eq"], b_eq=kw["b_eq"],
                       bounds=kw["bounds"], integrality=kw["integrality"],
                       time_limit=240.0, gap_tolerance=1e-9)
    checks += 1
    print(f"call {i}: hooked status={res.status.name} obj={res.objective!r} bound={res.bound!r} "
          f"rows={A_all.shape[0]} (static {A.shape[0]} + sep {len(rows)})")
    print(f"        plain re-solve status={plain.status.name} obj={plain.objective!r} "
          f"bound={plain.bound!r}")
    if res.bound is not None and plain.objective is not None and res.bound > plain.objective + 1e-9:
        print(f"        *** hooked dual bound {res.bound!r} is ABOVE a feasible master "
              f"point {plain.objective!r} -- FALSE BOUND")

print(f"\nEXECUTED CHECKS: {checks}")
sys.exit(1 if checks == 0 else 0)
