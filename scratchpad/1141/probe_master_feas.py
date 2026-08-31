"""Is the OA master itself excluding the true optimum, or is the driver returning
a bound above a point the master admits? (#1141 ON-arm bound inversion.)

Captures the exact system handed to `solve_milp_with_lazy_cuts` plus every row the
separators returned, then tests the reference optimum against the WHOLE system.
Prints an executed-check count (§6).
"""
import os, sys, pathlib
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import portfolio2
from discopt._relax.model_utils import flat_variable_bounds
from discopt._tape_nlp_evaluator import make_evaluator

KW = dict(n=40, K=6, spread=0.001, cap_scale=0.7)


def flat_point(model, xdict):
    lb, _ = flat_variable_bounds(model)
    out = np.zeros(len(lb)); k = 0
    for v in model._variables:
        arr = np.atleast_1d(np.asarray(xdict[v.name], float)).ravel()
        out[k:k + v.size] = arr; k += v.size
    return out


os.environ["DISCOPT_OA_NODE_CUTS"] = "0"
m_off = portfolio2.build(**KW)
r_off = m_off.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                    time_limit=120, gap_tolerance=1e-4)
x_ref = flat_point(m_off, r_off.x)
print(f"OFF obj={r_off.objective!r} bound={r_off.bound!r}")

import discopt.solvers.milp_simplex as ms
import discopt.solvers.oa as oa

cap = {"rows": []}
_orig = ms.solve_milp_with_lazy_cuts


def wrapped(**kw):
    cap["kw"] = {k: kw.get(k) for k in ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds", "integrality")}
    for key, tag in (("node_callback", "node"), ("lazy_callback", "lazy")):
        cb = kw.get(key)
        if cb is None:
            continue

        def spy(x, _cb=cb, _tag=tag):
            rows = _cb(x)
            for co, rhs in rows or []:
                cap["rows"].append((_tag, np.asarray(co, float).copy(), float(rhs)))
            return rows

        kw[key] = spy
    return _orig(**kw)


ms.solve_milp_with_lazy_cuts = wrapped
oa.solve_milp_with_lazy_cuts = wrapped
os.environ["DISCOPT_OA_NODE_CUTS"] = "1"
m_on = portfolio2.build(**KW)
r_on = m_on.solve(solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex",
                  time_limit=120, gap_tolerance=1e-4)
print(f"ON  obj={r_on.objective!r} bound={r_on.bound!r}")

kw = cap["kw"]
c = np.asarray(kw["c"], float)
N = c.shape[0]
nv = x_ref.shape[0]
print(f"master columns={N}  model variables={nv}  callback rows={len(cap['rows'])}")

x = np.zeros(N)
x[:nv] = x_ref
checks = 0
worst = []


def check(A, b, sense):
    global checks
    if A is None:
        return
    A = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    b = np.asarray(b, float)
    for i in range(A.shape[0]):
        checks += 1
        r = float(A[i] @ x) - b[i]
        if (sense == "<=" and r > 1e-6) or (sense == "==" and abs(r) > 1e-6):
            worst.append((f"static {sense} row {i}", r))


check(kw["A_ub"], kw["b_ub"], "<=")
check(kw["A_eq"], kw["b_eq"], "==")
bl = kw["bounds"]
if bl is not None:
    for j, (lo, hi) in enumerate(bl):
        checks += 1
        if lo is not None and x[j] < lo - 1e-6:
            worst.append((f"lb col {j}", lo - x[j]))
        if hi is not None and x[j] > hi + 1e-6:
            worst.append((f"ub col {j}", x[j] - hi))
for tag, co, rhs in cap["rows"]:
    checks += 1
    r = float(co[:N] @ x[:co.shape[0]]) - rhs if co.shape[0] <= N else None
    if r is None:
        continue
    if r > 1e-6:
        worst.append((f"{tag} cut", r))


# Integrality of the reference point on the master's own integer columns.
integ = kw["integrality"]
if integ is not None:
    integ = np.asarray(integ).ravel()
    icols = np.where(integ != 0)[0]
    resid = max((abs(x[j] - round(x[j])) for j in icols), default=0.0)
    checks += 1
    print(f"master integer columns={len(icols)}  reference integrality residual={resid:.3e}")
    if resid > 1e-9:
        worst.append(("integrality", resid))

# Decisive: re-solve the captured system as a PLAIN MILP (no callbacks). Its
# optimum is what any valid dual bound from the hooked run must not exceed.
from discopt.solvers.milp_simplex import solve_milp
rows = [(co, rhs) for _t, co, rhs in cap["rows"] if co.shape[0] == N]
A_all = np.vstack([np.asarray(kw["A_ub"].toarray() if sp.issparse(kw["A_ub"]) else kw["A_ub"], float)]
                  + ([np.vstack([r[0] for r in rows])] if rows else []))
b_all = np.concatenate([np.asarray(kw["b_ub"], float)]
                       + ([np.array([r[1] for r in rows])] if rows else []))
plain = solve_milp(c=c, A_ub=A_all, b_ub=b_all, A_eq=kw["A_eq"], b_eq=kw["b_eq"],
                   bounds=kw["bounds"], integrality=kw["integrality"],
                   time_limit=180.0, gap_tolerance=1e-9)
print(f"plain re-solve of the captured system: status={plain.status} "
      f"obj={plain.objective!r} bound={plain.bound!r}  (rows={A_all.shape[0]})")

print(f"\nEXECUTED CHECKS: {checks}   VIOLATIONS: {len(worst)}")
for w in worst[:10]:
    print("   ", w)
print(f"master objective at reference point: {float(c @ x):.12g}   ON master bound proxy: {r_on.bound!r}")
if checks == 0:
    sys.exit(1)
