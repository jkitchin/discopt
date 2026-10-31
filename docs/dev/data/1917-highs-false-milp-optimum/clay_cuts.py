"""Which lp_nlp_bb master row cuts off the certified optimum on clay0203 (big-M)?"""
import sys, warnings
import numpy as np
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks"); warnings.simplefilter("ignore")
from discopt.solvers import milp_highs, milp_simplex
from discopt.pyomo.gdp import convert_pyomo_gdp
from benchmarks import gdplib_runner as gr
CAP = {}
def wrap(mod):
    real = mod.solve_milp_with_lazy_cuts
    def w(**kw):
        CAP.update(base=kw, cuts=[], backend=mod.__name__)
        for key in ("lazy_callback", "node_callback"):
            cb = kw.get(key)
            if cb is not None:
                def cbw(*a, _cb=cb, _k=key, **k):
                    out = _cb(*a, **k)
                    try:
                        for row, rhs in (out or []): CAP["cuts"].append((_k, np.asarray(row, float).copy(), float(rhs)))
                    except Exception as e:
                        CAP.setdefault("unparsed", []).append(repr(type(out)))
                    return out
                kw[key] = cbw
        res = real(**kw); CAP['result'] = res; return res
    mod.solve_milp_with_lazy_cuts = w
wrap(milp_highs); wrap(milp_simplex)
name = "pyomo.constrained_layout_clay0203"
m = convert_pyomo_gdp(gr.discover_models(include=[name])[0].builder()).model
r = m.solve(gdp_method="big-m", time_limit=120)
print("final", r.status, r.objective, "backend", CAP.get("backend"), "lazy rows", len(CAP.get("cuts", [])), "unparsed", CAP.get("unparsed", [])[:2])
b = CAP["base"]; n = len(b["c"])
# optimum in master space: model variables flattened in order (+ eta columns if any)
xs = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
print("master cols", n, "model scalars", len(xs))
x = np.zeros(n); x[: min(n, len(xs))] = xs[: min(n, len(xs))]
if n > len(xs): x[len(xs):] = r.objective  # epigraph eta
viol = []; checks = 0
def chk(kind, A, ub, i=None):
    global checks
    A = np.asarray(A.todense() if hasattr(A, "todense") else A, float)
    if A.ndim == 1: A = A[None, :]
    for k in range(A.shape[0]):
        checks += 1
        s = A[k] @ x - ub[k]
        if s > 1e-5 * (1 + abs(ub[k])): viol.append((kind, k if i is None else i, float(s)))
if b.get("A_ub") is not None and np.size(b["b_ub"]): chk("A_ub", b["A_ub"], np.asarray(b["b_ub"], float))
lo = np.array([bd[0] if bd[0] is not None else -np.inf for bd in b["bounds"]]); hi = np.array([bd[1] if bd[1] is not None else np.inf for bd in b["bounds"]])
bv = [(j, x[j], lo[j], hi[j]) for j in range(n) if x[j] < lo[j] - 1e-6 or x[j] > hi[j] + 1e-6]
for i, (k, row, rhs) in enumerate(CAP["cuts"]):
    if row.shape[0] == n: chk(k, row, np.array([rhs]), i)
if b.get("A_eq") is not None and np.size(b["b_eq"]):
    Ae = np.asarray(b["A_eq"].todense() if hasattr(b["A_eq"], "todense") else b["A_eq"], float); be = np.asarray(b["b_eq"], float)
    res = Ae @ x - be; checks += len(be); print("A_eq max |res|", float(np.max(np.abs(res))) if len(be) else None)
c = np.asarray(b["c"], float)
print("c.x* =", float(c @ x), "objective", r.objective, "diff", r.objective - float(c @ x))
mr = CAP["result"]; print("master status", mr.status, "bound", mr.bound, "obj", getattr(mr, "objective", None)); cs = mr.callback_stats or {}; print("callback_stats", {k: cs[k] for k in cs if not isinstance(cs[k], (list, np.ndarray))})
print("checks", checks, "violated", len(viol), "bound violations", len(bv))
for v in viol[:8]: print("  ", v)
for v in bv[:5]: print("  bound", v)
# --- independent check: fixed-integer LP of the captured master rows, and the full MILP
import highspy, scipy.sparse as sp
def dense(A): return np.asarray(A.todense() if hasattr(A, "todense") else A, float)
rows = []; rhs = []
if b.get("A_ub") is not None and np.size(b["b_ub"]):
    rows.append(dense(b["A_ub"])); rhs.append(np.asarray(b["b_ub"], float))
for k, row, rr in CAP["cuts"]:
    rows.append(row[None, :]); rhs.append(np.array([rr]))
A = np.vstack(rows); bu = np.concatenate(rhs)
Ae = dense(b["A_eq"]) if b.get("A_eq") is not None and np.size(b["b_eq"]) else np.zeros((0, n)); be = np.asarray(b["b_eq"], float) if Ae.shape[0] else np.zeros(0)
integ = np.asarray(b["integrality"]) if b.get("integrality") is not None else np.zeros(n)
START = {}
def solve(fix_int, mip, **opts):
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    for k_, v_ in opts.items(): h.setOptionValue(k_, v_)
    l2 = lo.copy(); h2 = hi.copy()
    if fix_int:
        for j in np.nonzero(integ)[0]: l2[j] = h2[j] = round(x[j])
    l2 = np.where(np.isfinite(l2), l2, -highspy.kHighsInf); h2 = np.where(np.isfinite(h2), h2, highspy.kHighsInf)
    lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = A.shape[0] + Ae.shape[0]
    lp.col_cost_ = c; lp.col_lower_ = l2; lp.col_upper_ = h2
    M = sp.csc_matrix(np.vstack([A, Ae]))
    lp.row_lower_ = np.concatenate([np.full(A.shape[0], -highspy.kHighsInf), be]); lp.row_upper_ = np.concatenate([bu, be])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = M.indptr; lp.a_matrix_.index_ = M.indices; lp.a_matrix_.value_ = M.data
    if mip: lp.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous for t in integ]
    h.passModel(lp)
    if opts.pop("_start", False) if False else START.get("on"):
        sol = highspy.HighsSolution(); sol.col_value = list(x); sol.value_valid = True; h.setSolution(sol)
    if START.get("write"): h.writeModel(START["write"]); START["write"] = None
    h.run()
    return h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value, h.getInfo().mip_dual_bound if mip else None
print("fixed-int LP:", solve(True, False))
print("full MILP (all captured rows):", solve(False, True))
print("  presolve off:", solve(False, True, presolve="off"))
print("  feas tol 1e-9:", solve(False, True, mip_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9))
print("  presolve off + tight:", solve(False, True, presolve="off", mip_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9))
res = A @ x - bu; order = np.argsort(-res)
print("max violation at x*: %.3e; rows violated >1e-9: %d, >1e-7: %d, >1e-6: %d" % (res.max(), (res>1e-9).sum(), (res>1e-7).sum(), (res>1e-6).sum()))
nb = A.shape[0] - len(CAP["cuts"])
for i in order[:5]:
    src = "base A_ub" if i < nb else CAP["cuts"][i-nb][0]
    nz = np.nonzero(A[i])[0]
    print("   row", i, src, "viol %.3e" % res[i], "nnz", len(nz), "min|a| %.2e" % np.min(np.abs(A[i][nz])), "rhs %.4g" % bu[i])
print("fixed-int LP tight:", solve(True, False, primal_feasibility_tolerance=1e-9))
START["on"] = True; print("MILP with x* as start:", solve(False, True)); START["on"] = False
START["write"] = "docs/dev/data/1917-highs-false-milp-optimum/clay0203_master.mps"; solve(False, True)
A_orig, bu_orig = A.copy(), bu.copy()
for thr in (1e-12, 1e-9, 1e-7):
    A = A_orig.copy(); bu = bu_orig.copy()
    small = (A != 0) & (np.abs(A) < thr)
    xmax = np.maximum(np.abs(np.where(np.isfinite(lo), lo, 0)), np.abs(np.where(np.isfinite(hi), hi, 0)))
    inf_cols = ~(np.isfinite(lo) & np.isfinite(hi))
    bad = small & inf_cols[None, :]
    bu = bu + (np.abs(np.where(small & ~inf_cols[None, :], A, 0.0)) @ xmax)
    A[small & ~inf_cols[None, :]] = 0.0
    print("thr %.0e: dropped %d coefs (%d on unbounded cols kept)" % (thr, int((small & ~inf_cols[None,:]).sum()), int(bad.sum())), "MILP:", solve(False, True), "tight:", solve(False, True, mip_feasibility_tolerance=1e-9, primal_feasibility_tolerance=1e-9))
A, bu = A_orig, bu_orig
absA = np.abs(A[A != 0]); print("coef range |a|: min %.3g max %.3g; rhs max %.3g; rows %d" % (absA.min(), absA.max(), np.max(np.abs(bu)), A.shape[0]))
print("integer cols", int(np.sum(integ != 0)), "x* int values", [round(x[j], 6) for j in np.nonzero(integ)[0]][:20])
print("version", highspy.Highs().version() if hasattr(highspy.Highs(), "version") else None)

sys.exit(0 if checks else 3)
