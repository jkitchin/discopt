"""G3.2 entry experiment: does FBBT + big-M coefficient tightening move the MTP master root LP?
Rows: A_ub x <= b_ub, A_eq x == b_eq. Reports root LP (HiGHS LP via linprog) at each stage."""
import pickle, sys, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
d = pickle.load(open(sys.argv[1], "rb"))
c = np.asarray(d["c"], float); I = np.asarray(d["integrality"]) > 0; n = len(c)
lb = np.array([-np.inf if b[0] is None else b[0] for b in d["bounds"]], float)
ub = np.array([np.inf if b[1] is None else b[1] for b in d["bounds"]], float)
ub[ub >= 1e19] = np.inf
Aub = d["A_ub"].tocsr().astype(float); bub = np.asarray(d["b_ub"], float)
Aeq = d["A_eq"].tocsr().astype(float); beq = np.asarray(d["b_eq"], float)
def lp(Aub, bub, lb, ub):
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, np.where(np.isinf(ub), None, ub))), method="highs")
    assert r.status == 0, r.message; return r.fun
# rows as (A, b) pairs with sense <=; equality as two <= rows for FBBT
rows = [(Aub, bub), (Aeq, beq), (-Aeq, -beq)]
def fbbt(lb, ub, sweeps=20):
    lb, ub = lb.copy(), ub.copy(); changed_total = 0
    for _ in range(sweeps):
        ch = 0
        for A, b in rows:
            for i in range(A.shape[0]):
                s, e = A.indptr[i], A.indptr[i+1]; cols = A.indices[s:e]; a = A.data[s:e]
                lo = np.where(a > 0, a * lb[cols], a * ub[cols])  # min activity per term
                if np.isinf(lo).sum() > 1: continue
                tot = lo[np.isfinite(lo)].sum()
                for k, j in enumerate(cols):
                    if np.isinf(lo).sum() == 1 and np.isfinite(lo[k]): continue
                    rest = tot - (lo[k] if np.isfinite(lo[k]) else 0.0)
                    bnd = (b[i] - rest) / a[k]
                    if a[k] > 0:
                        if I[j]: bnd = np.floor(bnd + 1e-9)
                        if bnd < ub[j] - 1e-9 * (1 + abs(bnd)): ub[j] = bnd; ch += 1
                    else:
                        if I[j]: bnd = np.ceil(bnd - 1e-9)
                        if bnd > lb[j] + 1e-9 * (1 + abs(bnd)): lb[j] = bnd; ch += 1
        changed_total += ch
        if ch == 0: break
    return lb, ub, changed_total
def coef_tighten(A, b, lb, ub):
    """SCIP/HiGHS-style: row a.x <= b, binary y in row with coef cy. maxact of the rest (over bounds)."""
    A = A.tolil(copy=True); b = b.copy(); nt = 0; dsum = 0.0
    Acsr = A.tocsr()
    for i in range(A.shape[0]):
        s, e = Acsr.indptr[i], Acsr.indptr[i+1]; cols = list(Acsr.indices[s:e]); a = list(Acsr.data[s:e])
        hi = np.array([aa * (ub[j] if aa > 0 else lb[j]) for aa, j in zip(a, cols)])
        if np.isinf(hi).any(): continue
        maxact = hi.sum()
        if maxact <= b[i] + 1e-9: continue  # redundant row
        for k, j in enumerate(cols):
            if not (I[j] and lb[j] == 0 and ub[j] == 1): continue
            cy = a[k]; rest = maxact - hi[k]
            if cy < 0:   # y=1 relaxes: rest <= b - cy ; tighten when rest_max < b - cy
                dlt = (b[i] - cy) - rest
                if dlt > 1e-7 * (1 + abs(b[i])) and rest > b[i]:  # keep y=0 branch binding
                    A[i, j] = cy + dlt; nt += 1; dsum += dlt
                    a[k] = cy + dlt; hi[k] = 0.0; maxact = hi.sum()
            elif cy > 0:  # y=0 relaxes: rest <= b ; y=1 gives rest <= b - cy
                dlt = b[i] - rest
                if dlt > 1e-7 * (1 + abs(b[i])) and rest > b[i] - cy:
                    A[i, j] = cy - dlt; b[i] -= dlt; nt += 1; dsum += dlt
                    a[k] = cy - dlt; hi[k] = a[k]; maxact = hi.sum()
    return A.tocsr(), b, nt
r0 = lp(Aub, bub, lb, ub); print(f"root LP original            {r0:.6f}", flush=True)
l1, u1, nch = fbbt(lb, ub); print(f"FBBT bound changes {nch}; finite ub cont {np.isfinite(u1[~I]).sum()} (was {np.isfinite(ub[~I]).sum()})")
r1 = lp(Aub, bub, l1, u1); print(f"root LP after FBBT          {r1:.6f}", flush=True)
A2, b2, nt = Aub, bub, 0
for rnd in range(5):
    A2, b2, k = coef_tighten(A2, b2, l1, u1); nt += k
    if k == 0: break
r2 = lp(A2, b2, l1, u1); print(f"coef tightenings {nt}; root LP after FBBT+coef   {r2:.6f}", flush=True)
# OBBT upper bounds on continuous columns (strongest bound-tightening proxy)
l3, u3 = l1.copy(), u1.copy(); nob = 0
for j in np.where(~I)[0]:
    e = np.zeros(n); e[j] = -1.0
    r = linprog(e, A_ub=A2, b_ub=b2, A_eq=Aeq, b_eq=beq, bounds=list(zip(l3, np.where(np.isinf(u3), None, u3))), method="highs")
    if r.status == 0 and -r.fun < u3[j] - 1e-6: u3[j] = -r.fun; nob += 1
A4, b4, nt4 = A2, b2, 0
for rnd in range(5):
    A4, b4, k = coef_tighten(A4, b4, l3, u3); nt4 += k
    if k == 0: break
r4 = lp(A4, b4, l3, u3); print(f"OBBT ub changes {nob}; coef {nt4}; root LP after OBBT+coef {r4:.6f}")
print(f"reference: optimum -6797.5397; HiGHS root after cuts -6929.5; checks=5")
import os
if os.environ.get("ABL") == "fbbt_only": A4, b4, l3, u3 = Aub, bub, l1, u1
if os.environ.get("ABL") == "fbbt_coef": A4, b4, l3, u3 = A2, b2, l1, u1
if os.environ.get("ABL") == "obbt_nocoef": A4, b4 = Aub, bub
print("ABL", os.environ.get("ABL"))
if len(sys.argv) > 2:  # dump the tightened model (A4,b4 / Aeq,beq / l3,u3) to MPS for a HiGHS presolve-off cut run
    import highspy
    A = sp.vstack([A4, Aeq]).tocsc(); inf = highspy.kHighsInf
    rl = np.r_[np.full(A4.shape[0], -inf), beq]; ru = np.r_[b4, beq]
    lpm = highspy.HighsLp(); lpm.num_col_ = n; lpm.num_row_ = A.shape[0]; lpm.col_cost_ = c
    lpm.col_lower_ = np.where(np.isinf(l3), -inf, l3); lpm.col_upper_ = np.where(np.isinf(u3), inf, u3)
    lpm.row_lower_ = rl; lpm.row_upper_ = ru; lpm.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lpm.a_matrix_.start_ = A.indptr; lpm.a_matrix_.index_ = A.indices; lpm.a_matrix_.value_ = A.data
    lpm.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous for t in I]
    lpm.offset_ = float(d.get("obj_const", 0.0)) if isinstance(d, dict) else 0.0
    h = highspy.Highs(); h.setOptionValue("output_flag", False); assert h.passModel(lpm) == highspy.HighsStatus.kOk
    h.writeModel(sys.argv[2]); print("wrote", sys.argv[2], "max|a|", np.abs(A.data).max())
