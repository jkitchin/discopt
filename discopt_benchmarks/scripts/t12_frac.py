# ruff: noqa  -- derivation/verification harness (math notation)
import os

os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
import discopt.modeling as dm
import numpy as np
import scipy.sparse as sp
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

_LIFT_MAX = 1e6
def _ok(s): return np.isfinite(s) and abs(s) <= _LIFT_MAX

def cold_rows(p, lb, ub):
    m=dm.Model(); x=m.continuous("x",lb=0.0001,ub=1000); m.minimize(x**p)
    t=classify_nonlinear_terms(m)
    # confirm this is classified as fractional_power
    is_fp = (len(getattr(t,"fractional_power",[]))>0)
    r,vm=build_milp_relaxation(m,t,DiscretizationState(),bound_override=(np.array([lb]),np.array([ub])))
    if r._A_ub is None: return None,None,is_fp
    A=np.asarray(sp.csr_matrix(r._A_ub).todense()); b=np.asarray(r._b_ub).ravel()
    bnds=np.asarray(r._bounds)
    # find aux col (the fractional power col). x is col0. aux is col with min/max f
    aux=None
    for c in range(1,A.shape[1]):
        if any(abs(A[k,c])>1e-9 for k in range(A.shape[0])): aux=c; break
    if aux is None: return [], None, is_fp
    rows=sorted((round(A[k,0],6),round(A[k,aux],6),round(b[k],6))
                for k in range(A.shape[0]) if abs(A[k,aux])>1e-9)
    return rows,(round(bnds[aux,0],6),round(bnds[aux,1],6)), is_fp

def frac_rows(p, lb, ub):
    """Reproduce cold fractional-power rows. aux col = a.
       concave (0<p<1): secant lower + endpoint tangent uppers.
       convex   (p>1,p<0): endpoint tangent lowers + secant upper.
       _slope_ok guard drops any ill-conditioned row.
       row form: (coeff_x, coeff_a, rhs).
       lower a>= slope*x+ic : row a=-1, x=+slope, rhs=-ic
       upper a<= slope*x+ic : row a=+1, x=-slope, rhs=+ic
    """
    fl,fu=lb**p, ub**p
    sec_s=(fu-fl)/(ub-lb); sec_i=fl-sec_s*lb
    concave = (0.0<p<1.0)
    tp=[lb,ub]
    tp=[t for t in tp if t>0.0 or (t==0.0 and p>1.0)]
    if not tp: tp=[max(lb,1e-12),ub]
    rows=[]
    if concave:
        if _ok(sec_s):
            rows.append((round(sec_s,6),-1.0,round(-sec_i,6)))
        for t in tp:
            ts=p*(t**(p-1.0))
            if not _ok(ts): continue
            tc=(1.0-p)*(t**p)
            # a - ts*x <= tc  -> (coeff_x=-ts, coeff_a=1, rhs=tc)
            rows.append((round(-ts,6),1.0,round(tc,6)))
    else:  # convex
        for t in tp:
            ts=p*(t**(p-1.0))
            if not _ok(ts): continue
            tc=(1.0-p)*(t**p)
            # -a + ts*x <= -tc -> (coeff_x=ts, coeff_a=-1, rhs=-tc)
            rows.append((round(ts,6),-1.0,round(-tc,6)))
        if _ok(sec_s):
            rows.append((round(-sec_s,6),1.0,round(sec_i,6)))
    return sorted(rows)

def frac_bounds(p,lb,ub):
    a,b=lb**p,ub**p
    return (round(min(a,b),6),round(max(a,b),6))

rng=np.random.default_rng(1)
for p in [0.5, 0.25, 0.75, 1.5, 2.5, -1.0, -0.5, -2.0]:
    bad=0; total=0; classified=0
    for _ in range(250):
        if p<0.0:
            lb=rng.uniform(0.05,5); ub=lb+rng.uniform(0.05,5)
        else:
            lb=rng.uniform(0.0,5); ub=lb+rng.uniform(0.05,6)
        if lb<=0.0 and p<0.0: continue
        total+=1
        cr,cb,is_fp=cold_rows(p,lb,ub)
        if is_fp: classified+=1
        if cr is None: continue
        mr=frac_rows(p,lb,ub); mb=frac_bounds(p,lb,ub)
        okr=(cr==mr)
        okb=(cb is None) or (abs(cb[0]-mb[0])<1e-5 and abs(cb[1]-mb[1])<1e-5)
        if not (okr and okb):
            bad+=1
            if bad<=2:
                print(f"[p={p}] MISMATCH box[{lb:.4f},{ub:.4f}]")
                print("  cold:",cr,cb)
                print("  mine:",mr,mb)
    print(f"p={p}: {total-bad}/{total} pass ({bad} mism), classified_as_fp={classified}/{total}")
