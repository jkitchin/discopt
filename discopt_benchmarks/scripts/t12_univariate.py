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

def cold_rows(make, lb, ub):
    m = make()
    terms = classify_nonlinear_terms(m)
    relax, vm = build_milp_relaxation(m, terms, DiscretizationState(),
                                      bound_override=(np.array([lb]), np.array([ub])))
    if relax._A_ub is None:
        return None, None
    A = np.asarray(sp.csr_matrix(relax._A_ub).todense())
    b = np.asarray(relax._b_ub).ravel()
    bnds = np.asarray(relax._bounds)
    # aux col is col 1 (var x is col 0). collect rows touching aux
    rows = sorted((round(A[k,0],6), round(A[k,1],6), round(b[k],6))
                  for k in range(A.shape[0]) if abs(A[k,1]) > 1e-9)
    return rows, (round(bnds[1,0],6), round(bnds[1,1],6))

# ---------- derived generators ----------
def _f(name, x):
    if name=="exp": return np.exp(x)
    if name=="log": return np.log(x)
    if name=="sqrt": return np.sqrt(x)
def _g(name, x):
    if name=="exp": return np.exp(x)
    if name=="log": return 1.0/x
    if name=="sqrt": return 0.5/np.sqrt(x)

def uni_rows(name, lb, ub):
    """Reproduce cold builder rows for f(x), arg_coeff=[1], arg_const=0.
    aux col = s. Rows in form (coeff_on_x, coeff_on_s, rhs) <= rhs.
    _add_lower_line(slope): row = [slope, -1], rhs = -intercept
    _add_upper_line(slope): row = [-slope, 1], rhs = intercept
    """
    fl, fu = _f(name,lb), _f(name,ub)
    sec_s = (fu-fl)/(ub-lb); sec_i = fl - sec_s*lb
    convex = (name=="exp")
    pts = [lb, 0.5*(lb+ub), ub]  # _tangent_points
    tp=[]
    for p in pts:
        # sqrt/log skip pt<=0 (singular slope) exactly like _tangent_points
        if name in ("sqrt","log") and p <= 0.0: continue
        if all(abs(p-q)>1e-12 for q in tp): tp.append(p)
    rows=[]
    if convex:
        for t in tp:
            s=_g(name,t); ic=_f(name,t)-s*t
            rows.append((round(s,6), -1.0, round(-ic,6)))
        rows.append((round(-sec_s,6), 1.0, round(sec_i,6)))
    else:  # concave: log, sqrt
        rows.append((round(sec_s,6), -1.0, round(-sec_i,6)))
        for t in tp:
            s=_g(name,t); ic=_f(name,t)-s*t
            rows.append((round(-s,6), 1.0, round(ic,6)))
    return sorted(rows)

def uni_bounds(name, lb, ub):
    a,b=_f(name,lb),_f(name,ub)
    return (round(min(a,b),6), round(max(a,b),6))

def make_uni(name):
    fn = {"exp":dm.exp,"log":dm.log,"sqrt":dm.sqrt}[name]
    def make():
        m=dm.Model(); x=m.continuous("x",lb=-100,ub=100); m.minimize(fn(x))
        return m
    return make

rng=np.random.default_rng(0)
for name in ["exp","log","sqrt"]:
    bad=0; total=0
    for _ in range(400):
        if name=="exp":
            lb=rng.uniform(-8,4); ub=lb+rng.uniform(0.1,6)
            if ub>13: continue
        elif name=="sqrt":  # sqrt allows lb==0; include boundary
            lb=0.0 if rng.random()<0.25 else rng.uniform(0.0,8)
            ub=lb+rng.uniform(0.1,8)
        else:  # log needs lb>0
            lb=rng.uniform(1e-3,8); ub=lb+rng.uniform(0.1,8)
        total+=1
        cr,cb=cold_rows(make_uni(name),lb,ub)
        if cr is None: continue
        mr=uni_rows(name,lb,ub); mb=uni_bounds(name,lb,ub)
        ok=(cr==mr) and abs(cb[0]-mb[0])<1e-5 and abs(cb[1]-mb[1])<1e-5
        if not ok:
            bad+=1
            if bad<=2:
                print(f"[{name}] MISMATCH box[{lb:.4f},{ub:.4f}]")
                print("  cold rows:", cr, "bnds", cb)
                print("  mine rows:", mr, "bnds", mb)
    print(f"{name}: {total-bad}/{total} pass  ({bad} mismatches)")
