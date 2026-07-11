import sys, numpy as np
import discopt.modeling as dm
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.alphabb import rigorous_alpha, alphabb_underestimator
import jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_platform_name','cpu')
NL_DIR="/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
inst=sys.argv[1]
m=dm.from_nl(f"{NL_DIR}/{inst}.nl")
obj=m._objective.expression
f=compile_expression(obj,m)
n=len(m._variables)
lb=np.array([v.lb for v in m._variables]); ub=np.array([v.ub for v in m._variables])
# rigorous alpha (interval Gershgorin) for the WHOLE objective
try:
    alpha=np.asarray(rigorous_alpha(obj,m),dtype=float).ravel()
except Exception as e:
    print(f"{inst}: rigorous_alpha FAILED: {type(e).__name__} {e}"); sys.exit()
if alpha.size==1: alpha=np.full(n,float(alpha))
print(f"{inst}: alpha (max) = {alpha.max():.4g}, width-scaled max perturb = {np.max(alpha*(ub-lb)**2/4):.4g}")
# minimize the convex underestimator L(x)=f(x)-sum alpha_i (x-lb)(ub-x) over box
fj=jax.jit(lambda x: jnp.reshape(f(x),()))
_=fj(jnp.asarray(lb))
def L(x):
    x=np.asarray(x,dtype=float)
    pert=float(np.sum(alpha*(x-lb)*(ub-x)))
    return float(fj(jnp.asarray(x)))-pert
bnds=list(zip(lb,ub))
rng=np.random.default_rng(2)
best=np.inf
for i in range(40):
    x0=lb+rng.random(n)*(ub-lb)
    try:
        res=minimize(L,x0,bounds=bnds,method='L-BFGS-B')
        if res.fun<best: best=res.fun
    except Exception: pass
print(f"{inst}: JOINT-alphaBB box-min underestimator ~= {best:.4f}")
