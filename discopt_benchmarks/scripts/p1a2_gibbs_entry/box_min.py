import sys, numpy as np
import discopt.modeling as dm
from discopt._jax.dag_compiler import compile_expression
import jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_platform_name','cpu')
NL_DIR="/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
inst=sys.argv[1]
m=dm.from_nl(f"{NL_DIR}/{inst}.nl")
obj=m._objective.expression
f=compile_expression(obj,m)
fj=jax.jit(lambda x: jnp.reshape(f(x),()))
gj=jax.jit(jax.grad(lambda x: jnp.reshape(f(x),())))
n=len(m._variables)
lb=np.array([v.lb for v in m._variables]); ub=np.array([v.ub for v in m._variables])
_=fj(jnp.asarray(lb)); _=gj(jnp.asarray(lb))
def val(x): return float(fj(jnp.asarray(x)))
def grad(x): return np.asarray(gj(jnp.asarray(x)),dtype=float)
rng=np.random.default_rng(1)
best=np.inf
bnds=list(zip(lb,ub))
for i in range(60):
    x0=lb+rng.random(n)*(ub-lb)
    try:
        res=minimize(val,x0,jac=grad,bounds=bnds,method='L-BFGS-B')
        if res.fun<best: best=res.fun
    except Exception: pass
print(f"{inst}: UNCONSTRAINED-box-min(obj) ~= {best:.4f}")
