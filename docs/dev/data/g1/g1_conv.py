import time, json, sys
import numpy as np
NL="python/tests/data/minlplib_nl/nvs05.nl"
def bench(fn,N=30000,warm=100):
    for _ in range(warm): fn()
    t0=time.perf_counter_ns()
    for _ in range(N): fn()
    return (time.perf_counter_ns()-t0)/N/1e3
def main():
    import jax, jax.numpy as jnp
    import discopt._jax.nlp_evaluator as nev
    from discopt.modeling.core import from_nl
    m=from_nl(NL); ev=nev.cached_evaluator(m); n=ev.n_variables
    rng=np.random.default_rng(3); lb,ub=ev.variable_bounds
    lb=np.where(np.isfinite(lb),lb,-5.); ub=np.where(np.isfinite(ub),ub,5.)
    x=(lb+rng.uniform(size=n)*(ub-lb)).astype(np.float64)
    obj=ev._obj_fn_jit; cons=ev._cons_fn_jit; p=ev._current_params()
    y=obj(x,p)  # jax scalar
    cv=cons(x,p) # jax array
    R={}
    R["float(y)"]=bench(lambda: float(y))
    R["y.item()"]=bench(lambda: y.item())
    R["np.asarray(y)"]=bench(lambda: np.asarray(y))
    R["np.asarray(y).item? via float"]=bench(lambda: np.float64(y))
    R["np.asarray(cv)"]=bench(lambda: np.asarray(cv))
    R["np.array(cv)"]=bench(lambda: np.array(cv))
    R["cv.__array__()"]=bench(lambda: cv.__array__())
    # device_get variants
    R["jax.device_get(y)"]=bench(lambda: jax.device_get(y))
    R["jax.device_get((y,cv))"]=bench(lambda: jax.device_get((y,cv)))
    R["np.asarray both"]=bench(lambda: (np.asarray(y), np.asarray(cv)))
    # single fused dispatch returning concatenated array [f, c...] then one asarray
    fcat=jax.jit(lambda xx,pp: jnp.concatenate([jnp.reshape(obj(xx,pp),(1,)), cons(xx,pp)]))
    jax.block_until_ready(fcat(x,p))
    def viacat():
        arr=np.asarray(fcat(x,p)); return arr[0], arr[1:]
    R["fused-concat 1 dispatch 1 asarray"]=bench(viacat)
    print(json.dumps(R,indent=2))
main()
