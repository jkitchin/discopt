import time, json, sys
import numpy as np

NL="python/tests/data/minlplib_nl/nvs05.nl"

def bench(fn, N=20000, warm=50):
    for _ in range(warm): fn()
    t0=time.perf_counter_ns()
    for _ in range(N): fn()
    return (time.perf_counter_ns()-t0)/N/1e3  # us/call

def main():
    import jax, jax.numpy as jnp
    import discopt._jax.nlp_evaluator as nev
    from discopt.modeling.core import from_nl
    m=from_nl(NL); ev=nev.cached_evaluator(m); n=ev.n_variables
    ev._ensure_coo_cache()
    print(f"n_vars={n} n_cons={ev.n_constraints} n_params={len(ev._parameters)}", file=sys.stderr)
    rng=np.random.default_rng(3)
    lb,ub=ev.variable_bounds
    lb=np.where(np.isfinite(lb),lb,-5.); ub=np.where(np.isfinite(ub),ub,5.)
    x=(lb+rng.uniform(size=n)*(ub-lb)).astype(np.float64)
    obj=ev._obj_fn_jit; cons=ev._cons_fn_jit; grad=ev._grad_fn_jit; jac=ev._jac_fn_jit
    p=ev._current_params()

    R={}
    R["_current_params()"]=bench(lambda: ev._current_params())
    R["obj dispatch only (no block)"]=bench(lambda: obj(x,p))
    R["float(obj())"]=bench(lambda: float(obj(x,p)))
    R["np.asarray(cons())"]=bench(lambda: np.asarray(cons(x,p)))
    R["obj+cons separate (float+asarray)"]=bench(lambda: (float(obj(x,p)), np.asarray(cons(x,p))))
    def fc():
        a,b=obj(x,p),cons(x,p); return jax.device_get((a,b))
    R["fused device_get((obj,cons))"]=bench(fc)
    def fc2():
        a,b=obj(x,p),cons(x,p); return float(a), np.asarray(b)
    R["fused-jit? no: 2 dispatch then float+asarray"]=bench(fc2)
    fcj=jax.jit(lambda xx,pp:(obj(xx,pp),cons(xx,pp)))
    jax.block_until_ready(fcj(x,p))
    def fc3():
        a,b=fcj(x,p); return float(a), np.asarray(b)
    R["jit-fused then float+asarray"]=bench(fc3)
    def fc4():
        a,b=fcj(x,p); return jax.device_get((a,b))
    R["jit-fused then device_get"]=bench(fc4)
    # full: obj,cons via current methods
    R["ev.evaluate_objective(x)"]=bench(lambda: ev.evaluate_objective(x))
    R["ev.evaluate_constraints(x)"]=bench(lambda: ev.evaluate_constraints(x))
    R["ev.eval obj+cons (methods)"]=bench(lambda:(ev.evaluate_objective(x),ev.evaluate_constraints(x)))
    print(json.dumps(R,indent=2))

main()
