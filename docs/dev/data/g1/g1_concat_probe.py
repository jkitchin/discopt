import time, json, sys
import numpy as np
NL="python/tests/data/minlplib_nl/nvs05.nl"
def main():
    import jax, jax.numpy as jnp
    import discopt._jax.nlp_evaluator as nev
    from discopt.modeling.core import from_nl
    m=from_nl(NL); ev=nev.cached_evaluator(m); n=ev.n_variables; mc=ev.n_constraints
    ev._ensure_coo_cache(); jr,jc=ev._jac_rows,ev._jac_cols
    obj=ev._obj_fn_jit; cons=ev._cons_fn_jit; grad=ev._grad_fn_jit; jac=ev._jac_fn_jit
    # fused concat groups
    def fc(x,p):
        return jnp.concatenate([jnp.reshape(obj(x,p),(1,)), jnp.reshape(cons(x,p),(-1,))])
    def gj(x,p):
        return jnp.concatenate([jnp.reshape(grad(x,p),(-1,)), jnp.reshape(jac(x,p),(-1,))])
    fcj=jax.jit(fc); gjj=jax.jit(gj)
    rng=np.random.default_rng(11); lb,ub=ev.variable_bounds
    lb=np.where(np.isfinite(lb),lb,-5.); ub=np.where(np.isfinite(ub),ub,5.)
    N=6000
    seq=[]
    for _ in range(N):
        x=(lb+rng.uniform(size=n)*(ub-lb)).astype(np.float64)
        seq.append((x, rng.random()<0.12))
    p=ev._current_params()
    for x,_ in seq[:1]:
        ev.evaluate_objective(x); ev.evaluate_constraints(x); ev.evaluate_gradient(x); ev.evaluate_jacobian_values(x)
        jax.block_until_ready(fcj(x,p)); jax.block_until_ready(gjj(x,p))
    # baseline current methods
    t0=time.perf_counter_ns(); base=[]
    for x,full in seq:
        f=ev.evaluate_objective(x); c=ev.evaluate_constraints(x)
        if full: g=ev.evaluate_gradient(x); jv=ev.evaluate_jacobian_values(x)
        else: g=jv=None
        base.append((f,c,g,jv))
    tb=(time.perf_counter_ns()-t0)/1e3
    # concat-fused memoized
    t0=time.perf_counter_ns(); fus=[]
    for x,full in seq:
        pp=ev._current_params()
        arr=np.asarray(fcj(x,pp)); f=arr[0]; c=arr[1:1+mc]
        if full:
            arr2=np.asarray(gjj(x,pp)); g=arr2[:n]; J=arr2[n:].reshape(mc,n); jv=J[jr,jc].astype(np.float64)
        else: g=jv=None
        fus.append((float(f),c,g,jv))
    tf=(time.perf_counter_ns()-t0)/1e3
    md={"f":0.,"c":0.,"g":0.,"J":0.}
    for (f0,c0,g0,j0),(f1,c1,g1,j1) in zip(base,fus):
        md["f"]=max(md["f"],abs(f0-f1)); md["c"]=max(md["c"],float(np.max(np.abs(c0-c1))) if c0.size else 0.)
        if g0 is not None:
            md["g"]=max(md["g"],float(np.max(np.abs(g0-g1)))); md["J"]=max(md["J"],float(np.max(np.abs(j0-j1))))
    out={"N":N,"baseline_us":tb,"fused_us":tf,"speedup_x":tb/tf,"us_per_iter_base":tb/N,"us_per_iter_fused":tf/N,"max_abs_diff":md}
    print(json.dumps(out))
main()
