"""cProfile layer-split + callback census on nvs05 (fixed budget).
Usage: g1_profile.py [--baseline] [--tl N]
Aggregates cProfile cumulative time by binding boundary + interop functions,
and per-quantity callback call counts/time."""
import cProfile, pstats, io, json, sys, time
from collections import defaultdict

baseline = "--baseline" in sys.argv
tl = 12.0
if "--tl" in sys.argv: tl = float(sys.argv[sys.argv.index("--tl")+1])

def dis():
    import discopt._jax.nlp_evaluator as nev
    oi=nev.NLPEvaluator.__init__
    def p(self,*a,**k):
        oi(self,*a,**k); self._fused_fc_jit=None; self._gj_fusable_cache=False
    nev.NLPEvaluator.__init__=p
if baseline: dis()

import numpy as np
import discopt._jax.nlp_evaluator as nev
# census counters
cnt=defaultdict(lambda:[0,0])  # name->[calls, ns]
Ev=nev.NLPEvaluator
def wrap(name,fn):
    def inner(self,*a,**k):
        t=time.perf_counter_ns(); r=fn(self,*a,**k); cnt[name][0]+=1; cnt[name][1]+=time.perf_counter_ns()-t; return r
    return inner
for nm in ["evaluate_objective","evaluate_constraints","evaluate_gradient","evaluate_jacobian_values","evaluate_hessian_values"]:
    setattr(Ev,nm,wrap(nm,getattr(Ev,nm)))

from discopt.modeling.core import from_nl
m=from_nl("python/tests/data/minlplib_nl/nvs05.nl")
pr=cProfile.Profile()
t0=time.perf_counter(); pr.enable()
r=m.solve(time_limit=tl, threads=1)
pr.disable(); wall=time.perf_counter()-t0

st=pstats.Stats(pr)
# aggregate cumulative time by layer + interop tokens
layers={"pounce":0.0,"jax_xla":0.0,"rust_lp":0.0,"np_asarray":0.0,"device_get":0.0,"float_conv":0.0,"jax_value":0.0}
for (fn,line,name),v in st.stats.items():
    ct=v[3]
    key=f"{fn}:{name}"
    kl=key.lower()
    if "device_get" in kl: layers["device_get"]+=ct
    if name=="asarray" and "numpy" in kl: layers["np_asarray"]+=ct
    if name in ("__float__",): layers["float_conv"]+=ct
    if name=="_value" or name=="_np" : layers["jax_value"]+=ct
    if "solve_lp_warm_csc" in kl or ("simplex" in kl and "solve" in name): layers["rust_lp"]+=ct
    if "pounce" in kl and name in ("solve","solve_nlp_batch"): layers["pounce"]+=ct
nc_val=float(getattr(r,"node_count",getattr(r,"_node_count",0)))
census={k:{"calls":v[0],"total_us":round(v[1]/1e3,1),"us_per_call":round(v[1]/1e3/v[0],3) if v[0] else 0} for k,v in cnt.items()}
out={"baseline":baseline,"wall":round(wall,2),"nodes":nc_val,"nodes_per_s":round(nc_val/wall,2),
     "bound":float(getattr(r,"bound",getattr(r,"_bound",float('nan')))),
     "layers_cumsec":{k:round(v,3) for k,v in layers.items()},"census":census}
print(json.dumps(out))
