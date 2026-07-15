import json, sys, time
def dis():
    import discopt._jax.nlp_evaluator as nev
    oi=nev.NLPEvaluator.__init__
    def p(self,*a,**k):
        oi(self,*a,**k); self._fused_fc_jit=None; self._gj_fusable_cache=False
    nev.NLPEvaluator.__init__=p
if "--baseline" in sys.argv: dis()
from discopt.modeling.core import from_nl
m=from_nl("python/tests/data/minlplib_nl/nvs05.nl")
t0=time.perf_counter(); r=m.solve(time_limit=20.0, threads=1); w=time.perf_counter()-t0
nc=float(getattr(r,"node_count",getattr(r,"_node_count",0)))
bd=float(getattr(r,"bound",getattr(r,"_bound",float("nan"))))
print(json.dumps({"baseline":"--baseline" in sys.argv,"nodes":nc,"bound":bd,"wall":w,"nodes_per_s":nc/w}))
