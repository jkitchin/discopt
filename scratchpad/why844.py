"""Why is the LP-per-node engine unproductive on a post-solve model, and why does
deepcopy not restore it? Compare three variants of the SAME instance."""
import os, time, copy
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
os.environ["DISCOPT_LP_SPATIAL_FALLBACK"]="0"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from discopt.modeling.core import from_nl
from discopt._jax.lp_spatial_bb import solve_lp_spatial_bb, _is_in_scope
BM=os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
NM="tln6"

def fingerprint(m, tag):
    lb=[];ub=[]
    for v in m._variables:
        lb.append(np.asarray(v.lb).ravel()); ub.append(np.asarray(v.ub).ravel())
    lb=np.concatenate(lb); ub=np.concatenate(ub)
    print(f"  [{tag}] vars={len(m._variables)} cons={len(m._constraints)} "
          f"in_scope={_is_in_scope(m)} lb_sum={lb.sum():.6g} ub_sum={ub.sum():.6g} "
          f"width_sum={(ub-lb).sum():.6g} nl_repr={m._nl_repr is not None} "
          f"branch_bounds={hasattr(m,'_branch_bounds')} "
          f"offcache={m._flat_var_offsets_cache is not None}", flush=True)
    return lb, ub

def run(m, tag):
    t=time.perf_counter()
    r=solve_lp_spatial_bb(m, time_limit=14.0, gap_tolerance=1e-4)
    w=time.perf_counter()-t
    print(f"  [{tag}] ENGINE wall={w:6.1f}s obj={None if r is None else r.objective} "
          f"bound={None if r is None else r.bound} nodes={None if r is None else r.node_count}", flush=True)

# 1) fresh
F=from_nl(f"{BM}/{NM}.nl"); lbF,ubF=fingerprint(F,"fresh")
# 2) deepcopy of a fresh model (taken BEFORE any solve)
S=from_nl(f"{BM}/{NM}.nl")
try:
    C=copy.deepcopy(S); lbC,ubC=fingerprint(C,"deepcopy")
    print(f"  box identical fresh vs deepcopy: {np.array_equal(lbF,lbC) and np.array_equal(ubF,ubC)}")
except Exception as e:
    C=None
    print(f"  [deepcopy] RAISES {type(e).__name__}: {str(e)[:160]}")
# 3) post-solve (solve S, then inspect S)
S.solve(time_limit=26)
lbS,ubS=fingerprint(S,"post-solve")
print(f"  box identical fresh vs post-solve: {np.array_equal(lbF,lbS) and np.array_equal(ubF,ubS)}")
print(f"  post-solve tighter anywhere: lb>fresh {int((lbS>lbF).sum())} ub<fresh {int((ubS<ubF).sum())}")
print("  --- engine on each ---")
run(F,"fresh")
if C is not None: run(C,"deepcopy")
run(S,"post-solve")
