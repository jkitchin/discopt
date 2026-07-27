"""Final #844 status: do the instances the issue names still return no incumbent?"""
import os, time
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
import warnings; warnings.filterwarnings("ignore")
from discopt.modeling.core import from_nl
BM=os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
CASES=[("tln4",8.3),("tln5",10.3),("tln6",15.3),("gastrans040",0.0),
       ("gastrans582_cool12_95",0.0),("rsyn0805m04hfsg",7174.2),("rsyn0810m04hfsg",6581.9),
       ("portfol_robust050_34",-0.0720755),("watercontamination0202",125.2),
       ("ball_mk2_30",0.0),("lip",5685067.9)]
TL=60.0
for nm,opt in CASES:
    p=f"{BM}/{nm}.nl"
    if not os.path.exists(p):
        print(f"  {nm:24s} MISSING"); continue
    try:
        t=time.perf_counter(); r=from_nl(p).solve(time_limit=TL); w=time.perf_counter()-t
        has="YES" if r.objective is not None else "NO "
        print(f"  {nm:24s} incumbent={has} obj={r.objective} status={r.status:12s} "
              f"wall={w:6.1f}s opt={opt}", flush=True)
    except Exception as e:
        print(f"  {nm:24s} EXC {type(e).__name__}: {str(e)[:60]}", flush=True)
