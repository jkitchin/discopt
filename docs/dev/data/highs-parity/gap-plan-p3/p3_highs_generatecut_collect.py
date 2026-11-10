import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO="/home/user/discopt"; S=sys.argv[1]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
pop=R.population()
def run(m):
    out=f"{S}/gcdump/{m['name']}.txt"
    env=dict(os.environ, HS_DUMP_GC=out, HS_DUMP_GC_MAX="3000")
    try:
        subprocess.run([f"{S}/hsab/build/bin/highs","--options_file",f"{S}/hsab/opt_root.txt",f"{REPO}/{m['path']}"],capture_output=True,env=env,timeout=60)
    except subprocess.TimeoutExpired: pass
    n=sum(1 for l in open(out) if l.startswith("GC_IN")) if os.path.exists(out) else 0
    print(m['name'], n, flush=True); return n
with ThreadPoolExecutor(4) as ex: tot=sum(ex.map(run, list(pop.values()) if isinstance(pop,dict) else pop))
print("TOTAL", tot); sys.exit(0 if tot else 1)
