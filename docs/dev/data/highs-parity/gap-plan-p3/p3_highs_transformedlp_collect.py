import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO="/home/user/discopt"; S=sys.argv[1]; outdir=sys.argv[2]; opt=sys.argv[3]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
pop=R.population(); mems=list(pop.values()) if isinstance(pop,dict) else pop
seen=set(); uniq=[]
for m in mems:
    if m['name'] in seen: continue
    seen.add(m['name']); uniq.append(m)
def run(m):
    out=f"{S}/{outdir}/{m['name']}.txt"
    env=dict(os.environ, HS_DUMP_TL=out, HS_DUMP_TL_MAX="12")
    try:
        subprocess.run([f"{S}/hsab/build/bin/highs","--options_file",opt,f"{REPO}/{m['path']}"],capture_output=True,env=env,timeout=90)
    except subprocess.TimeoutExpired: pass
    n=tf=vb=0
    if os.path.exists(out):
        for l in open(out):
            if l.startswith("TL_NEW"): n+=1
            elif l.startswith("TF "): tf+=1
            elif l.startswith("VB "): vb+=1
    print(m['name'], n, tf, vb, flush=True); return n
with ThreadPoolExecutor(4) as ex: tot=sum(ex.map(run, uniq))
print("TOTAL", tot); sys.exit(0 if tot else 1)
