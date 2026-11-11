import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO="/home/user/discopt"; sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
tgt=sys.argv[1].split(","); pop=[m for m in R.population() if m["name"] in tgt]
def run(mem):
    env=dict(os.environ, PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="highs")
    p=subprocess.run([sys.executable,f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py","worker",f"{REPO}/{mem['path']}","H","20","1e-4","100000000"],capture_output=True,text=True,env=env,timeout=400)
    line=[l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r=json.loads(line[-1][7:]) if line else {"status":"ERR","err":p.stderr[-300:]}
    o=mem["oracle_objective"]; b=r.get("bound")
    dg=None if o is None or b is None else round(abs(o-b)/max(1,abs(o))*100,3)
    return mem["name"], r.get("status"), dg, r.get("nodes"), round(r.get("wall") or 0,1)
with ThreadPoolExecutor(2) as ex:
    for res in ex.map(run, pop): print(*res, flush=True)
