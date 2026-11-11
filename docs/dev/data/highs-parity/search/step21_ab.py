import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO="/home/user/discopt"; sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
tgt=sys.argv[1].split(","); pop=[m for m in R.population() if m["name"] in tgt]
def run(job):
    mem, arm = job
    env=dict(os.environ, PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1",
             DISCOPT_HIGHS_SEARCH_CUT_LP=("1" if arm=="on" else "0"))
    p=subprocess.run([sys.executable,f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py","worker",f"{REPO}/{mem['path']}","R","20","1e-4","100000000"],capture_output=True,text=True,env=env,timeout=400)
    line=[l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r=json.loads(line[-1][7:]) if line else {"status":"ERR"}
    add=[l.split()[-1] for l in p.stderr.splitlines() if "HighsSearchCutRowsAdded" in l]
    dele=[l.split()[-1] for l in p.stderr.splitlines() if "HighsSearchCutRowsDeleted" in l]
    o=mem["oracle_objective"]; b=r.get("bound"); sg=-1 if mem.get("maximize") else 1
    dg=None if o is None or b is None else round(abs(o-b)/max(1,abs(o))*100,3)
    bad = b is not None and o is not None and sg*(b-o) > 1e-6+1e-4*abs(o)
    return mem["name"], arm, r.get("status"), dg, r.get("nodes"), round(r.get("wall") or 0,1), max([int(a) for a in add] or [0]), max([int(a) for a in dele] or [0]), "BOUND>OPT" if bad else ""
jobs=[]
for k,m in enumerate(pop): jobs += [(m,"off"),(m,"on")] if k%2==0 else [(m,"on"),(m,"off")]
with ThreadPoolExecutor(2) as ex:
    for res in ex.map(run, jobs): print(*res, flush=True)
