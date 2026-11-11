import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO="/home/user/discopt"; sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
tgt=sys.argv[1].split(","); pop=[m for m in R.population() if m["name"] in tgt]
ARMS={"fixed":{"DISCOPT_HIGHS_SEARCH_CUT_LP":"0","DISCOPT_HIGHS_SEPA_NODE":"0"},
      "gmi":{"DISCOPT_HIGHS_SEARCH_CUT_LP":"1","DISCOPT_HIGHS_SEPA_NODE":"0"},
      "highs":{"DISCOPT_HIGHS_SEARCH_CUT_LP":"1","DISCOPT_HIGHS_SEPA_NODE":"1"},
      "add":{"DISCOPT_HIGHS_SEARCH_CUT_LP":"1","DISCOPT_HIGHS_SEPA_NODE":"add"}}
if os.environ.get("ARMS"): ARMS={a:ARMS[a] for a in os.environ["ARMS"].split(",")}
names=list(ARMS)
def mx(err, key):
    v=[int(l.split()[-1]) for l in err.splitlines() if l.strip().startswith(key+" ")]
    return max(v or [0])
def run(job):
    mem, arm = job
    env=dict(os.environ, PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1", **ARMS[arm])
    p=subprocess.run([sys.executable,f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py","worker",f"{REPO}/{mem['path']}","R","20","1e-4","100000000"],capture_output=True,text=True,env=env,timeout=400)
    line=[l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r=json.loads(line[-1][7:]) if line else {"status":"ERR"}
    o=mem["oracle_objective"]; b=r.get("bound"); sg=-1 if mem.get("maximize") else 1
    dg=None if o is None or b is None else round(abs(o-b)/max(1,abs(o))*100,3)
    bad = b is not None and o is not None and sg*(b-o) > 1e-6+1e-4*abs(o)
    return (mem["name"], arm, r.get("status"), dg, r.get("nodes"), round(r.get("wall") or 0,1),
            "rows+", mx(p.stderr,"HighsSearchCutRowsAdded"), "rows-", mx(p.stderr,"HighsSearchCutRowsDeleted"),
            "sep_raw", mx(p.stderr,"NodeGmiRaw"), "BOUND>OPT" if bad else "")
jobs=[]
for k,m in enumerate(pop):
    rot=names[k%len(names):]+names[:k%len(names)]; jobs+=[(m,a) for a in rot]
with ThreadPoolExecutor(2) as ex:
    for res in ex.map(run, jobs): print(*res, flush=True)
