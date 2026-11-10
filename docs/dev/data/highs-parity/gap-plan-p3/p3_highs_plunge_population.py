"""HiGHS default vs HiGHS without plunging (HS_NODE_LIM=1: one node per dive, the next from the
queue), MIPLIB-38+M (62), 20 s, 1 thread, 2 in parallel. Per row: status, nodes. Prints rows run (§6)."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__)); HB = f"{S}/hsab/build/bin/highs"
pop = R.population(); OPT = f"{S}/hsab/opt_diag.txt"
ARMS = {"plunge": {}, "no_plunge": {"HS_NODE_LIM": "1"}}
def run(job):
    k, arm = job; env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}; env.update(ARMS[arm])
    try:
        p = subprocess.run([HB, "--options_file", OPT, f"/home/user/discopt/{pop[k]['path']}"], capture_output=True, text=True, env=env, timeout=90)
        out = p.stdout
    except subprocess.TimeoutExpired:
        return k, arm, "HUNG", None
    st = nodes = None
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("Status"): st = s.split()[-1]
        if s.startswith("Nodes") and s.split()[-1].isdigit(): nodes = int(s.split()[-1])
    return k, arm, st, nodes
jobs = [(k, a) for k in range(len(pop)) for a in ARMS]
n = 0
with open(f"{S}/hs_plunge_pop.jsonl", "w") as f, ThreadPoolExecutor(2) as ex:
    for k, arm, st, nodes in ex.map(run, jobs):
        n += 1; row = {"name": pop[k]["name"], "path": pop[k]["path"], "arm": arm, "status": st, "nodes": nodes}
        f.write(json.dumps(row) + "\n"); f.flush()
        print(f"{n:3d} {pop[k]['name'][:24]:24} {arm:10} {st} {nodes}", flush=True)
print("runs:", n); sys.exit(0 if n else 1)
