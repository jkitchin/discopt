"""Cumulative strip of HiGHS toward discopt's feature set (MIPLIB-38+M, 62 rows, 20 s, 1 thread,
2 parallel). A1 presolve off everywhere (incl. sub-MIPs); A2 +restart/symmetry/conflict off; A3
+mod-k/implied-bound/clique separators and node cuts off; A4 +heuristics discopt lacks off; A5 +RENS/RINS
off. Prints certified per arm and the instances lost relative to the previous arm (§6: exits 1 if
no solve ran)."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; S = sys.argv[1]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
pop = R.population(); H = f"{S}/hsab/build/bin/highs"
NOSEP = {"HS_NO_MODK": "1", "HS_NO_IMPLB": "1", "HS_NO_CLIQUE": "1"}
ARMS = {"A1": ("opt_A1.txt", {}), "A2": ("opt_A2.txt", {"HS_NO_CONFLICT": "1"}),
        "A3": ("opt_A3.txt", dict(NOSEP, HS_NO_CONFLICT="1")), "A4": ("opt_A4.txt", dict(NOSEP, HS_NO_CONFLICT="1")),
        "A5": ("opt_A5.txt", dict(NOSEP, HS_NO_CONFLICT="1"))}
def run(job):
    k, arm = job; opt, extra = ARMS[arm]
    env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}; env.update(extra)
    p = subprocess.run([H, "--options_file", f"{S}/hsab/{opt}", f"{REPO}/{pop[k]['path']}"], capture_output=True, text=True, timeout=120, env=env)
    st = [l.split(None, 1)[1].strip() for l in p.stdout.splitlines() if l.strip().startswith("Status")]
    nodes = [l.split()[-1] for l in p.stdout.splitlines() if l.strip().startswith("Nodes")]
    return k, arm, (st[-1] if st else "ERR"), (int(nodes[-1]) if nodes and nodes[-1].isdigit() else None)
jobs = [(k, a) for k in range(len(pop)) for a in ARMS]
res = {}
out = open(f"{S}/hs_strip.jsonl", "w")
with ThreadPoolExecutor(2) as ex:
    for k, arm, st, nd in ex.map(run, jobs):
        res[(k, arm)] = st
        out.write(json.dumps({"name": pop[k]["name"], "arm": arm, "status": st, "nodes": nd}) + "\n"); out.flush()
n = 0; prev = None
for a in ARMS:
    ok = {pop[k]["name"] for k in range(len(pop)) if res[(k, a)] == "Optimal"}; n += len(ok)
    cnt = sum(1 for k in range(len(pop)) if res[(k, a)] == "Optimal")
    lost = sorted({pop[k]["name"] for k in range(len(pop)) if prev and res[(k, prev)] == "Optimal" and res[(k, a)] != "Optimal"})
    gained = sorted({pop[k]["name"] for k in range(len(pop)) if prev and res[(k, prev)] != "Optimal" and res[(k, a)] == "Optimal"})
    print(f"{a}: certified {cnt:3d}  lost vs prev: {lost}  gained: {gained}"); prev = a
sys.exit(0 if n else 1)
