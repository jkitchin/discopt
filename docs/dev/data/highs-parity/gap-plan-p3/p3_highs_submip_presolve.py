"""HiGHS ablation: sub-MIP presolve. Arms: default; presolve off at the top only (sub-MIPs still
presolve, HighsPrimalHeuristics.cpp:124-130); presolve off everywhere (mip_root_presolve_only).
62 rows, 20 s, 1 thread, 2 parallel. Prints certified per arm and the instances each arm loses."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; S = sys.argv[1]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R
pop = R.population(); H = f"{S}/hsab/build/bin/highs"
ARMS = {"default": "opt_default2.txt", "pre_top_off": "opt_pre_top_off.txt", "pre_all_off": "opt_pre_all_off.txt"}
def run(job):
    k, arm = job
    p = subprocess.run([H, "--options_file", f"{S}/hsab/{ARMS[arm]}", f"{REPO}/{pop[k]['path']}"], capture_output=True, text=True, timeout=120)
    st = [l.split(None, 1)[1].strip() for l in p.stdout.splitlines() if l.strip().startswith("Status")]
    t = [l.split()[-1] for l in p.stdout.splitlines() if l.strip().startswith("Timing")]
    return k, arm, (st[-1] if st else "ERR"), (float(t[-1]) if t else None)
jobs = [(k, a) for k in range(len(pop)) for a in ARMS]
res = {}
out = open(f"{S}/hs_presolve_ab.jsonl", "w")
with ThreadPoolExecutor(2) as ex:
    for k, arm, st, t in ex.map(run, jobs):
        res[(k, arm)] = (st, t)
        out.write(json.dumps({"name": pop[k]["name"], "arm": arm, "status": st, "wall": t}) + "\n"); out.flush()
n = 0
for a in ARMS:
    ok = [k for k in range(len(pop)) if res[(k, a)][0] == "Optimal"]; n += len(ok)
    lost = sorted({pop[k]["name"] for k in range(len(pop)) if res[(k, "default")][0] == "Optimal" and res[(k, a)][0] != "Optimal"})
    print(f"{a:12} certified {len(ok):3d}  lost vs default: {lost}")
sys.exit(0 if n else 1)
