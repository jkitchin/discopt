"""Ablate HiGHS components over the MIPLIB-38+M population (20 s, 1 thread, 2 parallel).

Each arm turns one HiGHS stage off (options file and/or env switches of the ablation
build). Status per instance from the solver's own report; certified = 'Optimal'.
Prints per-arm certified counts, instances each arm loses vs default, and total wall.
Exits non-zero when no solve ran (CLAUDE.md §6).
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; S = sys.argv[1]
H = f"{S}/hsab/build/bin/highs"
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
NOSEP = {"HS_NO_TABLEAU": "1", "HS_NO_PATH": "1", "HS_NO_MODK": "1", "HS_NO_IMPLB": "1", "HS_NO_CLIQUE": "1"}
ARMS = {
    "default": ({}, {}),
    "presolve_off": ({"presolve": "off"}, {}),
    "restart_off": ({"mip_allow_restart": "false"}, {}),
    "symmetry_off": ({"mip_detect_symmetry": "false"}, {}),
    "heur_off": ({"mip_heuristic_effort": "0", "mip_heuristic_run_feasibility_jump": "false"}, {}),
    "sb_off": ({"mip_pscost_minreliable": "0"}, {}),
    "cuts_off": ({"mip_allow_cut_separation_at_nodes": "false"}, NOSEP),
    "conflict_off": ({}, {"HS_NO_CONFLICT": "1"}),
    "discopt_like": ({"presolve": "off", "mip_allow_restart": "false", "mip_detect_symmetry": "false"}, {"HS_NO_CONFLICT": "1"}),
}
def run(job):
    arm, k = job; opts, env_extra = ARMS[arm]; mem = pop[k]
    of = f"{S}/hsab/opt_{arm}.txt"
    env = dict(os.environ, **env_extra)
    t = time.time()
    p = subprocess.run([H, "--options_file", of, f"{REPO}/{mem['path']}"], capture_output=True, text=True, env=env, timeout=120)
    st = "?"
    for ln in p.stdout.splitlines():
        if ln.strip().startswith("Status"):
            st = ln.split()[-1]
    return arm, k, st, time.time() - t
for arm, (opts, _) in ARMS.items():
    with open(f"{S}/hsab/opt_{arm}.txt", "w") as f:
        f.write("threads = 1\ntime_limit = 20\n" + "".join(f"{a} = {b}\n" for a, b in opts.items()))
jobs = [(a, k) for k in range(len(pop)) for a in ARMS]
res = {}
out = open(f"{S}/hs_ablate_pop.jsonl", "w")
with ThreadPoolExecutor(2) as ex:
    for arm, k, st, w in ex.map(run, jobs):
        res[(arm, k)] = (st, w)
        out.write(json.dumps({"arm": arm, "name": pop[k]["name"], "status": st, "wall": w}) + "\n"); out.flush()
n = len(res)
base = {k for k in range(len(pop)) if res[("default", k)][0] == "Optimal"}
print(f"{'arm':14} certified  lost_vs_default  gained  total_wall")
for arm in ARMS:
    cert = {k for k in range(len(pop)) if res[(arm, k)][0] == "Optimal"}
    lost = sorted(pop[k]["name"] for k in base - cert); gained = sorted(pop[k]["name"] for k in cert - base)
    tw = sum(min(res[(arm, k)][1], 20.0) for k in range(len(pop)))
    print(f"{arm:14} {len(cert):9}  {len(lost):15}  {len(gained):6}  {tw:9.0f}   lost={lost} gained={gained}")
print("solves:", n)
sys.exit(0 if n else 1)
