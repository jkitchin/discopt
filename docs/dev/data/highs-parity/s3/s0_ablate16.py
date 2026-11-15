"""S0: HiGHS subsystem knockouts on the 16 rows HiGHS certifies and discopt does not
(population 2026-11-14, from_mps loader). Patched HiGHS build (env switches from
p3_highs_*_switch/ablation patches), 20 s, 1 thread, 2 jobs in parallel. Per row,
which knockout loses HiGHS's certificate. Exits non-zero if nothing ran (§6)."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
S, OLD = sys.argv[1], sys.argv[2]
H = f"{OLD}/hsab/build/bin/highs"
rows = open(f"{S}/s3/gap16.txt").read().split()
inst = {r: next(f"{OLD}/popc/inst/{f}" for f in os.listdir(f"{OLD}/popc/inst") if f.split(".")[0] == r) for r in rows}
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
}
os.makedirs(f"{S}/s3/hsab", exist_ok=True)
for arm, (opts, _) in ARMS.items():
    with open(f"{S}/s3/hsab/opt_{arm}.txt", "w") as f:
        f.write("threads = 1\ntime_limit = 20\n" + "".join(f"{a} = {b}\n" for a, b in opts.items()))
def run(job):
    arm, r = job
    env = dict(os.environ, OMP_NUM_THREADS="1", **ARMS[arm][1])
    t = time.time()
    p = subprocess.run([H, "--options_file", f"{S}/s3/hsab/opt_{arm}.txt", inst[r]], capture_output=True, text=True, env=env, timeout=120)
    st, nodes = "?", None
    for ln in p.stdout.splitlines():
        s = ln.strip()
        if s.startswith("Status"): st = s.split()[-1]
        if s.startswith("Nodes"): nodes = s.split()[-1]
    return arm, r, st, nodes, time.time() - t
jobs = [(a, r) for r in rows for a in ARMS]
out = open(f"{S}/s3/s0_ablate16.jsonl", "w"); res = {}
with ThreadPoolExecutor(2) as ex:
    for arm, r, st, nodes, w in ex.map(run, jobs):
        res[(arm, r)] = st
        out.write(json.dumps({"arm": arm, "name": r, "status": st, "nodes": nodes, "wall": round(w, 2)}) + "\n"); out.flush()
        print(f"{r:28} {arm:13} {st:10} nodes={nodes} {w:5.1f}s", flush=True)
print("\nper-row knockouts that lose the certificate:")
for r in rows:
    if res[("default", r)] != "Optimal":
        print(f"  {r:28} default not certified here ({res[('default', r)]})"); continue
    lost = [a for a in ARMS if a != "default" and res[(a, r)] != "Optimal"]
    print(f"  {r:28} {lost or 'robust'}")
print("solves:", len(res)); sys.exit(0 if res else 1)
