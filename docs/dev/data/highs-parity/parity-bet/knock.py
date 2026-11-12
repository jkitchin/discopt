"""HiGHS one-at-a-time knock-outs on the 8 HiGHS-only rows: nodes to close and gap at 60 s."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; SP = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
ROWS = ["mik-250-20-75-1", "mik-250-20-75-2", "mik-250-20-75-3", "mik-250-20-75-5",
        "neos17", "neos-911970", "b-ball", "enlight_hard"]
pop = {m["name"]: m for m in R.population() if m["name"] in ROWS}
ARMS = {"base": ({}, {}), "no_sym": ({"mip_detect_symmetry": "false"}, {}),
        "no_restart": ({"mip_allow_restart": "false"}, {}), "no_presolve": ({"presolve": "off"}, {}),
        "no_conflict": ({}, {"HS_NO_CONFLICT": "1"}), "no_nodesep": ({"mip_allow_cut_separation_at_nodes": "false"}, {}),
        "no_heur": ({"mip_heuristic_effort": "0"}, {}), "pscost_only": ({}, {"HS_PSCOST_ONLY": "1"}),
        "no_nodeprop": ({}, {"HS_NO_NODEPROP": "1"}), "no_redcost": ({}, {"HS_NO_REDCOST": "1"}),
        "no_objprop": ({}, {"HS_NO_OBJPROP": "1"})}
def run(job):
    name, arm = job; mem = pop[name]; o, e = ARMS[arm]
    opt = f"{SP}/bet/k_{arm}_{name}.txt"
    open(opt, "w").write("threads = 1\ntime_limit = 60\n" + "".join(f"{k} = {v}\n" for k, v in o.items()))
    env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}; env.update(e)
    p = subprocess.run([f"{SP}/hsab/build/bin/highs", "--options_file", opt, f"{REPO}/{mem['path']}"],
                       capture_output=True, text=True, env=env, timeout=200)
    r = {"name": name, "arm": arm, "opt": mem["oracle_objective"]}
    for ln in p.stdout.splitlines():
        s = ln.strip()
        if s.startswith("Dual bound"): r["bound"] = float(s.split()[-1])
        elif s.startswith("Nodes") and s.split()[-1].isdigit(): r["nodes"] = int(s.split()[-1])
        elif s.startswith("Timing") and "wall" not in r: r["wall"] = float(s.split()[1])
        elif s.startswith("Status"): r["status"] = s.split()[-1]
    if "Illegal" in p.stdout or "not valid" in p.stdout: r["warn"] = "option rejected"
    return r
jobs = [(n, a) for n in ROWS for a in ARMS]
k = 0
with open(OUT, "w") as out, ThreadPoolExecutor(2) as ex:
    for r in ex.map(run, jobs):
        k += 1; out.write(json.dumps(r) + "\n"); out.flush()
print("jobs", k); sys.exit(0 if k else 1)
