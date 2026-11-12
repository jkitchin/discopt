"""Dual bound at matched node budgets: HiGHS (mip_max_nodes) vs discopt (max_nodes,
deterministic), 8 HiGHS-only rows, budgets 100/1000/5000, 1 thread, 2 jobs."""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; SP = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
ROWS = ["mik-250-20-75-1", "mik-250-20-75-2", "mik-250-20-75-3", "mik-250-20-75-5",
        "neos17", "neos-911970", "b-ball", "enlight_hard"]
pop = {m["name"]: m for m in R.population() if m["name"] in ROWS}
assert len(pop) == len(ROWS), sorted(pop)
BUD = [100, 1000, 5000]
def run(job):
    name, solver, nb = job; mem = pop[name]
    if solver == "H":
        opt = f"{SP}/bet/o_{name}_{nb}.txt"
        open(opt, "w").write(f"threads = 1\ntime_limit = 300\nmip_max_nodes = {nb}\n")
        env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}
        p = subprocess.run([f"{SP}/hsab/build/bin/highs", "--options_file", opt, f"{REPO}/{mem['path']}"],
                           capture_output=True, text=True, env=env, timeout=400)
        r = {}
        for ln in p.stdout.splitlines():
            s = ln.strip()
            if s.startswith("Dual bound"): r["bound"] = float(s.split()[-1])
            elif s.startswith("Primal bound"):
                try: r["objective"] = float(s.split()[-1])
                except ValueError: pass
            elif s.startswith("Nodes") and s.split()[-1].isdigit(): r["nodes"] = int(s.split()[-1])
            elif s.startswith("Timing") and "wall" not in r: r["wall"] = float(s.split()[1])
            elif s.startswith("Status"): r["status"] = s.split()[-1]
    else:
        env = {a: b for a, b in os.environ.items() if not a.startswith("DISCOPT_")}
        env.update(PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PANEL_DETERMINISTIC="1")
        p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                            f"{REPO}/{mem['path']}", "R", "300", "1e-4", str(nb)], capture_output=True, text=True, env=env, timeout=600)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        r = json.loads(line[-1][7:]) if line else {"status": "ERR"}
    return {"name": name, "solver": solver, "budget": nb, "opt": mem["oracle_objective"],
            **{x: r.get(x) for x in ("status", "bound", "objective", "nodes", "wall")}}
jobs = [(n, s, b) for n in ROWS for b in BUD for s in ("H", "D")]
k = 0
with open(OUT, "w") as out, ThreadPoolExecutor(2) as ex:
    for r in ex.map(run, jobs):
        k += 1; out.write(json.dumps(r) + "\n"); out.flush()
        print(k, r, flush=True)
print("jobs", k); sys.exit(0 if k else 1)
