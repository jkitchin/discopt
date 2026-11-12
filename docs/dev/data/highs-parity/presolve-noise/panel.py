"""Bound-neutral check: new (perturbed infeasible verdict kept) vs old, old twice (noise).
62 rows, deterministic, max_nodes 3000, 60 s backstop, 1 thread, 2 jobs, arm order rotated."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; OUT = sys.argv[1]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
ARMS = {"new": {}, "old": {"DISCOPT_TMP_OLD_NOISE": "1"}, "old2": {"DISCOPT_TMP_OLD_NOISE": "1"}}
names = list(ARMS)
def run(job):
    k, arm = job; mem = pop[k]
    env = {a: b for a, b in os.environ.items() if not a.startswith("DISCOPT_")}
    env.update(PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PANEL_DETERMINISTIC="1", DISCOPT_PROFILE="1", **ARMS[arm])
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                        f"{REPO}/{mem['path']}", "R", "60", "1e-4", "3000"], capture_output=True, text=True, env=env, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r = json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-300:]}
    fired = sum(int(l.split()[-1]) for l in p.stderr.splitlines() if l.strip().startswith("MilpPresolveVerifyFallback"))
    return {"k": k, "name": mem["name"], "arm": arm, "opt": mem["oracle_objective"], "max": bool(mem.get("maximize")), "fired": fired,
            **{x: r.get(x) for x in ("status", "objective", "bound", "gap_certified", "nodes", "wall", "err")}}
jobs = []
for k in range(len(pop)):
    rot = names[k % 3:] + names[:k % 3]; jobs += [(k, a) for a in rot]
n = 0
with open(OUT, "w") as out, ThreadPoolExecutor(2) as ex:
    for r in ex.map(run, jobs):
        n += 1; out.write(json.dumps(r) + "\n"); out.flush()
        print(n, r["name"], r["arm"], r["status"], r["nodes"], r["fired"], flush=True)
print("jobs", n); sys.exit(0 if n else 1)
