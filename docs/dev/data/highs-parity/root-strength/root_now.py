"""Root-only discopt (current default stack, max_nodes=2) on MIPLIB-38+M; 2 jobs in parallel."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; OUT = sys.argv[1]
extra = dict(a.split("=", 1) for a in sys.argv[2:])
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
STRIP = [k for k in os.environ if k.startswith("DISCOPT_")]
def job(k):
    mem = pop[k]
    env = {a: b for a, b in os.environ.items() if a not in STRIP}
    env.update(PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="1", DISCOPT_LP_MILP_BACKEND="rust", **extra)
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                        f"{REPO}/{mem['path']}", "R", "20", "1e-4", "2"], capture_output=True, text=True, env=env, timeout=400)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r = json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-300:]}
    return {"name": mem["name"], "opt": mem["oracle_objective"], "max": bool(mem.get("maximize")),
            **{x: r.get(x) for x in ("status", "bound", "objective", "nodes", "wall", "err")},
            "root_rows": (r.get("solver_stats") or {}).get("cuts/root_rows")}
n = 0
with open(OUT, "w") as out, ThreadPoolExecutor(2) as ex:
    for row in ex.map(job, range(len(pop))):
        n += 1; out.write(json.dumps(row) + "\n"); out.flush()
        print(n, row["name"], row["status"], row["bound"], row["wall"], flush=True)
print("jobs:", n); sys.exit(0 if n else 1)
