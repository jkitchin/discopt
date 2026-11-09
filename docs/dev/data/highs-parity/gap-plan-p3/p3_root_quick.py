"""Root-only (max_nodes=2) bound per (row, arm): dual gap % vs the manifest optimum, wall; 2 in parallel."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
REPO = "/home/user/discopt"
rows = sys.argv[1].split(","); arms = json.loads(sys.argv[2]); tl = sys.argv[3] if len(sys.argv) > 3 else "20"
pop = {m["name"]: m for m in R.population()}
def run(job):
    name, arm = job; mem = pop[name]
    env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", PYTHONUNBUFFERED="1", **arms[arm])
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                        f"{REPO}/{mem['path']}", "R", tl, "1e-4", "2"], capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    return name, arm, (json.loads(line[-1][7:]) if line else None), mem["oracle_objective"]
res = {}
with ThreadPoolExecutor(2) as ex:
    for name, arm, r, opt in ex.map(run, [(n, a) for n in rows for a in arms]):
        res[(name, arm)] = (r, opt)
n = 0
print(f"{'row':18}" + "".join(f"{a:>18}" for a in arms))
for name in rows:
    cells = []
    for a in arms:
        r, opt = res[(name, a)]
        if r is None or r.get("bound") is None: cells.append(f"{'ERR/none':>18}"); continue
        n += 1; cells.append(f"{abs(opt - r['bound']) / max(1, abs(opt)) * 100:7.2f}% {r['wall']:5.1f}s")
    print(f"{name[:18]:18}" + "".join(f"{c:>18}" for c in cells))
print("bounds compared:", n)
