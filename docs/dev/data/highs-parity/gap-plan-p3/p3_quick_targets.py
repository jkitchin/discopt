"""Quick full-solve check on target rows: one 20 s solve per (row, arm), 2 in parallel."""
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
                        f"{REPO}/{mem['path']}", "R", tl, "1e-4", "100000000"], capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line: return name, arm, None, mem["oracle_objective"]
    return name, arm, json.loads(line[-1][7:]), mem["oracle_objective"]
jobs = [(n, a) for n in rows for a in arms]
res = {}
with ThreadPoolExecutor(2) as ex:
    for name, arm, r, opt in ex.map(run, jobs):
        res[(name, arm)] = (r, opt)
cert = {a: 0 for a in arms}; n = 0
print(f"{'row':18}" + "".join(f"{a:>26}" for a in arms))
for name in rows:
    cells = []
    for a in arms:
        r, opt = res[(name, a)]; n += 1
        if r is None: cells.append(f"{'ERR':>26}"); continue
        b, o = r.get("bound"), r.get("objective")
        dg = "-" if b is None else f"{abs(opt-b)/max(1,abs(opt))*100:.2f}"
        pg = "-" if o is None else f"{abs(o-opt)/max(1,abs(opt))*100:.2f}"
        ok = r["status"] == "optimal"; cert[a] += ok
        cells.append(f"{('OPT' if ok else r['status'][:4]):>5} d={dg:>6} p={pg:>6} {r['wall']:4.1f}s")
    print(f"{name[:18]:18}" + "".join(f"{c:>26}" for c in cells))
print("certified:", cert, "solves:", n)
