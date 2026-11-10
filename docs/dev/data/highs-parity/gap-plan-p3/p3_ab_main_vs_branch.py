"""A/B: origin/main vs this branch (flag off / flag bundle on) vs HiGHS, 20 s, MIPLIB-38+M.

Each arm runs `lp_milp_panel.py worker` under its own interpreter (main: a venv holding
main's wheel). Two solves in parallel; arm order rotates per row. Prints per-row cells and a
summary (certified, dual/primal gap, wall); exits non-zero if no solve ran (CLAUDE.md §6).
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"
S = sys.argv[1]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
if len(sys.argv) > 2 and sys.argv[2]:
    keep = set(sys.argv[2].split(",")); pop = [m for m in pop if m["name"] in keep]
TL = "20"
BUNDLE = {"DISCOPT_MILP_ROOT_CMIR": "1", "DISCOPT_MILP_RAND_ROUND": "1", "DISCOPT_MILP_RUN1_CONTINUE": "1"}
ARMS = {
    "main": (sys.executable, f"{S}/wt_main/discopt_benchmarks/scripts/lp_milp_panel.py", "R", {"PYTHONPATH": f"{S}/wt_main/python"}),
    "br_off": (sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "R", {}),
    "br_flag": (sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "R", BUNDLE),
    "highs": (sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "H", {}),
}
names = list(ARMS)
STRIP = ("PYTHONPATH", "DISCOPT_MILP_ROOT_CMIR", "DISCOPT_MILP_RAND_ROUND", "DISCOPT_MILP_RUN1_CONTINUE", "DISCOPT_LP_MILP_BACKEND")
def run(job):
    k, arm = job; mem = pop[k]; py, script, route, extra = ARMS[arm]
    env = {a: b for a, b in os.environ.items() if a not in STRIP}
    env.update(PYTHONUNBUFFERED="1", **extra)
    env["DISCOPT_LP_MILP_BACKEND"] = "rust" if route == "R" else "highs"
    t = time.time()
    p = subprocess.run([py, script, "worker", f"{REPO}/{mem['path']}", route, TL, "1e-4", "100000000"],
                       capture_output=True, text=True, env=env, timeout=400)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r = json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-300:]}
    r["_wall_ext"] = time.time() - t
    return k, arm, r
jobs = []
for k in range(len(pop)):
    rot = names[k % 4:] + names[:k % 4]
    jobs += [(k, a) for a in rot]
res = {}
out = open(f"{S}/ab_main.jsonl", "w")
with ThreadPoolExecutor(2) as ex:
    for k, arm, r in ex.map(run, jobs):
        res[(k, arm)] = r
        out.write(json.dumps({"name": pop[k]["name"], "arm": arm, "opt": pop[k]["oracle_objective"], **{x: r.get(x) for x in ("status", "objective", "bound", "gap_certified", "nodes", "wall")}}) + "\n"); out.flush()
        if all((k, a) in res for a in names):
            cells = []
            opt = pop[k]["oracle_objective"]
            for a in names:
                x = res[(k, a)]
                if x.get("status") == "ERR": cells.append(f"{'ERR':>24}"); continue
                ok = x.get("status") == "optimal" and x.get("gap_certified")
                b, o = x.get("bound"), x.get("objective")
                dg = "-" if b is None or opt is None else f"{abs(opt-b)/max(1,abs(opt))*100:.1f}"
                pg = "-" if o is None or opt is None else f"{abs(o-opt)/max(1,abs(opt))*100:.1f}"
                cells.append(f"{('OPT' if ok else (x.get('status') or '?')[:4]):>4} d={dg:>5} p={pg:>5} {x.get('wall', 0):4.1f}")
            print(f"{pop[k]['name'][:20]:20} " + " ".join(cells), flush=True)
n = 0; summ = {}
for a in names:
    rs = [res[(k, a)] for k in range(len(pop))]
    cert = [x for x in rs if x.get("status") == "optimal" and x.get("gap_certified")]
    n += len(rs)
    walls = sorted(x.get("wall", 20.0) or 20.0 for x in cert)
    summ[a] = {"certified": len(cert), "errors": sum(x.get("status") == "ERR" for x in rs),
               "median_wall_solved": walls[len(walls) // 2] if walls else None,
               "total_wall": round(sum(min(x.get("wall", 20.0) or 20.0, 20.0) for x in rs), 1)}
print("arms: " + " | ".join(names))
print(json.dumps(summ, indent=1)); print("solves:", n)
sys.exit(0 if n else 1)
