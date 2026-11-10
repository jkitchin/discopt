"""A/B: HiGHS port (vbounds on, the highs-arm default) vs + node separation (DISCOPT_HIGHS_SEPA_NODE=1), 20 s, MIPLIB-38+M (62 rows).

Two solves in parallel, arm order rotated per row. Per row: certified?, dual gap %, primal gap %,
wall. Soundness (CLAUDE.md §1): a certified optimum must match the oracle (rel 1e-4, abs 1e-6) and
no dual bound may pass the oracle (minimisation). Exits non-zero if no solve ran (§6) or any
violation is found.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; S = sys.argv[1]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
if len(sys.argv) > 2 and sys.argv[2]:
    keep = set(sys.argv[2].split(",")); pop = [m for m in pop if m["name"] in keep]
TL = "20"
B = {"DISCOPT_MILP_RAND_ROUND": "1", "DISCOPT_MILP_RUN1_CONTINUE": "1"}
H = dict(B, DISCOPT_MILP_ROOT_CMIR="highs")
BUNDLE = dict(DISCOPT_HIGHS_ROOT_RENS="1", DISCOPT_HIGHS_DIVE="1", DISCOPT_HIGHS_SEPA_NODE="add",
              DISCOPT_HIGHS_ROOT_POOL="1", DISCOPT_HIGHS_SB_TREE="1")
ARMS = {"base": ("R", dict(H)), "bundle": ("R", dict(H, **BUNDLE))}
names = list(ARMS)
STRIP = ("DISCOPT_HIGHS_DIVE", "DISCOPT_HIGHS_ROOT_POOL", "DISCOPT_HIGHS_SB_TREE", "DISCOPT_HIGHS_ROOT_RENS", "DISCOPT_HIGHS_SEPA_MODK", "DISCOPT_HIGHS_SEPA_NODE", "DISCOPT_HIGHS_SEPA_PROPAGATE", "DISCOPT_HIGHS_SEPA_VBOUNDS", "DISCOPT_MILP_ROOT_CMIR", "DISCOPT_MILP_RAND_ROUND", "DISCOPT_MILP_RUN1_CONTINUE", "DISCOPT_LP_MILP_BACKEND")
def run(job):
    k, arm = job; mem = pop[k]; route, extra = ARMS[arm]
    env = {a: b for a, b in os.environ.items() if a not in STRIP}
    env.update(PYTHONUNBUFFERED="1", **extra)
    env["DISCOPT_LP_MILP_BACKEND"] = "rust" if route == "R" else "highs"
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                        f"{REPO}/{mem['path']}", route, TL, "1e-4", "100000000"], capture_output=True, text=True, env=env, timeout=400)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    r = json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-300:]}
    return k, arm, r
jobs = []
for k in range(len(pop)):
    rot = names[k % 2:] + names[:k % 2]; jobs += [(k, a) for a in rot]
res = {}; viol = []
out = open(f"{S}/ab_node.jsonl" if len(sys.argv) < 4 else sys.argv[3]+"", "w")
with ThreadPoolExecutor(2) as ex:
    for k, arm, r in ex.map(run, jobs):
        res[(k, arm)] = r; opt = pop[k]["oracle_objective"]
        out.write(json.dumps({"name": pop[k]["name"], "arm": arm, "opt": opt, **{x: r.get(x) for x in ("status", "objective", "bound", "gap_certified", "nodes", "wall")}}) + "\n"); out.flush()
        tol = 1e-6 + 1e-4 * abs(opt) if opt is not None else None
        if opt is not None and r.get("status") not in ("ERR", None):
            if r.get("status") == "optimal" and r.get("gap_certified") and (r.get("objective") is None or abs(r["objective"] - opt) > tol):
                viol.append((pop[k]["name"], arm, "certified objective", r.get("objective"), opt))
            if r.get("bound") is not None and r["bound"] > opt + tol:
                viol.append((pop[k]["name"], arm, "bound above oracle", r["bound"], opt))
            if r.get("status") == "infeasible":
                viol.append((pop[k]["name"], arm, "infeasible verdict", None, opt))
        if all((k, a) in res for a in names):
            cells = []
            for a in names:
                x = res[(k, a)]
                if x.get("status") == "ERR": cells.append(f"{'ERR':>24}"); continue
                ok = x.get("status") == "optimal" and x.get("gap_certified")
                b, o = x.get("bound"), x.get("objective")
                dg = "-" if b is None or opt is None else f"{abs(opt-b)/max(1,abs(opt))*100:.1f}"
                pg = "-" if o is None or opt is None else f"{abs(o-opt)/max(1,abs(opt))*100:.1f}"
                cells.append(f"{('OPT' if ok else (x.get('status') or '?')[:4]):>4} d={dg:>5} p={pg:>5} {x.get('wall', 0):4.1f}")
            print(f"{pop[k]['name'][:20]:20} " + " ".join(cells), flush=True)
n = 0
print("\narm      certified  ran  total_wall")
for a in names:
    rs = [res[(k, a)] for k in range(len(pop)) if (k, a) in res]
    ran = [x for x in rs if x.get("status") != "ERR"]; n += len(ran)
    cert = sum(1 for x in ran if x.get("status") == "optimal" and x.get("gap_certified"))
    print(f"{a:8} {cert:9d} {len(ran):4d} {sum(x.get('wall') or 0 for x in ran):10.1f}")
print("violations:", len(viol))
for v in viol: print("  VIOLATION", v)
print("solves:", n)
sys.exit(0 if n and not viol else 1)
