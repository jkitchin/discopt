"""Where is the HiGHS-vs-discopt gap? Per MIPLIB-38+M row (62), 20 s, 1 thread, 2 jobs in parallel:

  H   HiGHS default (patched build, no env switches): status, nodes, final bound/objective, the root
      bound (last log row with 0 B&B nodes), whether it restarted, presolve reductions.
  D   discopt, current highs arm (ROOT_CMIR=highs, vbounds on, propagation/node/modk off): full solve.
  R   discopt, same arm, root only (max_nodes=2): the bound after the root cut loop.

Writes one jsonl row per (instance, job) as it finishes (flush). Exits non-zero if no job ran (§6).
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; S = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
HB = f"{S}/hsab/build/bin/highs"
OPT = f"{S}/hsab/opt_diag.txt"
open(OPT, "w").write("threads = 1\ntime_limit = 20\n")
ENV_D = {"DISCOPT_MILP_RAND_ROUND": "1", "DISCOPT_MILP_RUN1_CONTINUE": "1", "DISCOPT_MILP_ROOT_CMIR": "highs",
         "DISCOPT_LP_MILP_BACKEND": "rust", "PYTHONUNBUFFERED": "1"}
STRIP = [k for k in os.environ if k.startswith("DISCOPT_") or k.startswith("HS_")]
ROW = re.compile(r"^\s*([A-Za-z])?\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\S+)\s+(\S+)")

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def run_h(mem):
    env = {a: b for a, b in os.environ.items() if a not in STRIP}
    t = time.time()
    p = subprocess.run([HB, "--options_file", OPT, f"{REPO}/{mem['path']}"], capture_output=True, text=True, env=env, timeout=200)
    r = {"status": None, "nodes": None, "bound": None, "objective": None, "root_bound": None, "first_root_bound": None,
         "restarts": 0, "presolve": None, "wall": time.time() - t}
    seen_tree = False
    for ln in p.stdout.splitlines():
        s = ln.strip()
        if s.endswith("restarting"):
            r["restarts"] += 1
        if s.startswith("Presolve reductions") and r["presolve"] is None:
            r["presolve"] = s
        m = ROW.match(ln)
        if m and int(m.group(2)) == 0 and not seen_tree:
            d = num(m.group(6))
            if d is not None:
                if r["first_root_bound"] is None and abs(d) != float("inf"):
                    r["first_root_bound"] = d
                r["root_bound"] = d
        elif m:
            seen_tree = True
        if s.startswith("Status"):
            r["status"] = s.split()[-1]
        elif s.startswith("Dual bound"):
            r["bound"] = num(s.split()[-1])
        elif s.startswith("Primal bound"):
            r["objective"] = num(s.split()[-1])
        elif s.startswith("Nodes") and s.split()[-1].isdigit():
            r["nodes"] = int(s.split()[-1])
    return r

def run_d(mem, max_nodes):
    env = {a: b for a, b in os.environ.items() if a not in STRIP}
    env.update(ENV_D)
    p = subprocess.run([sys.executable, f"{REPO}/discopt_benchmarks/scripts/lp_milp_panel.py", "worker",
                        f"{REPO}/{mem['path']}", "R", "20", "1e-4", str(max_nodes)], capture_output=True, text=True, env=env, timeout=400)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    return json.loads(line[-1][7:]) if line else {"status": "ERR", "err": p.stderr[-300:]}

def job(j):
    k, kind = j; mem = pop[k]
    if kind == "H":
        r = run_h(mem)
    elif kind == "D":
        r = run_d(mem, 100000000)
    else:
        r = run_d(mem, 2)
    keep = ("status", "nodes", "bound", "objective", "root_bound", "first_root_bound", "restarts", "presolve", "wall", "gap_certified", "err")
    return {"name": mem["name"], "job": kind, "opt": mem["oracle_objective"], **{x: r.get(x) for x in keep if x in r}}

jobs = [(k, kind) for k in range(len(pop)) for kind in ("H", "D", "R")]
n = 0
with open(OUT, "w") as out, ThreadPoolExecutor(2) as ex:
    for row in ex.map(job, jobs):
        n += 1
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"{n:3d}/{len(jobs)} {row['name'][:22]:22} {row['job']} {row.get('status')} b={row.get('bound')} rb={row.get('root_bound')} nodes={row.get('nodes')} {row.get('wall', 0) or 0:.1f}s", flush=True)
print("jobs:", n)
sys.exit(0 if n else 1)
