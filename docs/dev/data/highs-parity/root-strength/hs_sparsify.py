"""HiGHS default vs presolve_rule_off=16384 (sparsify off), 62 rows, 20 s, 1 thread, 2 jobs."""
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/user/discopt"; SP = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, f"{REPO}/discopt_benchmarks/scripts")
import p42_rens_panel as R  # noqa: E402
pop = R.population()
HB = f"{SP}/hsab/build/bin/highs"
ROW = re.compile(r"^\s*([A-Za-z])?\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\S+)\s+(\S+)")
def run(job):
    k, arm = job; mem = pop[k]
    opt = f"{SP}/root/hs_{arm}_{k}.txt"
    open(opt, "w").write("threads = 1\ntime_limit = 20\n" + ("presolve_rule_off = 16384\n" if arm == "nosparsify" else ""))
    env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}
    p = subprocess.run([HB, "--options_file", opt, f"{REPO}/{mem['path']}"], capture_output=True, text=True, env=env, timeout=200)
    r = {"name": mem["name"], "arm": arm, "status": None, "nodes": None, "time": None, "root_bound": None, "presolve": None}
    tree = False
    for ln in p.stdout.splitlines():
        s = ln.strip(); m = ROW.match(ln)
        if m and int(m.group(2)) == 0 and not tree:
            try: r["root_bound"] = float(m.group(6))
            except ValueError: pass
        elif m: tree = True
        if s.startswith("Status"): r["status"] = s.split()[-1]
        elif s.startswith("Nodes") and s.split()[-1].isdigit(): r["nodes"] = int(s.split()[-1])
        elif s.startswith("Timing") and r["time"] is None: r["time"] = float(s.split()[1])
        elif s.startswith("Presolve reductions") and r["presolve"] is None: r["presolve"] = s
    r["opt"] = mem["oracle_objective"]
    return r
jobs = [(k, a) for k in range(len(pop)) for a in (("default", "nosparsify") if k % 2 else ("nosparsify", "default"))]
n = 0
with open(OUT, "w") as out, ThreadPoolExecutor(2) as ex:
    for r in ex.map(run, jobs):
        n += 1; out.write(json.dumps(r) + "\n"); out.flush()
print("jobs", n); sys.exit(0 if n else 1)
