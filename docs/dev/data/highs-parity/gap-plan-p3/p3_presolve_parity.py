"""Presolve parity: per row, HiGHS's presolved size (diag.jsonl, HiGHS default) vs discopt's
(DISCOPT_MILP_PRESOLVE_LOG, engine run with a 1 s limit). Prints the table and totals; counts
rows compared (§6)."""
import json, os, re, subprocess, sys
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__))
pop = R.population()
hs = {}
for l in open(f"{S}/diag.jsonl"):
    r = json.loads(l)
    if r["job"] == "H": hs[r["name"]] = r.get("presolve")
ENV = dict(DISCOPT_MILP_PRESOLVE="1", DISCOPT_MILP_PRESOLVE_LOG="1", RAYON_NUM_THREADS="1", DISCOPT_MILP_RAND_ROUND="1",
           DISCOPT_MILP_RUN1_CONTINUE="1", DISCOPT_MILP_ROOT_CMIR="highs", DISCOPT_LP_MILP_BACKEND="rust", TL="1")
num = lambda s: [int(x) for x in re.findall(r"(\d+)\(-?\d+\)", s)] if s else None
out = []; n = 0
for m in pop:
    env = {**os.environ, **ENV}
    p = subprocess.run([sys.executable, f"{S}/prof_tl.py", f"/home/user/discopt/{m['path']}"], capture_output=True, text=True, env=env, timeout=300)
    line = next((l for l in p.stderr.splitlines() if l.startswith("MILP presolve:")), None)
    mm = re.search(r"(\d+)x(\d+) -> (\d+)x(\d+)", line) if line else None
    h = num(hs.get(m["name"]))
    n += 1
    row = dict(name=m["name"], ours=[int(x) for x in mm.groups()] if mm else None, highs=h, stats=line.split("PresolveStats")[1] if line and "PresolveStats" in line else None)
    out.append(row)
    print(json.dumps(row), flush=True)
json.dump(out, open(f"{S}/presolve_parity.json", "w"))
print("rows:", n)
