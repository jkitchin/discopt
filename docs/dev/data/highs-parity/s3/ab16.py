import json, os, subprocess, sys
S, OLD = sys.argv[1], sys.argv[2]; arms = json.loads(sys.argv[3]); out = sys.argv[4]
rows = open(f"{S}/s3/gap16.txt").read().split()
inst = {r: next(f"{OLD}/popc/inst/{f}" for f in os.listdir(f"{OLD}/popc/inst") if f.split(".")[0] == r) for r in rows}
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
opt = {json.loads(l)["name"]: json.loads(l)["opt"] for l in open(f"{S}/s3/s4_entry.jsonl")}
f = open(out, "w"); n = 0
for r in rows:
    line = []
    for arm, extra in arms.items():
        p = subprocess.run([sys.executable, "/home/user/discopt/discopt_benchmarks/scripts/lp_milp_panel.py", "worker", inst[r], "R", "20", "1e-4", "100000000"],
                           capture_output=True, text=True, env=dict(ENV, **extra), timeout=300)
        l = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
        d = json.loads(l[7:]) if l else {"status": "ERR"}
        d = {k: d.get(k) for k in ("status", "objective", "bound", "nodes", "wall", "max_viol")}
        d.update(name=r, arm=arm, opt=opt.get(r)); n += 1
        f.write(json.dumps(d) + "\n"); f.flush()
        line.append(f"{arm}:{(d['status'] or '')[:4]} b={d['bound']} o={d['objective']}")
    print(f"{r:28} " + "  ".join(line), flush=True)
print("solves:", n); sys.exit(0 if n else 1)
