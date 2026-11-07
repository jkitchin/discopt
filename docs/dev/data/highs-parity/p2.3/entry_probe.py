import sys, os, re, json, subprocess
names = sys.argv[1].split(",")
man = {x["name"]: x for x in json.load(open("discopt_benchmarks/data/lp_milp/manifest.json"))["members"]}
CHILD = open("/tmp/claude-0/-home-user-discopt/38303ad9-c9f4-5a5b-9f75-f33b92707a61/scratchpad/p22_one.py").read()
pat = re.compile(r"^\s+(P23\w+|Phase1Pivots|Phase2Pivots|DualWarmSolves)\s+(\d+)$")
rows = []
for n in names:
    env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", DISCOPT_PROFILE="1")
    p = subprocess.run([sys.executable, "-c", CHILD, man[n]["path"]], env=env, capture_output=True, text=True, timeout=200)
    c = {}
    for ln in p.stderr.splitlines():
        m = pat.match(ln)
        if m: c[m.group(1)] = c.get(m.group(1), 0) + int(m.group(2))
    fs, bs = c.get("P23FtranSolves", 0), c.get("P23BtranSolves", 0)
    dim = c.get("P23DimSum", 0) / max(1, fs)
    r = {"name": n, "m": man[n]["m"], "ftran": fs, "btran": bs,
         "ftran_low": c.get("P23FtranOutLow", 0) / max(1, fs), "btran_low": c.get("P23BtranOutLow", 0) / max(1, bs),
         "ftran_out_fill": c.get("P23FtranOutNnz", 0) / max(1, c.get("P23DimSum", 1)),
         "btran_out_fill": c.get("P23BtranOutNnz", 0) / max(1, bs * dim),
         "ftran_in_fill": c.get("P23FtranInNnz", 0) / max(1, c.get("P23DimSum", 1))}
    rows.append(r)
    print(f"{n:12s} m={r['m']:5d} ftran {fs:7d} low {r['ftran_low']:.2f} fill in {r['ftran_in_fill']:.3f} out {r['ftran_out_fill']:.3f} | btran {bs:7d} low {r['btran_low']:.2f} out {r['btran_out_fill']:.3f}", flush=True)
tf = sum(r["ftran"] for r in rows); tb = sum(r["btran"] for r in rows)
print("pooled ftran low %.3f  btran low %.3f  (solves %d / %d)" % (sum(r["ftran_low"] * r["ftran"] for r in rows) / max(1, tf), sum(r["btran_low"] * r["btran"] for r in rows) / max(1, tb), tf, tb))
json.dump(rows, open(sys.argv[2], "w"), indent=1)
