import json, sys, importlib.util, collections
S, OLD = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], S, OLD]
src = open(f"{S}/s3/s0_ablate16.py").read().split("jobs = [")[0]
exec(src)
first = [json.loads(l) for l in open(f"{S}/s3/s0_ablate16.jsonl")]
st = {(d["arm"], d["name"]): d["status"] for d in first}
jobs = sorted({(a, r) for (a, r), v in st.items() if a != "default" and v != "Optimal" and st[("default", r)] == "Optimal"} | {("default", r) for r in rows})
out = open(f"{S}/s3/s0_confirm.jsonl", "w"); agree = 0
for j in jobs:
    arm, r, s2, nodes, w = run(j)
    same = (s2 == "Optimal") == (st[j] == "Optimal"); agree += same
    out.write(json.dumps({"arm": arm, "name": r, "status": s2, "first": st[j], "wall": round(w, 2)}) + "\n"); out.flush()
    print(f"{r:28} {arm:13} first={st[j]:9} confirm={s2:9} {w:5.1f}s {'' if same else 'DIFFERS'}", flush=True)
print(f"confirmed {agree}/{len(jobs)}"); sys.exit(0 if jobs else 1)
