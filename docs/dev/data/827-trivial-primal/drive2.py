"""Completion arms: ON@10 where OFF@10 had an incumbent; OFF@60 and ON@60 on the rescued set."""
import json, os, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.argv = [sys.argv[0], "solve", "4", "10"]
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "drive.py")).read().split("if stage ==")[0])  # imports + run()
by = defaultdict(dict)
for l in open(os.path.join(HERE, "solve_tl10.jsonl")):
    r = json.loads(l); by[r["name"]][r["flag"]] = r
jobs = [(n, "1", "10") for n, d in by.items() if d["0"].get("obj") is not None]
resc = [n for n, d in by.items() if d["0"].get("obj") is None and d.get("1", {}).get("obj") is not None]
jobs += [(n, f, "60") for n in resc for f in ("0", "1")]
out = open(os.path.join(HERE, "solve2.jsonl"), "w")
def one(j):
    n, f, tl = j
    r = run(["solve", f"{B}/minlplib/nl/{n}.nl", f, tl], float(tl) * 6 + 60)
    r.setdefault("name", n); r["flag"] = f; r["tl"] = tl; return r
done = 0
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(one, jobs):
        out.write(json.dumps(r) + "\n"); out.flush(); done += 1
        print(f"{done}/{len(jobs)} {r['name']} {r['flag']}@{r['tl']} {r.get('status', r.get('err'))} {r.get('obj')}", flush=True)
print("DONE", done, flush=True)
