"""Interleaved OFF/ON x3 at 10 s on the cert-lost / incumbent-lost suspects."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.argv = [sys.argv[0], "solve", "2", "10"]
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "drive.py")).read().split("if stage ==")[0])
names = "ex4_1_5 nvs17 nvs18 nvs19 nvs23 nvs24 pointpack06 rbrock sfacloc2_2_95 st_rv9 sonet20v6".split()
jobs = [(n, rep, f) for n in names for rep in range(3) for f in (("0", "1") if rep % 2 == 0 else ("1", "0"))]
out = open(os.path.join(HERE, "solve3.jsonl"), "w")
def one(j):
    n, rep, f = j
    r = run(["solve", f"{B}/minlplib/nl/{n}.nl", f, "10"], 120); r.setdefault("name", n); r["flag"] = f; r["rep"] = rep; return r
with ThreadPoolExecutor(2) as ex:
    for i, r in enumerate(ex.map(one, jobs)):
        out.write(json.dumps(r) + "\n"); out.flush()
        print(i + 1, len(jobs), r["name"], r["flag"], r.get("status", r.get("err")), r.get("cert"), round(r.get("t", 0), 1), flush=True)
print("DONE", flush=True)
