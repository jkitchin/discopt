# ruff: noqa -- frozen #1731 panel record, kept as run.
"""#1731 graduation panel: DISCOPT_SPATIAL_OBJ_LATTICE OFF vs ON over hits.csv.

argv: hits.csv out.jsonl T workers. Arms interleaved per instance (order
alternates). Prints per-run progress and an executed-run count; exits 1 if 0.
"""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

D = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark")
HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.getcwd()
hits, out, T, W = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(open(hits)))
done = set()
if os.path.exists(out):
    for line in open(out):
        d = json.loads(line)
        done.add((d["name"], d["arm"]))


def path_of(r):
    if r["lib"] == "minlplib":
        return f"{D}/minlplib/nl/{r['name']}.nl"
    return f"{D}/qplib/qplib/{r['name']}.qplib"


def run(r, arm):
    env = dict(os.environ, PYTHONPATH=f"{TREE}/python", PANEL_TREE=TREE)
    env["DISCOPT_SPATIAL_OBJ_LATTICE"] = "1" if arm == "on" else "0"
    p = subprocess.run(
        [sys.executable, f"{HERE}/one.py", r["lib"], path_of(r), str(T)],
        env=env,
        capture_output=True,
        text=True,
        timeout=T + 300,
    )
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if res is None:
        res = dict(crash=True, rc=p.returncode, stderr=p.stderr[-600:])
    elif res["flag"] != (arm == "on"):
        raise RuntimeError(f"{r['name']} {arm}: flag not applied {res}")
    return res


def job(i_r):
    i, r = i_r
    order = ("off", "on") if i % 2 == 0 else ("on", "off")
    recs = []
    for arm in order:
        if (r["name"], arm) in done:
            continue
        res = run(r, arm)
        rec = dict(name=r["name"], lib=r["lib"], ref=float(r["ref"]), arm=arm, **res)
        recs.append(rec)
        print(
            f"{r['name']:<14} {arm:<3} {res.get('status')} obj={res.get('objective')} "
            f"bnd={res.get('bound')} nodes={res.get('nodes')} "
            f"wall={res.get('wall', 0):.1f} feas={res.get('feas')} "
            f"{'CRASH ' + str(res.get('rc')) if res.get('crash') else ''}",
            flush=True,
        )
    return recs


n = 0
with ThreadPoolExecutor(W) as ex, open(out, "a") as f:
    for fut in as_completed([ex.submit(job, x) for x in enumerate(rows)]):
        recs = fut.result()
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
print(f"executed runs {n}", flush=True)
sys.exit(0 if n or done else 1)
