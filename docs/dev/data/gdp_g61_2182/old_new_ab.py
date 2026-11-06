"""#2182: is discopt main slower than the #2149 build (1f16c936) on the same machine?

For every (family arm, instance) that main solved in this run's G6.1 rep 1, run
``gdp_baseline.py --worker`` from each tree (OLD = 1f16c936, NEW = main 63910b2),
interleaved with alternating order, one run per lane at a time, each lane pinned
to its own core (``taskset -c <core>``). Appends one JSON row per run (flushed).

    python -u old_new_ab.py <rep1.jsonl> <out.jsonl> <old_tree> <new_tree> <py>
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

rep1, out, old_t, new_t, py = sys.argv[1:6]
TL, GRACE = 300.0, 120.0
ARMS = [
    "discopt-big-m",
    "discopt-hull",
    "discopt-loa",
    "discopt-gloa",
    "discopt-ric",
    "discopt-enumerate",
    "discopt-lbb",
]
cells = []
for ln in open(rep1):
    r = json.loads(ln)
    if r["arm"] in ARMS and r["verdict"] == "solved":
        cells.append((r["instance"], r["arm"]))
cells.sort()
done = set()
if Path(out).exists():
    for ln in open(out):
        r = json.loads(ln)
        done.add((r["instance"], r["arm"], r["tree"]))
tasks = []
for k, (inst, arm) in enumerate(cells):
    order = ("old", "new") if k % 2 == 0 else ("new", "old")
    tasks.append([(inst, arm, t) for t in order])
print(f"{len(cells)} cells, {2 * len(cells)} runs, {len(done)} already done", flush=True)
lock = threading.Lock()
queue = list(tasks)


def run(inst, arm, tree, core):
    root = old_t if tree == "old" else new_t
    env = dict(os.environ, PYTHONPATH=f"{root}/python")
    cmd = ["taskset", "-c", str(core), py, "-u",
           f"{root}/discopt_benchmarks/scripts/gdp_baseline.py", "--worker", inst, arm, str(TL)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TL + GRACE, cwd=root, env=env)
        lines = [x for x in p.stdout.splitlines() if x.startswith("@@RESULT ")]
        rec = json.loads(lines[-1][9:]) if lines else {"error": (p.stderr or "")[-300:], "rc": p.returncode}
    except subprocess.TimeoutExpired:
        rec = {"status": "killed", "wall": TL + GRACE}
    rec.update(instance=inst, arm=arm, tree=tree, core=core,
               elapsed=round(time.perf_counter() - t0, 2), load1=round(os.getloadavg()[0], 2))
    assert rec.get("discopt", root).startswith(root), (rec.get("discopt"), root)  # §8
    return rec


def lane(core):
    while True:
        with lock:
            if not queue:
                return
            pair = queue.pop(0)
        for inst, arm, tree in pair:
            if (inst, arm, tree) in done:
                continue
            rec = run(inst, arm, tree, core)
            with lock:
                with open(out, "a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                print(f"core{core} {inst:34s} {arm:18s} {tree} {rec.get('status')} "
                      f"wall={rec.get('wall')}", flush=True)


th = [threading.Thread(target=lane, args=(c,)) for c in (0, 1, 2)]
for t in th:
    t.start()
for t in th:
    t.join()
print("OLDNEW_DONE", flush=True)
