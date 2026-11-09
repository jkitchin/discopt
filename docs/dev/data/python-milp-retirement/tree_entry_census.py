"""Python-MILP-tree retirement, step 1: which solves reach `_solve_milp_bb`, and why.

Solves every in-repo corpus instance (python/tests/data/minlplib_nl/*.nl) in a fresh
subprocess with `DISCOPT_LP_MILP_BACKEND=rust` (the in-house MILP route; the same as an
install without highspy) and records, per solve, the route, status, certification and
`solver_stats["route/python_milp_tree_entry"]` (the reason code, see
`solver._PY_MILP_TREE_REASONS`; absent when the tree did not run). Prints per-item
progress and an executed count; exits 1 when nothing ran.

    python -u docs/dev/data/python-milp-retirement/tree_entry_census.py OUT.jsonl [TL] [JOBS]
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CORPUS = sorted((REPO / "python/tests/data/minlplib_nl").glob("*.nl"))
OUT = Path(sys.argv[1])
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
JOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

PROBE = """
import json, sys, time, warnings
warnings.simplefilter("ignore")
import discopt.modeling as dm
t = time.perf_counter()
r = dm.from_nl(sys.argv[1]).solve(time_limit=float(sys.argv[2]))
st = r.solver_stats or {}
print("ROW " + json.dumps({
    "status": r.status, "certified": bool(r.gap_certified), "objective": r.objective,
    "bound": r.bound, "wall": time.perf_counter() - t, "route": r.algorithm_route,
    "tree_entry": st.get("route/python_milp_tree_entry"),
    "tree_engine_had_incumbent": st.get("route/python_milp_tree_engine_had_incumbent"),
}))
"""


def one(path):
    env = {**os.environ, "DISCOPT_LP_MILP_BACKEND": "rust"}
    p = subprocess.run([sys.executable, "-c", PROBE, str(path), str(TL)], env=env,
                       capture_output=True, text=True, timeout=10 * TL + 300)  # fmt: skip
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
    row = json.loads(rows[0][4:]) if rows else {"error": p.stderr[-1500:]}
    row["instance"] = path.stem
    return row


executed = 0
with ThreadPoolExecutor(JOBS) as ex, OUT.open("w") as fh:
    for row in ex.map(one, CORPUS):
        executed += "error" not in row
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        route = str(row.get("route"))[:60]
        print(f"{row['instance']:24s} {row.get('status')} cert={row.get('certified')} "
              f"tree={row.get('tree_entry')} route={route}", flush=True)  # fmt: skip
print(f"executed {executed} of {len(CORPUS)}", flush=True)
sys.exit(0 if executed else 1)
