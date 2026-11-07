"""#2202: nous1 to certification on the default path, retry OFF vs ON (CLAUDE.md §9).

    python -u nous1_timing.py REPS > nous1_timing.log

One fresh process per solve, arms interleaved with the order alternating by repetition,
each run gated on the 1-minute load average (waits until it is below 1.5). Reports mean
and standard deviation of wall per arm, plus status / certification / nodes per run.
Prints an executed-assertion count (§6) and asserts this tree's build (§8).
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NL = REPO / "python/tests/data/issue_1871/nous1.nl"
CHILD = r"""
import json, sys, time
import discopt, discopt.modeling as dm
from discopt import _rust
assert discopt.__file__.startswith(sys.argv[2]), discopt.__file__
assert "node_lp_slack_retry" in (_rust.solve_spatial_tree_py.__text_signature__ or "")
t = time.perf_counter()
r = dm.from_nl(sys.argv[1]).solve(time_limit=150, gap_tolerance=1e-4)
w = time.perf_counter() - t
s = r.solver_stats or {}
print("RUN " + json.dumps({"status": r.status, "cert": bool(r.gap_certified), "bound": r.bound,
      "nodes": int(r.node_count or 0), "wall": w,
      "undecided": s.get("tree/undecided_nodes"), "slack_decided": s.get("tree/slack_retry_decided")}))
"""


def run(flag: str) -> dict:
    while os.getloadavg()[0] > 1.5:
        time.sleep(5)
    load = os.getloadavg()[0]
    env = {**os.environ, "DISCOPT_NODE_LP_SLACK_RETRY": flag, "PYTHONHASHSEED": "0"}
    p = subprocess.run(
        [sys.executable, "-c", CHILD, str(NL), str(REPO / "python")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RUN ")]
    if not line:
        raise RuntimeError(p.stderr[-2000:])
    rec = {"flag": flag, "load": round(load, 2), **json.loads(line[0][4:])}
    print(json.dumps(rec), flush=True)
    return rec


reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
walls: dict[str, list[float]] = {"0": [], "1": []}
n = 0
for k in range(reps):
    for flag in ("0", "1") if k % 2 == 0 else ("1", "0"):
        rec = run(flag)
        walls[flag].append(rec["wall"])
        n += 1
for flag, w in walls.items():
    sd = statistics.stdev(w) if len(w) > 1 else float("nan")
    print(f"retry={flag}: wall {statistics.mean(w):.2f} +- {sd:.2f} s over {len(w)} runs")
print(f"executed assertions: {n}")
sys.exit(0 if n else 1)
