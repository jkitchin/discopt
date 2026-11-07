"""#2202 fire screen: which .nl instances does DISCOPT_NODE_LP_SLACK_RETRY act on?
(after docs/dev/data/2142-dual-proof/panel/v4_screen.py)

    python -u fire_screen.py OUT.jsonl TIME_LIMIT MAX_NODES file.nl ...

ON arm only (the default), DISCOPT_PROFILE=1, deterministic under MAX_NODES, one fresh
process per instance; records the NodeLpSlackRetry* profile counters and the kernel's
tree/slack_retry* and tree/undecided_nodes stats. Asserts every child loaded this tree's
discopt with the #2202 kwarg (CLAUDE.md section 8) and prints an executed-assertion count
(section 6); exits non-zero when nothing was asserted.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[4])
out, tl, mn, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
CHILD = r"""
import json, sys
import discopt, discopt.modeling as dm
from discopt import _rust
from discopt._rust import profile_counters_py, profile_reset_py
assert discopt.__file__.startswith(sys.argv[4]), discopt.__file__
assert "node_lp_slack_retry" in (_rust.solve_spatial_tree_py.__text_signature__ or "")
m = dm.from_nl(sys.argv[1]); profile_reset_py()
r = m.solve(time_limit=float(sys.argv[2]), deterministic=True, max_nodes=int(sys.argv[3]))
c = dict(profile_counters_py())
s = r.solver_stats or {}
print("SCREEN " + json.dumps({"status": str(r.status), "nodes": int(r.node_count or 0),
      "bound": r.bound, "objective": r.objective,
      **{k: int(v) for k, v in c.items() if k.startswith("NodeLpSlack")},
      **{k: s.get(k) for k in ("tree/slack_retries", "tree/slack_retry_decided",
                               "tree/undecided_nodes")}}), flush=True)
"""
n = 0
for f in files:
    env = {**os.environ, "DISCOPT_PROFILE": "1", "PYTHONHASHSEED": "0"}
    env.pop("DISCOPT_NODE_LP_SLACK_RETRY", None)
    p = subprocess.run(
        [sys.executable, "-c", CHILD, f, tl, mn, REPO + "/python"],
        env=env,
        capture_output=True,
        text=True,
        timeout=float(tl) + 300,
    )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("SCREEN ")]
    rec = {"instance": os.path.basename(f)[:-3]}
    if line:
        rec.update(json.loads(line[0][7:]))
        n += 1
    else:
        rec["error"] = (p.stderr or "")[-400:]
    print(json.dumps(rec), flush=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
print(f"executed assertions: {n}")
sys.exit(0 if n else 1)
