"""Attribution probe: is contvar's lost bound #928 alone, or the #928x#966 pair?

The 3-arm graduation panel has no arm with DISCOPT_LP_WARM_DEADLINE=1 and the
#966 seam flags OFF, so "the cand arm loses the bound and the seam arm keeps it"
attributes the loss to #928 *given* the seam, not to #928 by itself. This adds
that fourth arm on the two instances that moved, and reports every cell.

Prints an executed-comparison count and exits non-zero if it compared nothing
(CLAUDE.md §6). Worker exceptions propagate (§7).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "python/tests/data/minlplib_nl"
WORKER = Path(__file__).with_name("issue966_coupled_worker.py")
WARM, ROUND, HESS = (
    "DISCOPT_LP_WARM_DEADLINE",
    "DISCOPT_NODE_ROUND_BUDGET",
    "DISCOPT_HESS_COMPILE_GATE",
)

ARMS = (
    ("base", {WARM: "0", ROUND: "0", HESS: "0"}),
    ("warm", {WARM: "1", ROUND: "0", HESS: "0"}),  # <- the missing arm: #928 alone
    ("seam", {WARM: "0", ROUND: "1", HESS: "1"}),
    ("cand", {WARM: "1", ROUND: "1", HESS: "1"}),
)

INSTANCES = sys.argv[1].split(",")
BUDGET = float(sys.argv[2])
REPS = int(sys.argv[3])

fired = 0
print(f"loadavg_start={[round(x, 2) for x in os.getloadavg()]}", flush=True)
rows = []
for rep in range(1, REPS + 1):
    for name in INSTANCES:
        for key, env in ARMS:  # interleaved within instance (§9)
            t = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-u", str(WORKER), str(CORPUS / f"{name}.nl"), str(BUDGET)],
                capture_output=True,
                text=True,
                env={**os.environ, **env},
                timeout=40 * BUDGET + 900,
            )
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-4000:] + "\n")
                raise SystemExit(f"worker failed: {name} {key}")
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            fired += 1
            rows.append({"rep": rep, "instance": name, "arm": key, **r})
            print(
                f"rep{rep} {name:12s} {key:5s} wall={r['wall']:5.1f} nodes={r['node_count']:4d} "
                f"status={r['status']:12s} bound={r['bound']} obj={r['objective']} "
                f"cert={int(bool(r['gap_certified']))}  [{time.perf_counter() - t:.0f}s]",
                flush=True,
            )
print(f"loadavg_end={[round(x, 2) for x in os.getloadavg()]}")
print(f"CELLS_EXECUTED={fired}")
Path(ROOT / "discopt_benchmarks/results/issue928_contvar_attribution.json").write_text(
    json.dumps(rows, indent=2)
)
raise SystemExit(0 if fired else 1)
