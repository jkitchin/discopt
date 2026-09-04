"""#1154 panel, arm 2b — the WITHIN-ARM control for the 3 corpus mismatches.

The corpus differential found 63/66 instances byte-identical between
DISCOPT_GDP_SUMOVER OFF and ON, and 3 that differ: syn05hfsg, tanksize, tls2.

Arm 1 measured ZERO ``SumOverExpression`` nodes in the whole corpus, so the
flag's read sites are structurally unreachable on these instances and a real
flag effect is impossible. That is an argument, not a measurement, so this probe
makes the measurement: it repeats the SAME arm (flag OFF) against itself, and
then the SAME arm (flag ON) against itself, on exactly those three instances.

If the within-arm spread reproduces the between-arm difference, the corpus
"mismatches" are wall-clock truncation artifacts of the role-1 ``time_limit``
(``deterministic=True`` neutralizes only role-2 budgets; the user's time limit
and the phase-entry gates stay wall-dependent), and not a flag effect. This is
the control PR #1150 established when ``beuster`` produced both outcomes in both
arms.

Prints per-rep progress (§10) and an executed-comparison count (§6).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CORPUS = Path("python/tests/data/minlplib_nl")
INSTANCES = ["syn05hfsg.nl", "tanksize.nl", "tls2.nl"]
REPS = 3
TIME_LIMIT = 10.0

CHILD = r'''
import json, os, sys
from discopt.modeling.core import from_nl
r = from_nl(sys.argv[1]).solve(time_limit=float(os.environ["PANEL_TL"]), deterministic=True)
print("RESULT" + json.dumps({
    "status": str(r.status),
    "objective": None if r.objective is None else repr(float(r.objective)),
    "bound": None if r.bound is None else repr(float(r.bound)),
    "node_count": getattr(r, "node_count", None),
}))
'''


def run(path: Path, arm: str) -> dict:
    env = dict(os.environ, DISCOPT_GDP_SUMOVER=arm, PANEL_TL=str(TIME_LIMIT))
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(path)],
        capture_output=True, text=True, env=env, timeout=TIME_LIMIT * 20 + 120,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise RuntimeError(f"no RESULT for {path.name}: {(proc.stderr or proc.stdout)[-400:]}")


print(f"load at start: {open('/proc/loadavg').read().split()[0]}", flush=True)

comparisons = 0
within_arm_varies: dict[str, set[str]] = {}

for name in INSTANCES:
    path = CORPUS / name
    for arm in ("0", "1"):
        seen = set()
        for rep in range(REPS):
            res = run(path, arm)
            key = json.dumps(res, sort_keys=True)
            seen.add(key)
            comparisons += 1
            print(
                f"  {name} arm={arm} rep={rep}: status={res['status']} "
                f"bound={res['bound']} nodes={res['node_count']}",
                flush=True,
            )
        within_arm_varies[f"{name}/arm{arm}"] = seen

print()
for key, seen in within_arm_varies.items():
    print(f"{key}: {len(seen)} distinct outcome(s) in {REPS} reps")
n_varying = sum(1 for s in within_arm_varies.values() if len(s) > 1)
print(f"arms_whose_own_repeats_disagree={n_varying}/{len(within_arm_varies)}")
print(f"executed_comparisons={comparisons}")
if comparisons == 0:
    print("PROBE DID NOT FIRE", file=sys.stderr)
    sys.exit(1)
