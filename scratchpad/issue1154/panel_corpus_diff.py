"""#1154 panel, arm 2 — CORPUS DIFFERENTIAL (bound-neutral regime, CLAUDE.md §5).

Solves every vendored MINLPLib instance twice, INTERLEAVED (§9) under
``deterministic=True`` so node counts are comparable, with DISCOPT_GDP_SUMOVER
OFF then ON, and requires:

  * ``node_count`` exactly unchanged,
  * ``objective`` exactly unchanged,
  * ``bound`` exactly unchanged,
  * ``status`` unchanged.

Arm 1 (panel_inertness.py) measured zero ``SumOverExpression`` nodes in the whole
corpus, so the flag's own read sites are unreachable here. What this arm is
really regression-testing is the part of #1154 that is NOT behind the flag: the
unconditional independent-walker coverage guard in ``_reformulate_hull``.

Prints per-instance progress (§10) and an executed-comparison count (§6).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CORPUS = Path("python/tests/data/minlplib_nl")
TIME_LIMIT = float(os.environ.get("PANEL_TL", "10"))

CHILD = r'''
import json, os, sys
import discopt
from discopt.modeling.core import from_nl
path = sys.argv[1]
m = from_nl(path)
r = m.solve(time_limit=float(os.environ["PANEL_TL"]), deterministic=True)
print("RESULT" + json.dumps({
    "status": str(r.status),
    "objective": None if r.objective is None else repr(float(r.objective)),
    "bound": None if r.bound is None else repr(float(r.bound)),
    "node_count": getattr(r, "node_count", None),
    "sources": discopt.__file__,
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
            import json
            return json.loads(line[len("RESULT"):])
    return {"error": (proc.stderr or proc.stdout)[-400:]}


compared = 0
mismatches: list[str] = []
errors: list[str] = []

for path in sorted(CORPUS.glob("*.nl")):
    off = run(path, "0")
    on = run(path, "1")
    if "error" in off or "error" in on:
        errors.append(f"{path.name}: OFF={off.get('error','ok')!r} ON={on.get('error','ok')!r}")
        print(f"  {path.name}: ERROR", flush=True)
        continue
    compared += 1
    same = off == on
    if not same:
        mismatches.append(f"{path.name}: OFF={off} ON={on}")
    print(
        f"  {path.name}: {'SAME' if same else 'DIFFERS'} "
        f"status={off['status']} obj={off['objective']} bound={off['bound']} "
        f"nodes={off['node_count']}",
        flush=True,
    )

print()
print(f"instances_compared={compared}")
print(f"mismatches={len(mismatches)}")
for line in mismatches:
    print("  MISMATCH", line)
print(f"errored={len(errors)}")
for line in errors:
    print("  ERROR", line)
print(f"executed_comparisons={compared}")
if compared == 0:
    print("PROBE DID NOT FIRE", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if mismatches else 0)
