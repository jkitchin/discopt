"""Bound-neutral panel for #1147: node_count and certified objective must be
EXACTLY unchanged by the provenance carry (CLAUDE.md §5, bound-neutral regime).

Usage:  python -u scratchpad/panel_1147.py <arm-name> <out.json> [time_limit]

Asserts which code it loaded (CLAUDE.md §8): both ``discopt.__file__`` and a
marker unique to the version under test, printed so a baseline run can be
checked to NOT carry it. Prints per-instance progress (§10) and an
executed-comparison count, exiting non-zero if it solved nothing (§6).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import discopt
import discopt.mpec as mpec
from discopt.modeling.core import from_nl

MARKER = "carry_complementarities"

arm = sys.argv[1]
out_path = Path(sys.argv[2])
time_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

print(f"[{arm}] discopt.__file__ = {discopt.__file__}", flush=True)
print(f"[{arm}] mpec.__file__    = {mpec.__file__}", flush=True)
print(f"[{arm}] marker {MARKER!r} present: {hasattr(mpec, MARKER)}", flush=True)

corpus = sorted(Path("python/tests/data/minlplib_nl").glob("*.nl"))
assert corpus, "corpus is empty — the panel would report a vacuous pass"

rows = {}
solved = 0
for i, path in enumerate(corpus, 1):
    t0 = time.time()
    try:
        m = from_nl(str(path))
        res = m.solve(time_limit=time_limit)
        rows[path.stem] = {
            "status": str(res.status),
            "objective": None if res.objective is None else float(res.objective),
            "bound": None if res.bound is None else float(res.bound),
            "node_count": int(res.node_count),
            "gap_certified": bool(getattr(res, "gap_certified", False)),
        }
        solved += 1
    except Exception as exc:  # recorded, never swallowed into a silent pass
        rows[path.stem] = {"error": f"{type(exc).__name__}: {exc}"}
    print(
        f"[{arm}] {i:3d}/{len(corpus)} {path.stem:28s} {rows[path.stem]} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

out_path.write_text(json.dumps({"arm": arm, "marker": hasattr(mpec, MARKER), "rows": rows}, indent=1))
print(f"[{arm}] EXECUTED_SOLVES: {solved} / {len(corpus)}", flush=True)
if solved == 0:
    raise SystemExit(f"[{arm}] solved nothing — the panel measured nothing")
