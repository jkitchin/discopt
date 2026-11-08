#!/usr/bin/env python3
"""#2212: `hda`'s root solve under one arm of `DISCOPT_LP_LOGICAL_DUAL_COLD`.

Run once per arm (the flag is a Rust `OnceLock`, read once per process)::

    DISCOPT_PROFILE=1 DISCOPT_LP_LOGICAL_DUAL_COLD=0 python hda_300s.py 1
    DISCOPT_PROFILE=1 DISCOPT_LP_LOGICAL_DUAL_COLD=1 python hda_300s.py 1

Prints one JSON line (`hda_300s.jsonl` holds both arms' lines).
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "python"))
import discopt  # noqa: E402
from discopt._rust import profile_counters_py, profile_reset_py  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert Path(discopt.__file__).resolve().is_relative_to(REPO), discopt.__file__

# #2212: `DISCOPT_LP_LOGICAL_DUAL_COLD` was RETIRED (docs/dev/flag-retirement-audit.md).
# Against a tree without it, the "ON" arm runs the same code as OFF and would read as
# "no effect" (CLAUDE.md §6), so refuse instead. Re-run from 70ae79b or earlier.
if os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "").strip().lower() in ("1", "true", "on"):
    raise SystemExit(
        "DISCOPT_LP_LOGICAL_DUAL_COLD was retired (#2212); this arm would measure the "
        "default path. Check out 70ae79b or earlier to reproduce the recorded rows."
    )

PREFIXES = (
    "LogicalDualCold",
    "LpVerdict",
    "DualColdFallbacks",
    "DualWarmSolves",
    "WarmVerdict",
    "Phase1Pivots",
)

profile_reset_py()
t = time.time()
model = from_nl(str(REPO / "python/tests/data/minlplib_nl/hda.nl"))
r = model.solve(deterministic=True, max_nodes=int(sys.argv[1]), time_limit=300)
wall = time.time() - t
c = profile_counters_py()
row = {
    "arm": os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "0"),
    "nodes": r.node_count,
    "status": str(r.status),
    "bound": r.bound,
    "wall": round(wall, 1),
}
row.update({k: v for k, v in c.items() if v and k.startswith(PREFIXES)})
print(json.dumps(row))
assert row.get("LpVerdictOptimal", 0) > 0, "no LP verdict counted: run with DISCOPT_PROFILE=1"
