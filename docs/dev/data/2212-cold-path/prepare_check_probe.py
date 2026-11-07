#!/usr/bin/env python3
"""#2212: does the new `PreparedDual::prepare` consistency check ever fire on a real solve?

The check refuses a start whose `col_status` and `basic_vars` disagree. It is meant
to catch only malformed starts from outside the engine; if any engine-internal
caller produced such a start, the check would move that caller from the warm path
to the cold one, a behavior change. This solves every in-repo corpus instance
(`python/tests/data/minlplib_nl`, deterministic, 200 nodes, 60 s backstop) with the
profile counters on and sums `DualPrepRejectInconsistent`, which only the new check
increments. Expected: 0. Prints per-instance progress and an executed count (§6, §10).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
import discopt  # noqa: E402
from discopt._rust import profile_counters_py, profile_reset_py  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert Path(discopt.__file__).resolve().is_relative_to(REPO), discopt.__file__
blob = (Path(discopt.__file__).parent / "_rust.abi3.so").read_bytes()
assert b"DualPrepRejectInconsistent" in blob, "build without the #2212 check"

n, fired, accepts = 0, 0, 0
for p in sorted((REPO / "python/tests/data/minlplib_nl").glob("*.nl")):
    profile_reset_py()
    r = from_nl(str(p)).solve(deterministic=True, max_nodes=200, time_limit=60)
    c = profile_counters_py()
    k = int(c.get("DualPrepRejectInconsistent", 0))
    fired += k
    accepts += int(c.get("DualPrepAccept", 0))
    n += 1
    acc = c.get("DualPrepAccept", 0)
    print(f"{p.stem:24s} {str(r.status):14s} inconsistent={k} prep_accept={acc}", flush=True)
print(f"INSTANCES {n}  DualPrepAccept total {accepts}  DualPrepRejectInconsistent total {fired}")
sys.exit(0 if n and accepts and fired == 0 else 1)
