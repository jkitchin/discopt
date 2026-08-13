"""#1013: solve one captured .npz LP through the warm dual path; print JSON.

Run as a CHILD process (env knobs are read once per process). Prints one `RES `
line with status/objective/iterations and the LP-stats counters. Nothing is
caught (CLAUDE.md §7).
"""

import json
import os
import sys
import time

import discopt._rust as _rust
import numpy as np
import scipy.sparse as sp
from discopt.solvers.milp_simplex import _dual_start_slack_basis

path = sys.argv[1]
tl = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

z = np.load(path)
nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
st = _dual_start_slack_basis(c, lo, hi, nrow)
if st is None:
    # No sign-matched dual-feasible slack basis (a column whose objective sign
    # selects an infinite bound): this LP has no warm dual start to measure.
    print(
        "SKIP " + json.dumps({"lp": os.path.basename(path)[:-4], "why": "no dual start"}),
        flush=True,
    )
    sys.exit(0)
af = sp.hstack([A, sp.identity(nrow, format="csc")], format="csc")
args = (
    np.ascontiguousarray(np.concatenate([c, np.zeros(nrow)])),
    nrow,
    ncol + nrow,
    np.ascontiguousarray(af.indptr, dtype=np.int64),
    np.ascontiguousarray(af.indices, dtype=np.int64),
    np.ascontiguousarray(af.data, dtype=np.float64),
    np.ascontiguousarray(b),
    np.ascontiguousarray(np.concatenate([lo, np.zeros(nrow)])),
    np.ascontiguousarray(np.concatenate([hi, np.full(nrow, np.inf)])),
    np.ascontiguousarray(st[0], dtype=np.int8),
    np.ascontiguousarray(st[1], dtype=np.int64),
    1e-9,
    100000,
    tl,
)
t0 = time.perf_counter()
out = _rust.solve_lp_warm_csc_py(*args)
wall = time.perf_counter() - t0
snap = dict(_rust.profile_counters_py())
rec = {
    "lp": os.path.basename(path)[:-4],
    "rows": nrow,
    "cols": ncol,
    "wall": wall,
    "status": out[0],
    "obj": out[2],
    "iters": int(out[3]),
}

# CLAUDE.md §8: prove which build we loaded before recording anything from it.
# These counters ship WITH the #1013 change, so a stale `_rust` (built before it)
# exposes none of them. The previous `if k in snap` skipped them silently, which
# turned "this panel measured a build without the fix" into a full set of
# clean-looking RES lines -- the exact failure mode §8 exists to stop.
COUNTERS = (
    "DualDegeneratePivots",
    "DualStallTrips",
    "DualBlandActivations",
    "DualWarmSolves",
    "DualColdFallbacks",
    "DualDegenerateRunArms",
    "DualDegenerateRunMax",
    "DualDegenerateStallBails",
)
missing = [k for k in COUNTERS if k not in snap]
if missing:
    raise SystemExit(
        f"stale or wrong discopt._rust at {_rust.__file__}: missing counters {missing}. "
        f"`DualDegenerateStallBails` is unique to #1013, so this build predates the "
        f"change under test -- rebuild (maturin develop) before running the panel."
    )
# ...and PRESENT is not FIRED (CLAUDE.md §6). The counters are gated on
# DISCOPT_PROFILE: without it every one of them reads 0, which is a perfectly
# well-formed record that says "no degeneracy, no bail, no warm solve" about a
# solve that just did 1279 warm dual pivots with 553 degenerate ones. A driver
# that forgot the variable produced exactly that -- 32 clean cells, every counter
# 0 -- and the presence check above passed the whole way, because the symbols
# existed and only their values were meaningless. `DualWarmSolves` is the proof:
# this script only ever calls the warm entry point, so it CANNOT legitimately be 0.
if snap["DualWarmSolves"] < 1:
    raise SystemExit(
        "counters are present but did not fire (DualWarmSolves=0 after a warm "
        "solve): set DISCOPT_PROFILE=1. Without it every counter reads 0 and the "
        "record looks clean while measuring nothing."
    )
rec["_rust"] = _rust.__file__
for k in COUNTERS:
    rec[k] = snap[k]
print("RES " + json.dumps(rec), flush=True)
