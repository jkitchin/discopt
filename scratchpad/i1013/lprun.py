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
for k in (
    "DualDegeneratePivots",
    "DualStallTrips",
    "DualBlandActivations",
    "DualWarmSolves",
    "DualColdFallbacks",
    "DualDegenerateRunArms",
    "DualDegenerateRunMax",
    "DualDegenerateStallBails",
):
    if k in snap:
        rec[k] = snap[k]
print("RES " + json.dumps(rec), flush=True)
