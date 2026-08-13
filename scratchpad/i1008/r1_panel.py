"""#1008 R1 graduation panel: `DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY` ON vs OFF.

One process per arm — the flag is read once per process via `OnceLock`, so a
single process cannot measure both. Emits one JSON line per LP; `r1_report.py`
joins the two arms.

Every LP is solved with `time_limit=None`, which is the whole point: that is the
call shape where `lp_bindings` used to withhold the recovery because it rode on
`bank_deadline_duals = deadline.is_some()`.

HiGHS is the oracle. A discopt objective BELOW the HiGHS optimum by more than the
tolerance is a false (too-good) bound on a minimization LP and fails the panel
outright (CLAUDE.md §1/§5 cert-clean); a bound that is merely absent is a
retention loss, counted separately.

Prints a per-LP line as it goes (§10) and an executed count, exiting non-zero if
zero LPs were compared (§6). Nothing is caught (§7).
"""

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import discopt

WT = "/private/tmp/wtR1"
assert discopt.__file__.startswith(WT), discopt.__file__
import discopt._rust as _rust  # noqa: E402

assert _rust.__file__.startswith(WT), _rust.__file__

# §8: the option under test exists ONLY in the build under test. Assert the marker
# is present rather than trusting the path — an editable install can point
# elsewhere than it appears to.
MARKER = "DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY"
blob = open(_rust.__file__, "rb").read()
assert MARKER.encode() in blob, f"loaded a build without {MARKER}: {_rust.__file__}"
print(f"# marker {MARKER} present in {_rust.__file__}", flush=True)

ARM = os.environ["DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY"]
OUT = os.environ["R1_OUT"]
print(f"# arm DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY={ARM!r}", flush=True)

from discopt.solvers.milp_simplex import _dual_start_slack_basis  # noqa: E402

# Iteration cap, applied IDENTICALLY to both arms and reported per LP. It is not
# a deadline: `time_limit=None` — the call shape that used to lose the recovery —
# is preserved exactly. Without a cap the large LPs (QPLIB_1451_rlt0 is
# 7392 x 1890) run for hours at the 100k default and the panel never completes.
# An LP that hits the cap returns `iteration_limit` in BOTH arms, i.e. it
# contributes a neutral row rather than a silently dropped one; `r1_report.py`
# counts them so the coverage this costs is visible (CLAUDE.md: no silent caps).
MAX_ITER = int(os.environ.get("R1_MAX_ITER", "20000"))
print(f"# max_iter cap = {MAX_ITER} (both arms; time_limit stays None)", flush=True)

paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
assert paths, "no captured LPs"

fh = open(OUT, "w")
n_done = 0
n_skipped = 0
skipped = []
for p in paths:
    tag = os.path.basename(p)[:-4]
    z = np.load(p)
    nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]

    r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
    ref = float(r.fun) if r.status == 0 else None

    st = _dual_start_slack_basis(c, lo, hi, nrow)
    if st is None:
        # No dual-feasible slack start => this LP never enters the warm path, so it
        # cannot exercise the option under test. Skipped, but COUNTED and named: a
        # panel that silently drops rows reads as broader coverage than it has.
        n_skipped += 1
        skipped.append(tag)
        print(f"{tag:<24} SKIP (dual start rejected)", flush=True)
        continue
    c_std = np.ascontiguousarray(np.concatenate([c, np.zeros(nrow)]))
    lb_std = np.ascontiguousarray(np.concatenate([lo, np.zeros(nrow)]))
    ub_std = np.ascontiguousarray(np.concatenate([hi, np.full(nrow, np.inf)]))
    af = sp.hstack([A, sp.identity(nrow, format="csc")], format="csc")

    _rust.profile_reset_py()
    t0 = time.perf_counter()
    out = _rust.solve_lp_warm_csc_py(
        c_std,
        nrow,
        ncol + nrow,
        np.ascontiguousarray(af.indptr, dtype=np.int64),
        np.ascontiguousarray(af.indices, dtype=np.int64),
        np.ascontiguousarray(af.data, dtype=np.float64),
        np.ascontiguousarray(b),
        lb_std,
        ub_std,
        np.ascontiguousarray(st[0], dtype=np.int8),
        np.ascontiguousarray(st[1], dtype=np.int64),
        1e-9,
        MAX_ITER,
        None,  # time_limit: the call shape that used to lose the recovery
    )
    wall = time.perf_counter() - t0
    ctr = _rust.profile_counters_py()

    rec = {
        "tag": tag,
        "arm": ARM,
        "rows": nrow,
        "cols": ncol,
        "nnz": int(A.nnz),
        "status": out[0],
        "obj": float(out[2]),
        "iters": int(out[3]),
        "wall": wall,
        "ref": ref,
        "iter_capped": int(out[3]) >= MAX_ITER,
        "recoveries": int(ctr.get("DualUnstablePivotRecoveries", 0)),
        "bails": int(ctr.get("DualUnstablePivotBails", 0)),
    }
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    n_done += 1
    print(
        f"{tag:<24} {rec['status']:<10} obj={rec['obj']:.9g} ref={ref} "
        f"iters={rec['iters']}{'(CAP)' if rec['iter_capped'] else ''} "
        f"rec={rec['recoveries']} bail={rec['bails']} ({wall:.2f}s)",
        flush=True,
    )

fh.close()
print(f"\nexecuted: lps={n_done}  skipped={n_skipped} {skipped}")
assert n_done, "panel compared nothing"
