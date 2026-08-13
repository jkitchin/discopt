"""#1008 D3 entry experiment: does the fixed refactorization interval set the cost?

Hypothesis (from the attribution over 19 captured relaxation LPs): `LuNumeric` is
72.6% of LP wall and `FtUpdate` is 1.9%, so the hardcoded `updates >= 48` cap
forces the dominant cost far more often than the updates it truncates are worth.
Raising the interval should cut `LuSparseFactorizations` and the total factor
nonzeros roughly in proportion, without changing the LP's answer.

Kill criterion: if raising the interval does NOT reduce total factor work, the
interval is not what sets the cost and this direction dies here. If it reduces
factor work but any LP's objective moves off the HiGHS optimum, it dies for
correctness (CLAUDE.md §1, zero slack).

**Counter-based on purpose.** The R1 graduation panel is running on this machine,
so wall-clock is contended and a timing claim would violate §9 (no load gate, no
interleave, no spread). `LuSparseFactorizations` and `LuFactorNnz` are exact
integer counts, independent of load — they measure the factor work performed, and
`LuNumeric` time is proportional to it at fixed fill. Wall is recorded but is
reported as directional only and no speed claim is made from it.

One process per interval: the gate is a `OnceLock`, read once per process.

Prints an executed-comparison count and exits non-zero if it is zero (§6).
Nothing is caught (§7).
"""

import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import discopt

WT = "/private/tmp/wt1008d"
assert discopt.__file__.startswith(WT), discopt.__file__
import discopt._rust as _rust  # noqa: E402

assert _rust.__file__.startswith(WT), _rust.__file__

# §8: assert the marker unique to the build under test. The 48-cap was a literal
# before this change, so the env-var name appears ONLY in a build that has the
# gate — a stale .so from another worktree fails here instead of silently
# reporting the baseline twice.
MARKER = "DISCOPT_LP_REFAC_INTERVAL"
blob = open(_rust.__file__, "rb").read()
assert MARKER.encode() in blob, f"loaded a build without {MARKER}: {_rust.__file__}"
print(f"# marker {MARKER} present in {_rust.__file__}", flush=True)

INTERVAL = os.environ[MARKER]
OUT = os.environ["REFAC_OUT"]
MAX_ITER = int(os.environ.get("REFAC_MAX_ITER", "20000"))
print(f"# arm {MARKER}={INTERVAL!r}  max_iter={MAX_ITER}", flush=True)

from discopt.solvers.milp_simplex import _dual_start_slack_basis  # noqa: E402

paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
assert paths, "no captured LPs"
LIMIT = int(os.environ.get("REFAC_LIMIT", "0"))
if LIMIT:
    paths = paths[:LIMIT]

fh = open(OUT, "w")
n_done = 0
n_skipped = 0
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
        n_skipped += 1
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
        None,
    )
    wall = time.perf_counter() - t0
    ctr = _rust.profile_counters_py()

    rec = {
        "tag": tag,
        "interval": INTERVAL,
        "rows": nrow,
        "cols": ncol,
        "status": out[0],
        "obj": float(out[2]),
        "wall": wall,
        "ref": ref,
        "facs": int(ctr.get("LuSparseFactorizations", 0)),
        "basis_nnz": int(ctr.get("LuBasisNnz", 0)),
        "factor_nnz": int(ctr.get("LuFactorNnz", 0)),
        "dual_refac": int(ctr.get("DualRefactorizations", 0)),
        "dual_refac_cap": int(ctr.get("DualRefacCap", 0)),
        "dual_refac_ft": int(ctr.get("DualRefacFtFail", 0)),
        "primal_refac": int(ctr.get("Refactorizations", 0)),
        "dual_pivots": int(ctr.get("DualDegeneratePivots", 0)),
        "p1": int(ctr.get("Phase1Pivots", 0)),
        "p2": int(ctr.get("Phase2Pivots", 0)),
        "cold_fallback": int(ctr.get("DualColdFallbacks", 0)),
    }
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    n_done += 1
    print(
        f"{tag:<24} {rec['status']:<10} obj={rec['obj']:.9g} "
        f"facs={rec['facs']} (dual_cap={rec['dual_refac_cap']} "
        f"dual_ft={rec['dual_refac_ft']} primal={rec['primal_refac']}) "
        f"fnnz={rec['factor_nnz']} ({wall:.2f}s)",
        flush=True,
    )

print(f"\nexecuted: solved={n_done} skipped={n_skipped}", flush=True)
sys.exit(0 if n_done else 1)
