"""#1008: does feral's post-0.15.1 basis triangularization cut LU fill?

Context. §18f of docs/dev/performance-plan.md attributed 72.6% of LP wall to
`LuNumeric` and named *fill* — not refactorization frequency — as the remaining
lever. It also eliminated "ordering" on the grounds that feral's `analyze`
already triangularizes the basis and runs AMD only on the residual bump. **That
elimination was read off feral's unreleased main, not the v0.15.1 that discopt
links.** `git grep "fn triangularize" v0.15.1` returns nothing; triangularization
landed in feral 1217992 (PR #160), seven commits past the tag. So the ordering
lever was never actually tested on discopt's build (CLAUDE.md §11 retraction is
recorded in the plan doc).

Hypothesis: building discopt against feral b071d54 (Suhl-Suhl triangularization +
AMD on the residual bump) reduces the factor nonzeros per factorization on the
captured relaxation LPs. Fill = LuFactorNnz / LuBasisNnz.

Kill criterion: if the fill ratio and total factor nonzeros are unchanged (within
a percent) across the two arms, triangularization does not help these bases and
the fill lever dies on feral's side — the remaining gap is discopt's to close.
If any LP's objective moves off the HiGHS optimum, it dies for correctness first
(§1, zero slack).

Counter-based. `LuFactorNnz`/`LuBasisNnz`/`LuSparseFactorizations` are exact
integer counts, independent of machine load — the load gate that a timing claim
would need (§9) does not apply to them. Wall is recorded but reported as
directional only; the timing arm is a separate interleaved run.

§8: each arm asserts a marker that discriminates the two builds. `sparse_triangular`
appears in the binary only when feral's post-0.15.1 LU is linked, so a stale .so
fails loudly here instead of silently reporting one arm twice.

§6: prints an executed-comparison count and exits non-zero if it is zero.
§7: nothing is caught.
"""

import glob
import json
import os
import sys
import time

import discopt
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

WT = os.environ["WT"]
ARM = os.environ["FERAL_ARM"]  # a crates.io version, e.g. "0.15.1" or "0.16.0"
OUT = os.environ["FERAL_OUT"]
LP_DIR = os.environ.get("LP_DIR", "/private/tmp/wt1008d/scratchpad/i1008/lps")
MAX_ITER = int(os.environ.get("REFAC_MAX_ITER", "20000"))

assert discopt.__file__.startswith(WT), discopt.__file__
import discopt._rust as _rust  # noqa: E402

assert _rust.__file__.startswith(WT), _rust.__file__
blob = open(_rust.__file__, "rb").read()
# §8: cargo bakes the dependency source path — hence the literal `feral-<version>`
# — into the extension's panic locations, so the arm name IS the marker. The
# foreign marker must be ABSENT or a stale .so reports one arm twice.
want_marker = f"feral-{ARM}".encode()
other_marker = b"feral-0.16.0" if ARM != "0.16.0" else b"feral-0.15.1"
assert want_marker in blob, f"arm {ARM}: marker {want_marker!r} absent from {_rust.__file__}"
assert other_marker not in blob, f"arm {ARM}: foreign marker {other_marker!r} present"
assert b"DISCOPT_LP_REFAC_INTERVAL" in blob, "build predates the #1024 counters"
# The counters are env-gated (profile.rs `init_from_env`). Without this the whole
# run reports facs=0/bnnz=0 and reads as "no factorization work" — the exact
# silent no-op §6 exists to prevent. Caught once on the first launch of this
# script; asserted here so it cannot recur.
assert os.environ.get("DISCOPT_PROFILE"), "DISCOPT_PROFILE must be set or every counter reads 0"
# `DISCOPT_LU_TRIANGULARIZE` was removed with the peel (§18i); a leftover setting
# in the environment means the caller is running a stale recipe, so refuse.
assert not os.environ.get("DISCOPT_LU_TRIANGULARIZE"), (
    "DISCOPT_LU_TRIANGULARIZE no longer exists; unset it (the peel was removed in §18i)"
)
print(f"# arm={ARM} marker={want_marker!r} present, {other_marker!r} absent so={_rust.__file__}",
      flush=True)

from discopt.solvers.milp_simplex import _dual_start_slack_basis  # noqa: E402

paths = sorted(glob.glob(os.path.join(LP_DIR, "*.npz")))
assert paths, f"no captured LPs in {LP_DIR}"
LIMIT = int(os.environ.get("REFAC_LIMIT", "0"))
if LIMIT:
    paths = paths[:LIMIT]

fh = open(OUT, "w")
n_done = 0
n_skipped = 0
n_facs = 0
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
        "arm": ARM,
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
        "p1": int(ctr.get("Phase1Pivots", 0)),
        "p2": int(ctr.get("Phase2Pivots", 0)),
        "cold_fallback": int(ctr.get("DualColdFallbacks", 0)),
    }
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    n_done += 1
    n_facs += rec["facs"]
    fill = rec["factor_nnz"] / rec["basis_nnz"] if rec["basis_nnz"] else float("nan")
    print(
        f"{tag:<24} {rec['status']:<10} obj={rec['obj']:.9g} "
        f"facs={rec['facs']} bnnz={rec['basis_nnz']} fnnz={rec['factor_nnz']} "
        f"fill={fill:.2f}x ({wall:.2f}s)",
        flush=True,
    )

print(f"\nexecuted: solved={n_done} skipped={n_skipped} factorizations={n_facs}", flush=True)
sys.exit(0 if (n_done and n_facs) else 1)
