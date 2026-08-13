"""#1008 re-scope probe: is the LU factor DENSER than it should be?

The cadence sweep falsified H1 (lengthening the refactorization interval buys
1.1-1.3x against an 8.7-29x gap, and destabilises one instance). The profile put
94.6% of wall in the LU layer split three ways -- factorize 59.5%, FT update
18.7%, ftran/btran 16.4%. A single cause consistent with all three being slow at
once is fill-in: a dense L+U makes the factorization, every eta update and every
triangular solve expensive together.

This measures the fill ratio nnz(L+U)/nnz(B) per sparse factorization, plus the
per-factorization wall, on the captured relaxation LPs. Prints a measured-LP
count; exits non-zero at zero (CLAUDE.md #6). Nothing is caught (#7).
"""

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

import discopt
import discopt._rust as _rust

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__
assert _rust.__file__.startswith(WT), _rust.__file__

MARKER = "LuFactorNnz"
nm_out = subprocess.run(
    f"strings {_rust.__file__}", shell=True, capture_output=True, text=True, check=True
).stdout
assert MARKER in nm_out, "loaded a build without the #1008 fill counters"
print(f"MARKER {MARKER!r} present", flush=True)

MAXROWS = int(os.environ.get("I1008_MAXROWS", "6000"))
ONLY = os.environ.get("I1008_ONLY", "")
paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
if ONLY:
    keep = set(ONLY.split(","))
    paths = [p for p in paths if os.path.basename(p)[:-4] in keep]
paths = [p for p in paths if int(np.load(p)["shape"][0]) <= MAXROWS]
assert paths, "no captured LPs"

from discopt.solvers.milp_simplex import _dual_start_slack_basis

measured = 0
print(f"{'tag':22s} {'m':>5s} {'nnzB/col':>8s} {'facs':>5s} "
      f"{'nnzB':>9s} {'nnzLU':>10s} {'fill':>6s} {'ms/fac':>7s} {'wall':>7s}", flush=True)
for p in paths:
    tag = os.path.basename(p)[:-4]
    z = np.load(p)
    nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
    st = _dual_start_slack_basis(c, lo, hi, nrow)
    assert st is not None, f"{tag}: dual start rejected"
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
        100_000,
        float(os.environ.get("I1008_TL", "45")),
    )
    _rust.profile_reset_py()
    t0 = time.perf_counter()
    out = _rust.solve_lp_warm_csc_py(*args)
    wall = time.perf_counter() - t0
    snap = dict(_rust.profile_counters_py())
    facs = snap["LuSparseFactorizations"]
    assert facs > 0, f"{tag}: no sparse factorization observed -- probe measured nothing"
    nnz_b, nnz_lu = snap["LuBasisNnz"], snap["LuFactorNnz"]
    fill = nnz_lu / nnz_b
    rec = {
        "tag": tag, "m": nrow, "status": out[0], "iters": int(out[3]),
        "facs": int(facs), "nnzB": int(nnz_b // facs), "nnzLU": int(nnz_lu // facs),
        "fill": fill, "wall": wall,
    }
    print(
        f"{tag:22s} {nrow:5d} {nnz_b / facs / nrow:8.2f} {facs:5d} "
        f"{nnz_b // facs:9d} {nnz_lu // facs:10d} {fill:6.2f} "
        f"{'':7s} {wall:7.3f}  {out[0]}",
        flush=True,
    )
    print("JSONF " + json.dumps(rec), flush=True)
    measured += 1

print("measured LPs:", measured, flush=True)
if measured == 0:
    sys.exit(1)
