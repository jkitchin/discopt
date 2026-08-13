"""#1008: how much of the LU fill is AVOIDABLE?

The corpus arms show threshold pivoting alone barely moves the fill. Before
investing in a bigger LU change, measure the headroom: factor the SAME basis
that discopt's engine ends on with a reference sparse LU (SuperLU via
scipy.sparse.linalg.splu, COLAMD + threshold pivoting + a maximum transversal)
and compare nnz(L+U).

- If SuperLU's fill is far lower on the same matrix, the fill is an artifact of
  our factorization and is worth fixing.
- If SuperLU's fill is comparable, the fill is intrinsic to these bases and the
  whole fill theory dies.

Same basis on both sides, so this isolates the factorization from the pivot
path. Prints a compared count; exits non-zero at zero (CLAUDE.md #6). Nothing is
caught (#7).
"""

import glob
import json
import os
import subprocess
import sys

# The counters are gated by `profile::enabled()`. Without this the feral column
# reads 0 and the whole comparison is a no-op that looks like an answer — the
# exact CLAUDE.md #6 failure. Set before discopt is imported, and asserted
# non-zero on every row below.
os.environ["DISCOPT_PROFILE"] = "1"

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import discopt
import discopt._rust as _rust

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__
assert _rust.__file__.startswith(WT), _rust.__file__
strs = subprocess.run(
    f"strings {_rust.__file__}", shell=True, capture_output=True, text=True, check=True
).stdout
assert "LuFactorNnz" in strs, "loaded a build without the #1008 fill counters"

from discopt.solvers.milp_simplex import _dual_start_slack_basis

MAXROWS = int(os.environ.get("I1008_MAXROWS", "6000"))
paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
paths = [p for p in paths if int(np.load(p)["shape"][0]) <= MAXROWS]
assert paths, "no captured LPs"

print(
    f"{'tag':22s} {'m':>5s} {'nnzB':>8s} {'feral':>9s} {'f-fill':>6s} "
    f"{'splu(1.0)':>10s} {'splu(0.1)':>10s} {'s-fill':>6s} {'headroom':>8s}",
    flush=True,
)
compared = 0
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
    out = _rust.solve_lp_warm_csc_py(*args)
    snap = dict(_rust.profile_counters_py())
    assert snap["LuBasisNnz"] > 0, f"{tag}: fill counters read 0 — profiling is off, nothing measured"
    assert snap["LuSparseFactorizations"] > 0, f"{tag}: no SPARSE factorizations recorded"
    facs = snap["LuSparseFactorizations"]
    feral_fill = snap["LuFactorNnz"] / snap["LuBasisNnz"]

    # The basis the engine ENDED on, factored by SuperLU on the same matrix.
    basic = np.asarray(out[5], dtype=np.int64)
    assert basic.shape[0] == nrow, f"{tag}: basic_vars length {basic.shape[0]} != m {nrow}"
    B = af[:, basic].tocsc()
    nnzB = int(B.nnz)
    fills = {}
    for thresh in (1.0, 0.1):
        lu = splu(B, permc_spec="COLAMD", diag_pivot_thresh=thresh)
        fills[thresh] = int(lu.L.nnz + lu.U.nnz)
    s_fill = fills[0.1] / nnzB
    print(
        f"{tag:22s} {nrow:5d} {nnzB:8d} {snap['LuFactorNnz'] // facs:9d} {feral_fill:6.2f} "
        # Headroom compares FILL RATIOS, not raw nnz: feral's is averaged over the
        # bases its own path visits, splu's is the final basis, so the raw counts
        # are not commensurable but the ratios are.
        f"{fills[1.0]:10d} {fills[0.1]:10d} {s_fill:6.2f} "
        f"{feral_fill / s_fill:8.2f}x",
        flush=True,
    )
    print("JSONH " + json.dumps({
        "tag": tag, "m": nrow, "nnzB_final": nnzB, "feral_nnzLU_avg": int(snap["LuFactorNnz"] // facs),
        "feral_fill_avg": feral_fill, "splu_nnzLU_t1": fills[1.0], "splu_nnzLU_t01": fills[0.1],
        "splu_fill_t01": s_fill, "status": out[0], "iters": int(out[3]),
    }), flush=True)
    compared += 1

print("compared bases:", compared, flush=True)
if compared == 0:
    sys.exit(1)
