"""#1008 H4: is the NUMERIC sparse LU slow per unit of work?

H3 splits "refactorize" into symbolic (11-21% of wall) and numeric (47-52%).
`headroom.py` already proved the factor feral produces is the same SIZE as
SuperLU's on the same basis (median 0.94x). So if feral's numeric factorization
also costs the same wall per factorization as SuperLU's, the LU is simply not the
lever and the gap lives elsewhere; if it costs several times more for the same
output, the numeric kernel is the gap.

Measures: feral's ms/numeric-factorization from the profile dump, against
`scipy.sparse.linalg.splu` timed on the basis the engine ENDS on (best of 3, to
suppress a single scheduling artifact). Caveat stated in the report: feral's
figure is averaged over every basis on its path — early bases are sparser than
the final one, so this comparison is BIASED IN FERAL'S FAVOUR.

Prints a compared count and exits non-zero at zero (#6). Nothing is caught (#7).
"""

import glob
import json
import os
import subprocess
import sys
import time

os.environ["DISCOPT_PROFILE"] = "1"

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import discopt
import discopt._rust as _rust

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__
strs = subprocess.run(
    f"strings {_rust.__file__}", shell=True, capture_output=True, text=True, check=True
).stdout
assert "LuNumeric" in strs, "loaded a build without the #1008 LU phase split"

from discopt.solvers.milp_simplex import _dual_start_slack_basis

h3 = {}
with open(os.path.join(WT, "scratchpad/i1008/h3.log")) as f:
    for ln in f:
        if ln.startswith("JSON3 "):
            r = json.loads(ln[6:])
            h3[r["tag"]] = r
assert h3, "h3.log has no JSON3 rows — run h3.py first"

paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
print(
    f"{'tag':22s} {'m':>5s} {'nnzB':>7s} {'feral ms/fac':>12s} {'splu ms':>9s} {'slower':>7s}",
    flush=True,
)
compared = 0
for p in paths:
    tag = os.path.basename(p)[:-4]
    assert tag in h3, f"{tag}: no h3 row"
    z = np.load(p)
    nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
    st = _dual_start_slack_basis(c, lo, hi, nrow)
    assert st is not None, f"{tag}: dual start rejected"
    af = sp.hstack([A, sp.identity(nrow, format="csc")], format="csc")
    out = _rust.solve_lp_warm_csc_py(
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
    basic = np.asarray(out[5], dtype=np.int64)
    assert basic.shape[0] == nrow, f"{tag}: basic_vars length mismatch"
    B = af[:, basic].tocsc()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        splu(B, permc_spec="COLAMD", diag_pivot_thresh=0.1)
        ts.append((time.perf_counter() - t0) * 1e3)
    splu_ms = min(ts)
    r = h3[tag]
    assert r["n_num"] > 0, f"{tag}: zero numeric factorizations recorded"
    feral_ms = r["num_ms"] / r["n_num"]
    print(
        f"{tag:22s} {nrow:5d} {B.nnz:7d} {feral_ms:12.2f} {splu_ms:9.2f} "
        f"{feral_ms / splu_ms:6.1f}x",
        flush=True,
    )
    print(
        "JSON4 "
        + json.dumps(
            {
                "tag": tag,
                "m": nrow,
                "nnzB": int(B.nnz),
                "feral_ms_per_fac": feral_ms,
                "splu_ms": splu_ms,
                "splu_ms_all": ts,
                "ratio": feral_ms / splu_ms,
            }
        ),
        flush=True,
    )
    compared += 1

print("compared factorizations:", compared, flush=True)
if compared == 0:
    sys.exit(1)
