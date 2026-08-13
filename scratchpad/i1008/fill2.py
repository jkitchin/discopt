"""#1008 H2 arm: solve every captured LP at this process's LU pivot threshold,
recording fill, iterations, wall, status and the objective against HiGHS.

One process per threshold -- the knob is OnceLock-cached. Prints a measured
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
from scipy.optimize import linprog

import discopt
import discopt._rust as _rust

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__
assert _rust.__file__.startswith(WT), _rust.__file__

# #8 marker: the #1008 pivot-threshold knob exists ONLY in the version under test.
MARKER = "DISCOPT_LU_PIVOT_THRESHOLD"
strs = subprocess.run(
    f"strings {_rust.__file__}", shell=True, capture_output=True, text=True, check=True
).stdout
has = MARKER in strs
want = os.environ.get("I1008_EXPECT_MARKER", "1") == "1"
print(f"MARKER {MARKER!r} present={has} expected={want}", flush=True)
assert has == want, "loaded the wrong build"

U = os.environ.get("DISCOPT_LU_PIVOT_THRESHOLD", "1.0")
TL = float(os.environ.get("I1008_TL", "45"))
REPS = int(os.environ.get("I1008_REPS", "1"))
MAXROWS = int(os.environ.get("I1008_MAXROWS", "6000"))

paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
paths = [p for p in paths if int(np.load(p)["shape"][0]) <= MAXROWS]
assert paths, "no captured LPs"
print(f"arm: u={U} lps={len(paths)}", flush=True)

from discopt.solvers.milp_simplex import _dual_start_slack_basis

measured = 0
checked = 0
for p in paths:
    tag = os.path.basename(p)[:-4]
    z = np.load(p)
    nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]

    t0 = time.perf_counter()
    r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
    t_h = time.perf_counter() - t0
    assert r.status == 0, r.message
    obj_h = float(r.fun)

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
        TL,
    )
    _rust.profile_reset_py()
    walls = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = _rust.solve_lp_warm_csc_py(*args)
        walls.append(time.perf_counter() - t0)
    snap = dict(_rust.profile_counters_py())
    facs = max(1, snap["LuSparseFactorizations"])
    fill = snap["LuFactorNnz"] / max(1, snap["LuBasisNnz"])
    rec = {
        "tag": tag, "u": U, "m": nrow, "status": out[0], "obj": float(out[2]),
        "iters": int(out[3]), "facs": int(snap["LuSparseFactorizations"]),
        "nnzB": int(snap["LuBasisNnz"] // facs), "nnzLU": int(snap["LuFactorNnz"] // facs),
        "fill": fill, "wall": min(walls), "walls": walls,
        "highs_wall": t_h, "highs_iters": int(r.nit), "highs_obj": obj_h,
    }
    print("JSONP " + json.dumps(rec), flush=True)
    print(
        f"  {tag:20s} fill={fill:6.2f} nnzLU={rec['nnzLU']:8d} it={rec['iters']:6d} "
        f"wall={rec['wall']:8.3f} highs={t_h:6.3f} {out[0]}",
        flush=True,
    )
    if out[0] == "optimal":
        assert abs(rec["obj"] - obj_h) <= 1e-6 * max(1.0, abs(obj_h)), (
            f"{tag} u={U}: objective disagrees with HiGHS: {rec['obj']} vs {obj_h}"
        )
        checked += 1
    measured += 1

print(f"measured LPs: {measured}  objective checks executed: {checked}", flush=True)
if measured == 0 or checked == 0:
    sys.exit(1)
