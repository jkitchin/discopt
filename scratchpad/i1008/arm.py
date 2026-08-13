"""#1008 sweep arm: solve every captured LP with the in-house dual simplex at the
refactorization interval this process's env selects, and (optionally) with HiGHS.

One process per interval — the knob is read once per process via OnceLock.
Emits one JSON line per LP. Prints a solved count; exits non-zero at 0 (§6).
Nothing is caught: a probe that breaks must crash (§7).
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

# §8 marker: the #1008 cadence policy symbol exists ONLY in the version under test.
MARKER = "discopt_core::lp::simplex::refac::"
nm_out = subprocess.run(
    f"nm {_rust.__file__} | rustfilt", shell=True, capture_output=True, text=True, check=True
).stdout
has_marker = MARKER in nm_out
want_marker = os.environ.get("I1008_EXPECT_MARKER", "1") == "1"
print(f"MARKER {MARKER!r} present={has_marker} expected={want_marker}", flush=True)
assert has_marker == want_marker, "loaded the wrong build"

INTERVAL = os.environ.get("DISCOPT_LP_REFACTOR_INTERVAL", "default")
REPS = int(os.environ.get("I1008_REPS", "1"))
DO_HIGHS = os.environ.get("I1008_HIGHS", "0") == "1"
ONLY = os.environ.get("I1008_ONLY", "")

TL = os.environ.get("I1008_TL", "")
TIME_LIMIT = float(TL) if TL else None
MAXROWS = int(os.environ.get("I1008_MAXROWS", "0"))

paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
if ONLY:
    keep = set(ONLY.split(","))
    paths = [p for p in paths if os.path.basename(p)[:-4] in keep]
if MAXROWS:
    paths = [p for p in paths if int(np.load(p)["shape"][0]) <= MAXROWS]
assert paths, "no captured LPs"
print(f"arm: interval={INTERVAL} lps={len(paths)} time_limit={TIME_LIMIT}", flush=True)

solved = 0
for p in paths:
    tag = os.path.basename(p)[:-4]
    z = np.load(p)
    nrow, ncol = (int(z["shape"][0]), int(z["shape"][1]))
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]

    t_h, nit_h, obj_h = None, None, None
    if DO_HIGHS:
        from scipy.optimize import linprog

        t0 = time.perf_counter()
        r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
        t_h = time.perf_counter() - t0
        assert r.status == 0, r.message
        nit_h, obj_h = int(r.nit), float(r.fun)

    from discopt.solvers.milp_simplex import _dual_start_slack_basis

    st = _dual_start_slack_basis(c, lo, hi, nrow)
    assert st is not None, f"{tag}: dual start rejected"
    c_std = np.ascontiguousarray(np.concatenate([c, np.zeros(nrow)]))
    lb_std = np.ascontiguousarray(np.concatenate([lo, np.zeros(nrow)]))
    ub_std = np.ascontiguousarray(np.concatenate([hi, np.full(nrow, np.inf)]))
    af = sp.hstack([A, sp.identity(nrow, format="csc")], format="csc")
    args = (
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
        100_000,
        TIME_LIMIT,
    )
    walls = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = _rust.solve_lp_warm_csc_py(*args)
        walls.append(time.perf_counter() - t0)
    rec = {
        "tag": tag,
        "interval": INTERVAL,
        "rows": nrow,
        "cols": ncol,
        "nnz": int(A.nnz),
        "status": out[0],
        "obj": float(out[2]),
        "iters": int(out[3]),
        "wall": min(walls),
        "walls": walls,
        "highs_wall": t_h,
        "highs_iters": nit_h,
        "highs_obj": obj_h,
    }
    print("JSON " + json.dumps(rec), flush=True)
    if obj_h is not None and out[0] == "optimal":
        assert abs(rec["obj"] - obj_h) <= 1e-6 * max(1.0, abs(obj_h)), (
            f"{tag}: objective disagrees with HiGHS: {rec['obj']} vs {obj_h}"
        )
    solved += 1

print("solved LPs:", solved, flush=True)
if solved == 0:
    sys.exit(1)
