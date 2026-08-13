"""#1013: settle the `infeasible` verdict EXACTLY, in rational arithmetic.

The engine certifies infeasibility with a floating-point Neumaier-Shcherbina test
(`farkas_ray_certifies_cols`) plus floating-point *implied* upper bounds for the
slack columns (`slack_upper_bounds`). Both are the instrument AND the thing under
test, so this re-evaluates the returned ray over the same standard form
`[A | I][x; s] = b, l <= x <= u, s >= 0` with `fractions.Fraction` — no rounding
at all. Float inputs convert exactly, so the verdict here is the truth:

    g0(y) = b'y - max_{box} (A'y)'z   >  0   <=>   the LP is EXACTLY infeasible.

Prints the exact g0 for both signs of the ray, the float g0 the engine computed,
and how many terms were evaluated; exits non-zero if nothing was evaluated (§6).
"""

import json
import os
import subprocess
import sys
from fractions import Fraction as F

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lp = sys.argv[1]
arm = sys.argv[2] if len(sys.argv) > 2 else "0"
path = os.path.join(ROOT, "scratchpad/i1013/lps", lp + ".npz")

CHILD = r"""
import json, sys
import numpy as np, scipy.sparse as sp
import discopt._rust as _rust
from discopt.solvers.milp_simplex import _dual_start_slack_basis
z = np.load(sys.argv[1])
nrow, ncol = int(z['shape'][0]), int(z['shape'][1])
A = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=(nrow, ncol))
c, b, lo, hi = z['c'], z['b'], z['lo'], z['hi']
st = _dual_start_slack_basis(c, lo, hi, nrow)
af = sp.hstack([A, sp.identity(nrow, format='csc')], format='csc')
out = _rust.solve_lp_warm_csc_py(
    np.ascontiguousarray(np.concatenate([c, np.zeros(nrow)])), nrow, ncol + nrow,
    np.ascontiguousarray(af.indptr, dtype=np.int64),
    np.ascontiguousarray(af.indices, dtype=np.int64),
    np.ascontiguousarray(af.data, dtype=np.float64),
    np.ascontiguousarray(b),
    np.ascontiguousarray(np.concatenate([lo, np.zeros(nrow)])),
    np.ascontiguousarray(np.concatenate([hi, np.full(nrow, np.inf)])),
    np.ascontiguousarray(st[0], dtype=np.int8),
    np.ascontiguousarray(st[1], dtype=np.int64), 1e-9, 100000, 120.0)
print('OUT ' + json.dumps({'status': out[0], 'ray': np.asarray(out[6], float).tolist()}))
"""

e = dict(os.environ, DISCOPT_LP_DUAL_STALL_HARRIS=arm, DISCOPT_PROFILE="1")
p = subprocess.run([sys.executable, "-u", "-c", CHILD, path], capture_output=True, text=True, env=e)
assert p.returncode == 0, p.stderr[-2000:]
rec = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("OUT ")][0][4:])
print(f"{lp} arm={arm}: status={rec['status']} ray_len={len(rec['ray'])}")
if rec["status"] != "infeasible":
    print("no infeasibility certificate to check")
    sys.exit(0)

z = np.load(path)
nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol)).tocsc()
b, lo, hi = z["b"], z["lo"], z["hi"]
ray = np.asarray(rec["ray"], float)
assert ray.size == nrow, (ray.size, nrow)

# Exact standard form: columns 0..ncol-1 are structural with [lo, hi]; columns
# ncol.. are the slacks with [0, +inf).
bF = [F(v) for v in b]
loF = [F(v) for v in lo]
hiF = [F(v) for v in hi]

terms = 0
for sgn in (1, -1):
    y = [F(sgn) * F(v) for v in ray]
    by = sum(bi * yi for bi, yi in zip(bF, y))
    # Structural columns: exact box-max of (A_j . y) * x_j.
    contrib = F(0)
    open_col = None
    for j in range(ncol):
        s, e_ = A.indptr[j], A.indptr[j + 1]
        aty = sum(F(float(v)) * y[int(i)] for i, v in zip(A.indices[s:e_], A.data[s:e_]))
        terms += 1
        if aty > 0:
            contrib += aty * hiF[j]
        elif aty < 0:
            contrib += aty * loF[j]
    # Slack columns: identity, so (A_j . y) = y_j and s_j in [0, +inf). A positive
    # multiplier makes the box-max +inf UNLESS the slack has an implied finite
    # upper bound; that recovery is exactly what the engine does in floating point,
    # so recompute it here in exact arithmetic from the row it defines:
    #   s_i = b_i - sum_k A_ik x_k  =>  max s_i = b_i - min_box sum_k A_ik x_k.
    row_min = [F(0)] * nrow
    row_open = [0] * nrow
    Acoo = A.tocoo()
    for i, j, v in zip(Acoo.row, Acoo.col, Acoo.data):
        a = F(float(v))
        if a > 0:
            row_min[int(i)] += a * loF[int(j)]
        else:
            row_min[int(i)] += a * hiF[int(j)]
        terms += 1
    for i in range(nrow):
        yi = y[i]
        if yi > 0:
            if row_open[i]:
                open_col = i
                break
            contrib += yi * (bF[i] - row_min[i])  # exact implied upper bound
        # yi <= 0 contributes yi * 0 = 0 at the slack's lower bound
    if open_col is not None:
        print(f"  sign {sgn:+d}: ray selects an open slack column {open_col} -> no certificate")
        continue
    g0 = by - contrib
    print(
        f"  sign {sgn:+d}: exact g0 = {float(g0):.6e}  ({'INFEASIBLE proven' if g0 > 0 else 'certificate INVALID'})"
    )

print("evaluated terms:", terms)
if terms == 0:
    sys.exit(1)
