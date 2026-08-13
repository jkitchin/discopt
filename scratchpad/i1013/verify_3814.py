"""#1013: is the `infeasible` verdict on QPLIB_3814_rlt1 a FALSE infeasible?

The two pivot-path arms disagree on a terminal verdict (base: `infeasible`,
stability pass: `optimal`). At most one can be right. This verifies the
`optimal` arm's point against the LP data directly: if `A x <= b` and the bounds
hold, the LP is feasible and the `infeasible` certificate is false — a
correctness defect, not a performance one (CLAUDE.md §1).

Prints the number of checked rows/bounds and exits non-zero if nothing was checked.
"""

import json
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lp = sys.argv[1] if len(sys.argv) > 1 else "QPLIB_3814_rlt1"
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
x = np.asarray(out[1], float)[:ncol]
print('OUT ' + json.dumps({'status': out[0], 'obj': out[2], 'x': x.tolist()}))
"""

z = np.load(path)
nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
b, lo, hi, c = z["b"], z["lo"], z["hi"], z["c"]

checked = 0
for arm in ("0", "1"):
    e = dict(os.environ, DISCOPT_LP_DUAL_STALL_HARRIS=arm, DISCOPT_PROFILE="1")
    p = subprocess.run(
        [sys.executable, "-u", "-c", CHILD, path], capture_output=True, text=True, env=e
    )
    assert p.returncode == 0, p.stderr[-2000:]
    rec = json.loads([ln for ln in p.stdout.splitlines() if ln.startswith("OUT ")][0][4:])
    x = np.asarray(rec["x"], float)
    print(f"arm harris={arm}: status={rec['status']} obj={rec['obj']!r}")
    if rec["status"] != "optimal":
        continue
    resid = A @ x - b
    viol_rows = float(np.max(resid)) if resid.size else 0.0
    viol_lo = float(np.max(lo - x))
    viol_hi = float(np.max(x - hi))
    obj = float(c @ x)
    print(
        f"   rows checked={resid.size} max(Ax-b)={viol_rows:.3e}  "
        f"bounds checked={2 * x.size} max(lo-x)={viol_lo:.3e} max(x-hi)={viol_hi:.3e}  "
        f"cᵀx={obj!r}"
    )
    checked += resid.size + 2 * x.size
    if max(viol_rows, viol_lo, viol_hi) <= 1e-6:
        print("   => POINT IS FEASIBLE: the LP is feasible, so an `infeasible` verdict is FALSE.")

print("checked constraints:", checked)
if checked == 0:
    sys.exit(1)
