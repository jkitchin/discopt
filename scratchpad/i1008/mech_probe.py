"""Which mechanism actually loses the bound on QPLIB_2170?

Two candidate defects sit on the same warm-dual path and the A/B says they compose:

  R1  the unstable-pivot recovery (`dual.rs`) is gated on `bank_deadline_duals`,
      which `lp_bindings.rs` sets to `deadline.is_some()`. A caller that passes
      no `time_limit` silently loses the recovery.
  R2  #1013's degeneracy-stall bail abandons the warm solve for the cold one after
      `dual_stall_patience` consecutive degenerate pivots.

The A/B (bail off => deadline arm solves; bail on => nothing solves) is consistent
with "the bail fires first and takes the recovery's chance away", but consistent-with
is not measured. This reads `DualDegenerateStallBails` directly across the 2x2 so the
claim rests on a counter rather than on a timing shape.

Every cell asserts the binding fired (§6): a cell that silently ran no solve would
otherwise report `bails=0` and read as "the bail is innocent".
"""

import hashlib
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import discopt

ROOT = os.environ["ARM_ROOT"]
assert os.environ.get("DISCOPT_PROFILE"), "DISCOPT_PROFILE must be set or every counter reads 0"
assert discopt.__file__.startswith(ROOT), f"wrong discopt: {discopt.__file__}"
import discopt._rust as _rust  # noqa: E402

print(f"# so md5={hashlib.md5(open(_rust.__file__, 'rb').read()).hexdigest()}")
print(f"# DISCOPT_LP_DUAL_STALL_BAIL={os.environ.get('DISCOPT_LP_DUAL_STALL_BAIL', '<unset>')!r}")

from discopt._relax.uniform_relax import build_uniform_relaxation  # noqa: E402
from discopt.interfaces import qplib  # noqa: E402
import discopt.solvers.milp_simplex as ms  # noqa: E402

C = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib")
n_cells = 0

for nm in sys.argv[1:]:
    inst = qplib.read_qplib(os.path.join(C, "qplib", f"{nm}.qplib"))
    m = qplib.to_model(inst)
    lb0 = np.array([float(np.min(v.lb)) for v in m._variables])
    ub0 = np.array([float(np.max(v.ub)) for v in m._variables])
    M = build_uniform_relaxation(m, box=(lb0, ub0), rlt_lineq=True).model
    c = np.asarray(M._c, float).ravel()
    A = sp.csr_matrix(M._A_ub)
    b = np.asarray(M._b_ub, float).ravel()
    bl = [(float(lo), float(hi)) for lo, hi in np.asarray(M._bounds, float)]
    lo = np.array([x for x, _ in bl])
    hi = np.array([y for _, y in bl])
    st = ms._dual_start_slack_basis(c, lo, hi, A.shape[0])

    r = linprog(c, A_ub=A, b_ub=b, bounds=bl, method="highs")
    assert r.status == 0, f"{nm}: HiGHS status {r.status}"
    print(f"\n{nm}: rows={A.shape[0]} nnz={A.nnz}  HiGHS obj={float(r.fun):.9g} nit={r.nit}")

    for label, tl in (("time_limit=None", None), ("time_limit=40.0", 40.0)):
        _rust.profile_reset_py()
        t0 = time.perf_counter()
        res, _ = ms.solve_lp_warm_std(c, A, b, bl, in_basis=st, time_limit=tl)
        dt = time.perf_counter() - t0
        ctr = _rust.profile_counters_py()
        n_cells += 1
        if res is None or res.objective is None:
            out = "NO-SOLUTION"
        else:
            ok = abs(res.objective - float(r.fun)) <= 1e-6 * max(1.0, abs(float(r.fun)))
            out = f"{'ok' if ok else 'WRONG'} {res.objective:.9g}"
        keys = ("DualDegenerateStallBails", "DualStallTrips", "DualPivots", "DualIters")
        shown = {k: ctr[k] for k in keys if k in ctr}
        print(f"  {label:<16} {out:<14} ({dt:.2f}s)  {shown}")

print(f"\nexecuted: cells={n_cells}")
assert n_cells, "probe ran no cells"
