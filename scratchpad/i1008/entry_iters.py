"""#1008 entry experiment: is the 29x MORE iterations or MORE cost per iteration?

Builds the QPLIB_1157 root LP (RLT off and on), hands the IDENTICAL matrix to
HiGHS and to the in-house simplex, and reports wall AND iteration count for
both. A 29x that is 29x iterations at equal per-iteration cost and a 29x that
is equal iterations at 29x per-iteration cost need completely different fixes.

Prints an executed-comparison count and exits non-zero when it is zero (§6).
No exception is swallowed (§7): probes crash.
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import discopt
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.interfaces import qplib
from discopt.solvers.milp_simplex import _dual_start_slack_basis

# §8: prove which code is loaded.
print("LOADED discopt:", discopt.__file__, flush=True)
import discopt._rust as _rust  # noqa: E402

print("LOADED _rust:  ", _rust.__file__, flush=True)
WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), "not the worktree python!"
assert _rust.__file__.startswith(WT), "not the worktree extension!"

C = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib")
nm = sys.argv[1] if len(sys.argv) > 1 else "QPLIB_1157"
inst = qplib.read_qplib(os.path.join(C, "qplib", f"{nm}.qplib"))
m = qplib.to_model(inst)
lb0 = np.array([float(np.min(v.lb)) for v in m._variables])
ub0 = np.array([float(np.max(v.ub)) for v in m._variables])

fired = 0
for rlt in (False, True):
    rel = build_uniform_relaxation(m, box=(lb0, ub0), rlt_lineq=rlt)
    M = rel.model
    c = np.asarray(M._c, float).ravel()
    A = sp.csr_matrix(M._A_ub)
    b = np.asarray(M._b_ub, float).ravel()
    bl = [(float(lo), float(hi)) for lo, hi in np.asarray(M._bounds, float)]
    nrow, ncol = A.shape

    t0 = time.perf_counter()
    r = linprog(c, A_ub=A, b_ub=b, bounds=bl, method="highs")
    t_h = time.perf_counter() - t0
    assert r.status == 0, r.message
    nit_h = int(r.nit)

    lo = np.array([x for x, _ in bl])
    hi = np.array([y for _, y in bl])
    st = _dual_start_slack_basis(c, lo, hi, nrow)
    assert st is not None, "dual start basis rejected"

    # Call the binding directly so we see the iteration count.
    a_std = sp.csc_matrix(A)
    nz = a_std.nnz
    c_std = np.concatenate([c, np.zeros(nrow)])
    lb_std = np.concatenate([lo, np.zeros(nrow)])
    ub_std = np.concatenate([hi, np.full(nrow, np.inf)])
    # slack identity appended as explicit columns (mirrors solve_lp_warm_std)
    eye = sp.identity(nrow, format="csc")
    a_full = sp.hstack([a_std, eye], format="csc")

    t0 = time.perf_counter()
    out = _rust.solve_lp_warm_csc_py(
        np.ascontiguousarray(c_std),
        nrow,
        ncol + nrow,
        np.ascontiguousarray(a_full.indptr, dtype=np.int64),
        np.ascontiguousarray(a_full.indices, dtype=np.int64),
        np.ascontiguousarray(a_full.data, dtype=np.float64),
        np.ascontiguousarray(b),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        np.ascontiguousarray(st[0], dtype=np.int8),
        np.ascontiguousarray(st[1], dtype=np.int64),
        1e-9,
        100_000,
        None,
    )
    t_s = time.perf_counter() - t0
    status, x_full, obj, iters = out[0], out[1], out[2], out[3]
    assert status == "optimal", status
    assert abs(obj - r.fun) < 1e-6 * max(1.0, abs(r.fun)), (obj, r.fun)

    print(
        f"rlt={int(rlt)} rows={nrow:5d} cols={ncol:5d} nnz={nz:7d}  "
        f"highs={t_h:7.3f}s/{nit_h:6d}it ({1e6*t_h/max(nit_h,1):8.1f} us/it)  "
        f"inhouse={t_s:8.3f}s/{iters:6d}it ({1e6*t_s/max(iters,1):8.1f} us/it)  "
        f"wall_ratio={t_s/max(t_h,1e-9):6.1f}x  iter_ratio={iters/max(nit_h,1):6.1f}x  "
        f"usperit_ratio={(t_s/max(iters,1))/(t_h/max(nit_h,1)):6.1f}x",
        flush=True,
    )
    fired += 1

print("executed comparisons:", fired, flush=True)
if fired == 0:
    sys.exit(1)
