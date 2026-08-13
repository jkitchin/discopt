"""#1008: isolate the RLT-on QPLIB_1157 root LP in-house solve for the profiler.

Builds the LP once, then runs ONLY the in-house solve N times so the sampler's
mass lands in the simplex. Prints an executed-solve count; exits non-zero at 0.
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import discopt
import discopt._rust as _rust
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.interfaces import qplib
from discopt.solvers.milp_simplex import _dual_start_slack_basis

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
print("LOADED discopt:", discopt.__file__, flush=True)
print("LOADED _rust:  ", _rust.__file__, flush=True)
assert discopt.__file__.startswith(WT)
assert _rust.__file__.startswith(WT)

REPS = int(os.environ.get("I1008_REPS", "3"))
C = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib")
nm = os.environ.get("I1008_INST", "QPLIB_1157")
inst = qplib.read_qplib(os.path.join(C, "qplib", f"{nm}.qplib"))
m = qplib.to_model(inst)
lb0 = np.array([float(np.min(v.lb)) for v in m._variables])
ub0 = np.array([float(np.max(v.ub)) for v in m._variables])
rel = build_uniform_relaxation(m, box=(lb0, ub0), rlt_lineq=True)
M = rel.model
c = np.asarray(M._c, float).ravel()
A = sp.csc_matrix(M._A_ub)
b = np.asarray(M._b_ub, float).ravel()
bl = np.asarray(M._bounds, float)
lo = bl[:, 0].copy()
hi = bl[:, 1].copy()
nrow, ncol = A.shape
st = _dual_start_slack_basis(c, lo, hi, nrow)
assert st is not None

c_std = np.ascontiguousarray(np.concatenate([c, np.zeros(nrow)]))
lb_std = np.ascontiguousarray(np.concatenate([lo, np.zeros(nrow)]))
ub_std = np.ascontiguousarray(np.concatenate([hi, np.full(nrow, np.inf)]))
a_full = sp.hstack([A, sp.identity(nrow, format="csc")], format="csc")
indptr = np.ascontiguousarray(a_full.indptr, dtype=np.int64)
indices = np.ascontiguousarray(a_full.indices, dtype=np.int64)
data = np.ascontiguousarray(a_full.data, dtype=np.float64)
bb = np.ascontiguousarray(b)
cs0 = np.ascontiguousarray(st[0], dtype=np.int8)
bv0 = np.ascontiguousarray(st[1], dtype=np.int64)

print(f"LP: rows={nrow} cols={ncol} std_cols={ncol + nrow} nnz={a_full.nnz}", flush=True)
fired = 0
for k in range(REPS):
    t0 = time.perf_counter()
    out = _rust.solve_lp_warm_csc_py(
        c_std, nrow, ncol + nrow, indptr, indices, data, bb, lb_std, ub_std,
        cs0, bv0, 1e-9, 100_000, None,
    )
    dt = time.perf_counter() - t0
    assert out[0] == "optimal", out[0]
    print(f"rep {k}: {dt:8.3f}s  obj={out[2]:.10g}  iters={out[3]}", flush=True)
    fired += 1
print("executed solves:", fired, flush=True)
if fired == 0:
    sys.exit(1)
