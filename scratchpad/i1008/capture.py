"""#1008: capture root-relaxation LPs to .npz so the cadence sweep re-solves the
IDENTICAL matrix in every arm (env knobs are read once per process).

Selects continuous nonconvex QPLIB instances from the manifest by FILTER, never by
hardcoded name (CLAUDE.md §2). Prints a per-instance line and a captured count;
exits non-zero when nothing was captured (§6).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import discopt
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.interfaces import qplib

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__

C = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib")
OUT = os.path.join(WT, "scratchpad/i1008/lps")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(C, "qplib_manifest.csv"))
q = df[(df.usable_oracle) & (df.n_integral == 0) & (df.nonconvex)].copy()
q["lift"] = q.obj_quad_nnz + q.con_quad_nnz
# Keep instances whose lifted relaxation is big enough for the refactorization
# cost to matter but small enough to build quickly: bounded variable count.
q = q[(q.nvars <= 400) & (q.lift >= 300)].sort_values("lift", ascending=False)
want = int(os.environ.get("I1008_N", "8"))
names = list(q.name)[:want]
# The issue's gate probe must be in the panel even though it is not in the
# largest-lift prefix (CLAUDE.md §2: named instances are gate probes only).
probe = os.environ.get("I1008_PROBE", "QPLIB_1157")
if probe and probe in set(q.name) and probe not in names:
    names.append(probe)
print("candidates:", names, flush=True)

captured = 0
for nm in names:
    for rlt in (False, True):
        tag = f"{nm}_rlt{int(rlt)}"
        path = os.path.join(OUT, tag + ".npz")
        if os.path.exists(path):
            print(f"skip {tag} (exists)", flush=True)
            captured += 1
            continue
        t0 = time.perf_counter()
        inst = qplib.read_qplib(os.path.join(C, "qplib", f"{nm}.qplib"))
        m = qplib.to_model(inst)
        lb0 = np.array([float(np.min(v.lb)) for v in m._variables])
        ub0 = np.array([float(np.max(v.ub)) for v in m._variables])
        rel = build_uniform_relaxation(m, box=(lb0, ub0), rlt_lineq=rlt)
        M = rel.model
        A = sp.csc_matrix(M._A_ub)
        bl = np.asarray(M._bounds, float)
        np.savez_compressed(
            path,
            c=np.asarray(M._c, float).ravel(),
            indptr=A.indptr,
            indices=A.indices,
            data=A.data,
            shape=np.array(A.shape),
            b=np.asarray(M._b_ub, float).ravel(),
            lo=bl[:, 0].copy(),
            hi=bl[:, 1].copy(),
        )
        print(
            f"captured {tag}: rows={A.shape[0]} cols={A.shape[1]} nnz={A.nnz} "
            f"build={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        captured += 1
print("captured LPs:", captured, flush=True)
if captured == 0:
    sys.exit(1)
