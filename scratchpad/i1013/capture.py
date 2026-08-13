"""#1013: capture root-relaxation LPs from the IN-REPO corpus to .npz.

The QPLIB/MINLPLib snapshot under ~/Dropbox is not present in this environment,
so the panel is drawn from the vendored corpora (`python/tests/data/qplib`,
`python/tests/data/minlplib_nl`) by filter, never by hardcoded name (CLAUDE.md §2).

Prints one line per capture and a captured count; exits non-zero at zero (§6).
"""

import os
import sys
import time

import discopt
import numpy as np
import scipy.sparse as sp
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.interfaces import qplib
from discopt.modeling.core import from_nl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
assert discopt.__file__.startswith(ROOT), discopt.__file__
OUT = os.path.join(ROOT, "scratchpad/i1013/lps")
os.makedirs(OUT, exist_ok=True)

QDIR = os.path.join(ROOT, "python/tests/data/qplib/qplib")
NLDIR = os.path.join(ROOT, "python/tests/data/minlplib_nl")

MAX_BUILD_S = float(os.environ.get("I1013_MAX_BUILD_S", "60"))


def save(tag, M):
    A = sp.csc_matrix(M._A_ub)
    bl = np.asarray(M._bounds, float)
    path = os.path.join(OUT, tag + ".npz")
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
    return A.shape, A.nnz


def box_of(m):
    lb = np.array([float(np.min(v.lb)) for v in m._variables])
    ub = np.array([float(np.max(v.ub)) for v in m._variables])
    return lb, ub


def capture(tag, model, rlt):
    path = os.path.join(OUT, tag + ".npz")
    if os.path.exists(path):
        print(f"skip {tag} (exists)", flush=True)
        return 1
    t0 = time.perf_counter()
    lb, ub = box_of(model)
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        print(f"skip {tag}: unbounded box", flush=True)
        return 0
    rel = build_uniform_relaxation(model, box=(lb, ub), rlt_lineq=rlt)
    shape, nnz = save(tag, rel.model)
    print(
        f"captured {tag}: rows={shape[0]} cols={shape[1]} nnz={nnz} "
        f"build={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return 1


captured = 0
sources = []
for f in sorted(os.listdir(QDIR)):
    if f.endswith(".qplib"):
        sources.append(("qplib", f[:-6], os.path.join(QDIR, f)))
for f in sorted(os.listdir(NLDIR)):
    if f.endswith(".nl"):
        sources.append(("nl", f[:-3], os.path.join(NLDIR, f)))
print(f"sources: {len(sources)}", flush=True)

for kind, name, path in sources:
    t0 = time.perf_counter()
    if kind == "qplib":
        model = qplib.to_model(qplib.read_qplib(path))
    else:
        model = from_nl(path)
    load_s = time.perf_counter() - t0
    if load_s > MAX_BUILD_S:
        print(f"skip {name}: load {load_s:.1f}s", flush=True)
        continue
    for rlt in (False, True):
        captured += capture(f"{name}_rlt{int(rlt)}", model, rlt)

print("captured LPs:", captured, flush=True)
if captured == 0:
    sys.exit(1)
