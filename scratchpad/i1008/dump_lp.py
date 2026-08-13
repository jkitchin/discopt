"""Dump a QPLIB root relaxation LP into the `.npz` form `make_fixture.py` reads.

The #1013 fixture pipeline starts from `scratchpad/i1013/lps/<name>.npz`; that
directory holds the LPs captured during that issue's panel and does not include a
case where the *cold* solve fails. This writes one, so the bail's bound-neutrality
claim can be tested against an LP that falsifies it rather than one that confirms it.

Selection is by measured outcome (the caller passes an LP whose cold fallback was
observed to fail), not by name -- CLAUDE.md §2.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

import discopt

ROOT = os.environ["ARM_ROOT"]
assert discopt.__file__.startswith(ROOT), f"wrong discopt: {discopt.__file__}"

from discopt._relax.uniform_relax import build_uniform_relaxation  # noqa: E402
from discopt.interfaces import qplib  # noqa: E402

C = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/qplib")
nm, out = sys.argv[1], sys.argv[2]

inst = qplib.read_qplib(os.path.join(C, "qplib", f"{nm}.qplib"))
m = qplib.to_model(inst)
lb0 = np.array([float(np.min(v.lb)) for v in m._variables])
ub0 = np.array([float(np.max(v.ub)) for v in m._variables])
M = build_uniform_relaxation(m, box=(lb0, ub0), rlt_lineq=True).model
c = np.asarray(M._c, float).ravel()
A = sp.csc_matrix(M._A_ub)
b = np.asarray(M._b_ub, float).ravel()
bnd = np.asarray(M._bounds, float)
lo, hi = bnd[:, 0].copy(), bnd[:, 1].copy()

# The fixture is only useful if the LP's true status is known independently, and
# the test's whole point is that our engine gets it wrong -- so the oracle must
# come from outside our engine.
r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
assert r.status == 0, f"{nm}: HiGHS status {r.status} -- no usable oracle"
print(f"{nm}: {A.shape[0]}x{A.shape[1]} nnz={A.nnz}  HiGHS optimal {float(r.fun):.10g} "
      f"in {r.nit} pivots")

os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(
    out, shape=np.array(A.shape), data=A.data, indices=A.indices, indptr=A.indptr,
    c=c, b=b, lo=lo, hi=hi,
)
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
assert os.path.getsize(out) > 0
