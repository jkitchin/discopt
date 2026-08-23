"""#1116 stage 1: is the BUILT relaxation itself bit-identical across reps?

Usage: python -u build_hash_probe.py <instance-stem> <reps>

Builds the root uniform relaxation N times in ONE process and hashes the LP data
(A_ub structure+values, b_ub, c, bounds, integrality) plus the varmap orderings.
No exception is caught (CLAUDE.md §7); the number of executed comparisons is
printed and a zero count exits non-zero (§6).
"""

import hashlib
import json
import sys
import time

import discopt
import numpy as np
import scipy.sparse as sp
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, reps = sys.argv[1], int(sys.argv[2])


def _h(*arrays) -> str:
    m = hashlib.sha1()
    for a in arrays:
        m.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return m.hexdigest()[:12]


def _hs(obj) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]


rows = []
for rep in range(reps):
    t0 = time.perf_counter()
    model = from_nl(NL.format(stem))
    rel = build_uniform_relaxation(model)
    M = rel.model
    A = sp.csr_matrix(M._A_ub)
    row = {
        "rep": rep,
        # RAW (unsorted) CSR triple: catches row/column ORDER changes, not just values
        "A_raw": _h(A.data, A.indices, A.indptr),
        "b": _h(M._b_ub),
        "c": _h(M._c),
        "bounds": _hs(M._bounds),
        "integrality": _hs(list(M._integrality) if M._integrality is not None else None),
        "shape": list(A.shape),
        "nnz": int(A.nnz),
        "bilinear_map": _hs(list(rel.bilinear_map.items())),
        "monomial_map": _hs(list(rel.monomial_map.items())),
        "multilinear_map": _hs(list(rel.multilinear_map.items())),
        "univariate_square_map": _hs(list(rel.univariate_square_map.items())),
        "composite_specs_n": len(rel.composite_multivar_specs),
        "univariate_atom_specs": _hs(rel.univariate_atom_specs),
        "bilinear_linform_specs": _hs(rel.bilinear_linform_specs),
        "coverage_kinds": _hs(sorted(rel.coverage.values())),
        "wall": round(time.perf_counter() - t0, 2),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
for key in rows[0]:
    if key in ("rep", "wall"):
        continue
    vals = sorted({repr(r[key]) for r in rows})
    comparisons += len(rows) - 1
    print(f"{key:24s} {'STABLE' if len(vals) == 1 else 'VARIES'}  {vals}", flush=True)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    sys.exit(2)
