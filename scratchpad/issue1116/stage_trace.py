"""#1116 attribution: WHERE does the run-to-run divergence first appear?

Usage: python -u stage_trace.py <instance-stem> <max_nodes> <reps>

Traces, per repetition and without changing the solve route (no callbacks — those
decline the native kernel, CLAUDE.md §8):

  * a hash of the parsed model (does `from_nl` itself vary?),
  * the root relaxation LP bound built directly from the parsed model,
  * the solve's reported root_bound / bound / objective / node_count.

Every quantity is compared across reps and the number of comparisons executed is
printed; the probe exits non-zero if it compared nothing (§6).
"""

import hashlib
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.modeling.core import from_nl

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def _h(*arrays) -> str:
    m = hashlib.sha1()
    for a in arrays:
        b = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        m.update(b.tobytes())
    return m.hexdigest()[:12]


rows = []
for rep in range(reps):
    t0 = time.perf_counter()
    model = from_nl(NL.format(stem))
    bounds = np.asarray(model._bounds if hasattr(model, "_bounds") else [], dtype=float)
    rel = build_uniform_relaxation(model)
    M = rel.model
    A = sp.csr_matrix(M._A_ub, dtype=float)
    A.sort_indices()
    relax_hash = _h(A.data, A.indices.astype(float), A.indptr.astype(float), M._b_ub, M._c)
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    r = from_nl(NL.format(stem)).solve(max_nodes=max_nodes)
    row = {
        "rep": rep,
        "model_bounds_hash": _h(bounds) if bounds.size else "n/a",
        "relax_hash": relax_hash,
        "relax_rows": int(A.shape[0]),
        "root_bound": None if r.root_bound is None else float(r.root_bound),
        "bound": None if r.bound is None else float(r.bound),
        "objective": None if r.objective is None else float(r.objective),
        "nodes": int(r.node_count or 0),
        "status": r.status,
        "build_wall": round(t_build, 2),
        "solve_wall": round(time.perf_counter() - t1, 2),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
for key in rows[0]:
    if key in ("rep", "build_wall", "solve_wall"):
        continue
    vals = sorted({repr(r[key]) for r in rows})
    comparisons += len(rows) - 1
    print(f"{key:20s} {'STABLE' if len(vals) == 1 else 'VARIES'}  {vals}", flush=True)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    sys.exit(2)
