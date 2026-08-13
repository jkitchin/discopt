"""#1013: write a captured degenerate-stall LP as a Rust test fixture.

Emits the flat JSON `parse_stall_fixture` already reads (m, n, col_ptr, row_idx,
vals, c, l, u, b, basic_vars, col_status) for the SAME standard form the panel
solves: `[A | I]` with the sign-matched dual-feasible slack start. Selected by
measured degenerate-run length, not by name (CLAUDE.md §2) — the caller passes
the LP; `scratchpad/i1013/runstats.jsonl` is what ranks them.

Prints the emitted sizes and exits non-zero if nothing was written (§6).
"""

import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from discopt.solvers.milp_simplex import _dual_start_slack_basis

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lp, out = sys.argv[1], sys.argv[2]
z = np.load(os.path.join(ROOT, "scratchpad/i1013/lps", lp + ".npz"))
nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
c, b, lo, hi = z["c"], z["b"], z["lo"], z["hi"]
st = _dual_start_slack_basis(c, lo, hi, nrow)
assert st is not None, "no dual-feasible slack start"
af = sp.csc_matrix(sp.hstack([A, sp.identity(nrow, format="csc")], format="csc"))
INF = 1e20
payload = {
    "m": nrow,
    "n": ncol + nrow,
    "col_ptr": [int(v) for v in af.indptr],
    "row_idx": [int(v) for v in af.indices],
    "vals": [float(v) for v in af.data],
    "c": [float(v) for v in np.concatenate([c, np.zeros(nrow)])],
    "l": [float(min(v, INF)) for v in np.concatenate([lo, np.zeros(nrow)])],
    "u": [float(min(v, INF)) for v in np.concatenate([hi, np.full(nrow, INF)])],
    "b": [float(v) for v in b],
    "basic_vars": [int(v) for v in st[1]],
    "col_status": [int(v) for v in st[0]],
}
with open(out, "w") as fh:
    json.dump(payload, fh)
print(f"wrote {out}: m={nrow} n={ncol + nrow} nnz={af.nnz} bytes={os.path.getsize(out)}")
if af.nnz == 0:
    sys.exit(1)
