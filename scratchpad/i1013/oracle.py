"""#1013: independent verdict on a captured LP from SciPy's HiGHS.

Used to settle a disagreement between two pivot paths of our own engine: an
in-engine certificate cannot arbitrate itself (CLAUDE.md §7 in spirit — the
instrument must not be the thing under test). Prints the oracle status/objective
and the residuals of its point; exits non-zero if the solve did not run.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for lp in sys.argv[1:]:
    z = np.load(os.path.join(ROOT, "scratchpad/i1013/lps", lp + ".npz"))
    nrow, ncol = int(z["shape"][0]), int(z["shape"][1])
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(nrow, ncol))
    b, lo, hi, c = z["b"], z["lo"], z["hi"], z["c"]
    r = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
    res = None
    if r.x is not None:
        res = float(np.max(A @ r.x - b))
    print(
        f"{lp}: highs status={r.status} ({r.message.split('.')[0]}) obj={r.fun!r} max(Ax-b)={res!r}"
    )
print("oracle solves:", len(sys.argv) - 1)
if len(sys.argv) == 1:
    sys.exit(1)
