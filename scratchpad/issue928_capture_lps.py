"""Issue #928 entry experiment, step 1: capture the real node LPs of an hda solve.

Wraps ``solve_lp_warm_std`` to record every pure-LP call's inputs and wall time,
then pickles the slowest ones for offline replay (step 2 measures the deadline-exit
banked floor vs the true LP optimum on exactly these LPs).

Probe discipline (CLAUDE.md §6): prints an executed-call count and exits non-zero
when nothing was captured.

Usage: python issue928_capture_lps.py <instance.nl> <time_limit> <out.pkl>
"""

import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import discopt  # noqa: E402
import discopt.solvers.milp_simplex as MS  # noqa: E402
from discopt._relax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

nl_path, budget, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]

captured = []
orig = MS.solve_lp_warm_std


def spy(c, A_ub, b_ub, bounds, in_basis=None, *, return_cert=False, time_limit=None):
    t0 = time.perf_counter()
    out = orig(c, A_ub, b_ub, bounds, in_basis, return_cert=return_cert, time_limit=time_limit)
    wall = time.perf_counter() - t0
    rec = {
        "wall": wall,
        "time_limit": time_limit,
        "n": int(np.asarray(c).ravel().shape[0]),
        "m": 0 if A_ub is None else sp.csr_matrix(A_ub).shape[0],
        "c": np.asarray(c, dtype=np.float64).copy(),
        "A_ub": None if A_ub is None else sp.csr_matrix(A_ub).copy(),
        "b_ub": None if b_ub is None else np.asarray(b_ub, dtype=np.float64).copy(),
        "bounds": None if bounds is None else list(bounds),
        "had_basis": in_basis is not None,
        "in_basis": None
        if in_basis is None
        else (np.asarray(in_basis[0]).copy(), np.asarray(in_basis[1]).copy()),
    }
    captured.append(rec)
    return out


MS.solve_lp_warm_std = spy
# The relaxation layer imports the symbol lazily inside methods (from ... import),
# so patching the module attribute is enough — verify that below via the counter.

m = from_nl(nl_path)
t0 = time.perf_counter()
with deadline_scope(budget):
    r = solve_model(m, time_limit=budget)
wall = time.perf_counter() - t0
print(f"solve: wall={wall:.2f}s bound={r.bound} status={r.status} lp_calls={len(captured)}")

if not captured:
    print("PROBE FIRED NOTHING: solve_lp_warm_std never called", file=sys.stderr)
    sys.exit(1)

# Keep the slowest 12 LPs (the ones a deadline would actually bind on).
captured.sort(key=lambda rec: -rec["wall"])
keep = captured[:12]
with open(out_path, "wb") as fh:
    pickle.dump({"instance": nl_path, "budget": budget, "lps": keep}, fh)
walls = ", ".join(f"{rec['wall']:.3f}" for rec in keep)
print(f"captured {len(captured)} LP calls; kept slowest {len(keep)}: walls [{walls}] s")
print(f"wrote {out_path}")
