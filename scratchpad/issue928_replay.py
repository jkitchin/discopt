"""Issue #928 entry experiment, step 2: replay captured node LPs under a binding
deadline and measure the banked NS floor against the true LP optimum.

For each captured LP:
  * unlimited solve  -> p* (reference optimum) and full wall
  * limited solve at fractions of the full wall -> status + cert.safe_bound
    (what `_stash_deadline_bound` would bank today)

Prints a comparison count (CLAUDE.md §6) and exits non-zero if zero comparisons ran.

Usage: python issue928_replay.py <lps.pkl> [frac ...]
"""

import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt.solvers.milp_simplex import solve_lp_warm_std  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

pkl = sys.argv[1]
fracs = [float(f) for f in sys.argv[2:]] or [0.25, 0.5]

with open(pkl, "rb") as fh:
    data = pickle.load(fh)

comparisons = 0
for i, rec in enumerate(data["lps"]):
    args = (rec["c"], rec["A_ub"], rec["b_ub"], rec["bounds"])
    in_basis = tuple(rec["in_basis"]) if rec["in_basis"] is not None else None

    t0 = time.perf_counter()
    res, _, cert = solve_lp_warm_std(*args, in_basis, return_cert=True, time_limit=None)
    full_wall = time.perf_counter() - t0
    if res is None:
        print(f"LP{i}: unlimited solve did not converge (cert.safe_bound={cert.safe_bound})")
        continue
    p_star = res.objective
    print(
        f"LP{i}: m={rec['m']} n={rec['n']} warm={rec['had_basis']} "
        f"full_wall={full_wall:.3f}s p*={p_star:.6g} status={res.status}"
    )

    for frac in fracs:
        tl = full_wall * frac
        t0 = time.perf_counter()
        res_l, _, cert_l = solve_lp_warm_std(*args, in_basis, return_cert=True, time_limit=tl)
        wall_l = time.perf_counter() - t0
        floor = cert_l.safe_bound
        status = "None(yield)" if res_l is None else res_l.status
        gap = None if (floor is None or p_star is None) else p_star - floor
        print(
            f"  tl={tl:.3f}s ({frac:.0%}) wall={wall_l:.3f}s status={status} "
            f"banked_floor={floor} gap_to_p*={gap}"
        )
        comparisons += 1

print(f"comparisons_executed={comparisons}")
if comparisons == 0:
    sys.exit(1)
