"""Issue #940: what constraint tolerance is ATTAINABLE as a function of data scale?

A fixed absolute 1e-9 request is below double-precision noise once the data
magnitude reaches ~1e6: POUNCE then stops short of its criterion and exits with
Ipopt code 3/4, which the LP status map turns into a FALSE ``UNBOUNDED`` on a
bounded LP. Sweep the request against the exact-simplex oracle and find, per data
scale, the tightest request that still reproduces the oracle's status/objective.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.solvers.lp_pounce as LPP  # noqa: E402
import discopt.solvers.lp_simplex as LPS  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402

CHECKS = 0
REQUESTS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]


def make(n, m, scale, seed):
    rng = np.random.default_rng(seed)
    A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    return c, A, b, [(0.0, 10.0 * scale)] * n


def main():
    global CHECKS
    print(f"{'scale':>8s} {'n':>4s} | " + " ".join(f"{r:>9.0e}" for r in REQUESTS))
    print(" " * 16 + "  (ok = OPTIMAL and matches the simplex oracle; . = mismatch/nonopt)")
    for scale in (1e0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        for n, m in ((10, 6), (40, 20)):
            cells = []
            for rep in range(3):
                c, A, b, bounds = make(n, m, scale, seed=int(np.log10(scale)) * 97 + n + rep)
                ref = LPS.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds)
                row = []
                for req in REQUESTS:
                    res = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bounds,
                                       options={"constr_viol_tol": req})
                    CHECKS += 1
                    good = (res.status == ref.status == SolveStatus.OPTIMAL
                            and abs(res.objective - ref.objective)
                            <= 1e-6 * max(1.0, abs(ref.objective)))
                    row.append(good)
                cells.append(row)
            agg = ["ok" if all(c[k] for c in cells) else "." for k in range(len(REQUESTS))]
            print(f"{scale:8.0e} {n:4d} | " + " ".join(f"{a:>9s}" for a in agg), flush=True)
    print(f"CHECKS_EXECUTED={CHECKS}")
    return 0 if CHECKS else 1


if __name__ == "__main__":
    sys.exit(main())
