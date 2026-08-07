"""Issue #940: does POUNCE's bound_relax_factor also trip the #850 guard?

Ipopt relaxes variable bounds by ``bound_relax_factor*(1+|bound|)`` (default
1e-8), so on a binding bound of magnitude B the returned x can sit ~1e-8*B
outside it, while the guard allows only ``1e-6 + 1e-9*|x|``. Measure it.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.solvers.lp_pounce as LPP  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402

assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
CHECKS = 0


def run(label, opts):
    global CHECKS
    rows = []
    for B in (1e0, 1e2, 1e4, 1e6, 1e8):
        n = 4
        # max sum x  ==  min -sum x, every x pinned at its upper bound B
        c = -np.ones(n)
        A_ub = np.ones((1, n))
        b_ub = np.array([n * B * 10.0])  # non-binding row; the BOUND is what binds
        bounds = [(0.0, B)] * n
        res = LPP.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=opts)
        CHECKS += 1
        if res.status != SolveStatus.OPTIMAL:
            rows.append((B, res.status.name, float("nan"), float("nan")))
            continue
        x = np.asarray(res.x, float)
        over = float(np.max(x - B))
        thr = 1e-6 + 1e-9 * float(np.max(np.abs(x)))
        rows.append((B, "OPTIMAL", over, over / thr))
    print(f"--- {label}")
    for B, st, over, ratio in rows:
        print(f"    ub={B:<8.0e} {st:12s} over_bound={over:12.4g} ratio={ratio:8.3g}"
              f"{'  <-- GUARD TRIPS' if ratio > 1 else ''}")


def main():
    run("baseline", None)
    run("constr_viol_tol=1e-9", {"constr_viol_tol": 1e-9})
    run("constr_viol_tol=1e-9 + bound_relax_factor=0", 
        {"constr_viol_tol": 1e-9, "bound_relax_factor": 0.0})
    print(f"CHECKS_EXECUTED={CHECKS}")
    return 0 if CHECKS else 1


if __name__ == "__main__":
    sys.exit(main())
