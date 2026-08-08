"""Issue #940: is the POUNCE UNBOUNDED-vs-simplex-OPTIMAL disagreement at data
scale 1e6 pre-existing, or introduced by constr_viol_tol=1e-9?

Both arms are run in ONE process against the SAME instances, selecting the
tolerance through the caller option, so the comparison is interleaved and no
module-provenance ambiguity exists (§8, §9).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.solvers.lp_pounce as LPP  # noqa: E402
import discopt.solvers.lp_simplex as LPS  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402

assert LPP._CONSTR_VIOL_TOL == 1e-8
CHECKS = 0


def instances():
    rng = np.random.default_rng(20940)
    for n, m in ((5, 3), (10, 6), (20, 10), (40, 20)):
        for scale in (1e0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
            for rep in range(5):
                A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
                b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
                c = np.abs(rng.uniform(1.0, 10.0, size=n))
                yield (f"n{n}_s{scale:g}_r{rep}", c, A, b, [(0.0, 10.0 * scale)] * n)


def main():
    global CHECKS
    mism_old = mism_new = 0
    rows = []
    for tag, c, A_ub, b_ub, bounds in instances():
        ref = LPS.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        old = LPP.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                           options={"constr_viol_tol": 1e-4})  # Ipopt default = pre-fix
        new = LPP.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)  # post-fix default
        CHECKS += 1
        if old.status != ref.status:
            mism_old += 1
        if new.status != ref.status:
            mism_new += 1
        if old.status != ref.status or new.status != ref.status:
            rows.append((tag, ref.status.name, old.status.name, new.status.name))
    print(f"{'instance':22s} {'simplex':10s} {'pounce@1e-4':12s} {'pounce@default':14s}")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:10s} {r[2]:12s} {r[3]:12s}")
    print(f"\nstatus mismatches vs simplex oracle: pre-fix(1e-4)={mism_old}  "
          f"post-fix({LPP._CONSTR_VIOL_TOL:g})={mism_new}   of {CHECKS} instances")
    print(f"CHECKS_EXECUTED={CHECKS}")
    return 0 if CHECKS else 1


if __name__ == "__main__":
    sys.exit(main())
