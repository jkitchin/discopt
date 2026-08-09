"""Issue #940: does the same POUNCE constr_viol_tol mismatch reach the QP path?

The QP call sites in solver.py (``_solve_qp_matrix``) use the SAME #850 guard
``_matrix_solution_feasible``, and ``qp_pounce.solve_qp`` builds its option dict
the same way as ``lp_pounce.solve_lp``. Measure rather than assume (CLAUDE.md §4).
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.solvers.qp_pounce as QPP  # noqa: E402
from discopt.solver import _matrix_solution_feasible  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402

assert QPP.__file__.startswith("/home/user/discopt/python/"), QPP.__file__
assert QPP.POUNCE_AVAILABLE

CHECKS = 0


def worst_viol(x, A_ub, b_ub, bounds):
    x = np.asarray(x, float)
    v = np.asarray(A_ub, float) @ x - np.asarray(b_ub, float)
    wv = float(v.max())
    for xi, (lo, hi) in zip(x, bounds):
        wv = max(wv, lo - xi, xi - hi)
    return wv


def main():
    global CHECKS
    rng = np.random.default_rng(20)
    for opts_label, opts in (("Ipopt default 1e-4", {"constr_viol_tol": 1e-4}), ("SHIPPED 1e-8", {"constr_viol_tol": 1e-8})):
        trips = 0
        worst = 0.0
        n_inst = 0
        for n, m in ((5, 3), (10, 6), (20, 10)):
            for scale in (1e2, 1e3, 1e4):
                for rep in range(2):
                    B = rng.normal(size=(n, n))
                    Q = B @ B.T + n * np.eye(n)
                    c = -np.abs(rng.uniform(1.0, 10.0, size=n)) * scale
                    A_ub = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
                    b_ub = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
                    bounds = [(0.0, 10.0 * scale)] * n
                    res = QPP.solve_qp(Q=Q, c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                                       options=opts)
                    CHECKS += 1
                    n_inst += 1
                    if res.status != SolveStatus.OPTIMAL:
                        continue
                    ok = _matrix_solution_feasible(res.x, A_ub, b_ub, None, None, bounds)
                    worst = max(worst, worst_viol(res.x, A_ub, b_ub, bounds))
                    if not ok:
                        trips += 1
        print(f"{opts_label:12s} guard trips {trips}/{n_inst}  worst_abs_viol={worst:.4g}",
              flush=True)
    print(f"CHECKS_EXECUTED={CHECKS}")
    if CHECKS == 0:
        print("PROBE FIRED ZERO CHECKS", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
