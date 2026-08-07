"""Standalone POUNCE reproduction: an IPM exit of code 3/4 on a well-posed,
feasible, BOUNDED LP whose optimum scipy finds without difficulty.

No discopt involved — pounce + scipy only, so the report stands on its own.

    min  c'x   s.t.  A x >= b,  0 <= x <= 1e8
with c, A entries O(1) and b of magnitude ~1e7. The feasible set is a nonempty
subset of a compact box, so the LP is bounded by construction; scipy's HiGHS
confirms a finite optimum. POUNCE exits Search_Direction_Becomes_Too_Small (3)
or Diverging_Iterates (4).
"""

import sys

import numpy as np
import pounce
from scipy.optimize import linprog

_INF = 1e20
CODES = {0: "Solve_Succeeded", 1: "Solved_To_Acceptable_Level",
         2: "Infeasible_Problem_Detected", 3: "Search_Direction_Becomes_Too_Small",
         4: "Diverging_Iterates", -1: "Maximum_Iterations_Exceeded"}


class LP:
    """cyipopt-style callbacks for min c'x with constant Jacobian A."""

    def __init__(self, c, A):
        self.c, self.A = c, A
        m, n = A.shape
        self.jac = A.ravel().astype(float)
        r, cc = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
        self.rows, self.cols = r.ravel(), cc.ravel()

    def objective(self, x):
        return float(self.c @ x)

    def gradient(self, x):
        return self.c

    def constraints(self, x):
        return np.asarray(self.A @ x, dtype=float)

    def jacobian(self, x):
        return self.jac

    def jacobianstructure(self):
        return self.rows, self.cols

    def hessian(self, x, lagrange, obj_factor):
        return np.empty(0)

    def hessianstructure(self):
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)


def run(seed, n, m, scale):
    rng = np.random.default_rng(seed)
    A = np.abs(rng.uniform(0.5, 5.0, size=(m, n)))       # A x >= b
    b = scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    hi = 10.0 * scale

    ref = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0.0, hi)] * n, method="highs")

    lb = np.zeros(n)
    ub = np.full(n, hi)
    cl = b.copy()
    cu = np.full(m, _INF)
    prob = pounce.Problem(n=n, m=m, problem_obj=LP(c, A), lb=lb, ub=ub, cl=cl, cu=cu)
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.clip(0.5 * (lb + ub), -1e3, 1e3))
    code = info.get("status", -100)
    return code, (ref.status == 0), (float(ref.fun) if ref.status == 0 else None)


def main():
    runs = 0
    bad = []
    print(f"{'seed':>4s} {'n':>4s} {'scale':>8s}  {'pounce exit':38s} {'scipy':>10s}")
    for scale in (1e6, 1e7, 1e8):
        for n, m in ((20, 10), (40, 20)):
            for seed in range(8):
                code, ok, obj = run(seed, n, m, scale)
                runs += 1
                if code in (3, 4) and ok:
                    bad.append((seed, n, scale, code, obj))
                    print(f"{seed:4d} {n:4d} {scale:8.0e}  {CODES.get(code, code):38s} "
                          f"{obj:10.4g}", flush=True)
    print(f"\nRUNS_EXECUTED={runs}")
    print(f"code 3/4 exits on LPs scipy solves to a finite optimum: {len(bad)}/{runs}")
    if runs == 0:
        print("REPRO EXECUTED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
