"""Issue #940 experiment 3: objective drift of the now-accepted POUNCE LP answer
against the exact simplex oracle.

Before the fix these LPs were answered by the exact simplex (the guard rejected
POUNCE's point). After it, POUNCE's own answer is returned. Measure the drift
against the simplex oracle on exactly that population, plus the residual
infeasibility, so "the accepted point is as good as the one it replaces" is a
measurement, not a claim. §6: prints an executed-comparison count.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import discopt.solvers.lp_pounce as LPP  # noqa: E402
import discopt.solvers.lp_simplex as LPS  # noqa: E402
from discopt.solver import _matrix_solution_feasible  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402

assert LPP.__file__.startswith("/home/user/discopt/python/")
assert "_CONSTR_VIOL_TOL" in dir(LPP), "measuring the PRE-fix module - marker absent (§8)"
assert LPP._CONSTR_VIOL_TOL == 1e-8

COMPARISONS = 0


def instances():
    rng = np.random.default_rng(20940)
    for n, m in ((5, 3), (10, 6), (20, 10), (40, 20)):
        for scale in (1e0, 1e2, 1e3, 1e4, 1e6):
            for rep in range(3):
                A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
                b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
                c = np.abs(rng.uniform(1.0, 10.0, size=n))
                yield (f"n{n}_s{scale:g}_r{rep}", c, A, b, [(0.0, 10.0 * scale)] * n)


def main():
    global COMPARISONS
    drifts, viols = [], []
    worst = None
    disagree = 0
    for tag, c, A_ub, b_ub, bounds in instances():
        rp = LPP.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        rs = LPS.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if rp.status != SolveStatus.OPTIMAL or rs.status != SolveStatus.OPTIMAL:
            disagree += 1
            print(f"  {tag}: status pounce={rp.status.name} simplex={rs.status.name}")
            continue
        COMPARISONS += 1
        rel = abs(rp.objective - rs.objective) / max(1.0, abs(rs.objective))
        drifts.append(rel)
        x = np.asarray(rp.x, float)
        v = float(np.max(np.asarray(A_ub, float) @ x - np.asarray(b_ub, float)))
        viols.append(v)
        assert _matrix_solution_feasible(x, A_ub, b_ub, None, None, bounds), tag
        if worst is None or rel > worst[1]:
            worst = (tag, rel, rp.objective, rs.objective)

    d = np.array(drifts)
    print(f"\nCOMPARISONS_EXECUTED={COMPARISONS}  status disagreements={disagree}")
    print(f"relative objective drift  POUNCE vs exact simplex:")
    print(f"  max = {d.max():.3e}   p99 = {np.percentile(d, 99):.3e}   "
          f"median = {np.median(d):.3e}")
    print(f"  worst instance: {worst[0]}  pounce={worst[2]!r} simplex={worst[3]!r}")
    print(f"POUNCE residual row violation: max = {max(viols):.3e} "
          f"(discopt tolerance 1e-6)")
    print("  (every accepted point also passed _matrix_solution_feasible, asserted above)")
    if COMPARISONS == 0:
        print("PROBE MADE ZERO COMPARISONS", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
