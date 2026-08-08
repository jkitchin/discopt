"""Issue #940: candidate settings against pounce @ main, full LP battery.

Checks each candidate for (a) guard trips, (b) non-convergence, (c) status
disagreement with the exact simplex oracle. A candidate must fix the trips
WITHOUT costing convergence or a certificate.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.solvers.lp_pounce as LPP
from discopt.solvers.lp_simplex import solve_lp as simplex
from discopt.solver import _matrix_solution_feasible
from discopt.solvers import SolveStatus

CHECKS = 0

def battery():
    rng = np.random.default_rng(20940)
    for n, m in ((5, 3), (10, 6), (20, 10), (40, 20)):
        for scale in (1e0, 1e2, 1e3, 1e4, 1e6, 1e7):
            for rep in range(2):
                A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
                b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
                c = np.abs(rng.uniform(1.0, 10.0, size=n))
                yield (f"n{n}_s{scale:g}_r{rep}", c, A, b, [(0.0, 10.0 * scale)] * n)

INSTANCES = list(battery())
# The diet LP (the user-visible symptom) too.
_c = np.array([2.0, 3.5, 8.0, 11.0, 25.0])
_A = np.array([[3.0, 8.0, 15.0, 22.0, 31.0], [25.0, 120.0, 200.0, 10.0, 15.0],
               [1.0, 0.1, 0.5, 3.0, 5.5], [250.0, 150.0, 400.0, 200.0, 450.0]])
_r = np.array([55.0, 800.0, 12.0, 2000.0])
INSTANCES.append(("diet", _c, -_A, -_r, [(0.0, 10.0)] * 5))

CANDIDATES = {
    # Every arm must pin BOTH options explicitly: the module default now sets
    # both, so naming only one silently leaves the other at the shipped value
    # and the arm label becomes a lie (CLAUDE.md §8).
    "POUNCE's own defaults": {"constr_viol_tol": 1e-4, "bound_relax_factor": 1e-8},
    "cvt=1e-8 only": {"constr_viol_tol": 1e-8, "bound_relax_factor": 1e-8},
    "brf=0 only": {"constr_viol_tol": 1e-4, "bound_relax_factor": 0.0},
    "BOTH (shipped)": {"constr_viol_tol": 1e-8, "bound_relax_factor": 0.0},
}

def main():
    global CHECKS
    ref = {}
    for tag, c, A, b, bnds in INSTANCES:
        ref[tag] = simplex(c=c, A_ub=A, b_ub=b, bounds=bnds)
    print(f"{'candidate':26s} {'trips':>6s} {'nonopt':>7s} {'status_mismatch':>16s} "
          f"{'worst_ratio':>12s}  (n={len(INSTANCES)})")
    for label, opts in CANDIDATES.items():
        trips = nonopt = mism = 0
        worst = 0.0
        for tag, c, A, b, bnds in INSTANCES:
            r = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bnds, options=opts or None)
            CHECKS += 1
            if r.status != SolveStatus.OPTIMAL:
                nonopt += 1
                if ref[tag].status == SolveStatus.OPTIMAL:
                    mism += 1
                continue
            if ref[tag].status != SolveStatus.OPTIMAL:
                mism += 1
            x = np.asarray(r.x)
            if not _matrix_solution_feasible(x, A, b, None, None, bnds):
                trips += 1
            v = A @ x - b
            s = np.abs(A) @ np.abs(x)
            worst = max(worst, float(np.max(v / (1e-6 + 1e-9 * s))))
        print(f"{label:26s} {trips:6d} {nonopt:7d} {mism:16d} {worst:12.4g}", flush=True)
    print(f"\nCHECKS_EXECUTED={CHECKS}")
    return 0 if CHECKS else 1

if __name__ == "__main__":
    sys.exit(main())
