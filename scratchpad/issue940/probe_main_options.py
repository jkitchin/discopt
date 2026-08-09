"""Issue #940, against pounce @ git main (what CI actually installs).

The 1e-8 constr_viol_tol request that works on PyPI 0.9.0 does NOT bind on main:
the violation there is ~8.1e-8 * data scale — a RELATIVE criterion. Find what, if
anything, binds it to discopt's per-row test.

Provenance (§8): asserts the CI signature is reproduced before measuring anything.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import discopt.solvers.lp_pounce as LPP
from discopt.solver import _matrix_solution_feasible
from discopt.solvers import SolveStatus

CHECKS = 0


def case(scale, seed=940, n=20, m=10):
    rng = np.random.default_rng(seed)
    A = -np.abs(rng.uniform(0.5, 5.0, size=(m, n)))
    b = -scale * np.abs(rng.uniform(0.5, 2.0, size=m)) * n * 0.25
    c = np.abs(rng.uniform(1.0, 10.0, size=n))
    return c, A, b, [(0.0, 10.0 * scale)] * n


# ---- §8 provenance gate: must reproduce CI's numbers, else wrong build --------
c, A, b, bnds = case(1e2)
probe = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bnds)
sig = float(np.max(A @ np.asarray(probe.x) - b))
assert abs(sig - 8.096e-06) / 8.096e-06 < 0.01, (
    f"worst violation {sig:.4g} does not match the CI signature 8.096e-06 — "
    "this is NOT the build CI runs; abort before measuring")
print(f"provenance OK: reproduced CI signature {sig:.4g}\n")

CANDIDATES = {
    "(none - main default)": {},
    "constr_viol_tol=1e-8": {"constr_viol_tol": 1e-8},
    "constr_viol_tol=1e-12": {"constr_viol_tol": 1e-12},
    "tol=1e-10": {"tol": 1e-10},
    "tol=1e-12": {"tol": 1e-12},
    "tol=1e-12 + cvt=1e-12": {"tol": 1e-12, "constr_viol_tol": 1e-12},
    "bound_relax_factor=0": {"bound_relax_factor": 0.0},
    "bound_relax_factor=0 + tol=1e-12": {"bound_relax_factor": 0.0, "tol": 1e-12},
    "nlp_scaling_method=none": {"nlp_scaling_method": "none"},
    "nlp_scaling_method=none + tol=1e-12": {"nlp_scaling_method": "none", "tol": 1e-12},
    "honor_original_bounds=yes": {"honor_original_bounds": "yes"},
}


def main():
    global CHECKS
    print(f"{'candidate':38s} {'trips':>6s} {'worst_viol@1e2':>15s} {'@1e3':>12s} "
          f"{'@1e4':>12s} {'nonopt':>7s}")
    for label, opts in CANDIDATES.items():
        trips = nonopt = 0
        viols = []
        for scale in (1e2, 1e3, 1e4):
            c, A, b, bnds = case(scale)
            r = LPP.solve_lp(c=c, A_ub=A, b_ub=b, bounds=bnds, options=opts or None)
            CHECKS += 1
            if r.status != SolveStatus.OPTIMAL:
                nonopt += 1
                viols.append(float("nan"))
                continue
            x = np.asarray(r.x)
            viols.append(float(np.max(A @ x - b)))
            if not _matrix_solution_feasible(x, A, b, None, None, bnds):
                trips += 1
        print(f"{label:38s} {trips:6d} {viols[0]:15.4g} {viols[1]:12.4g} "
              f"{viols[2]:12.4g} {nonopt:7d}", flush=True)
    print(f"\nCHECKS_EXECUTED={CHECKS}")
    return 0 if CHECKS else 1


if __name__ == "__main__":
    sys.exit(main())
