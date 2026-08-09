"""Seed 16, exact arm: the GBD master reports bound 128.746 while the monolithic
optimum is 128.732. Tap the master MILP itself — every (c, A_ub, b_ub) it is
handed and every answer it returns — and evaluate the cut rows by hand at all
2^n first-stage points, to see whether the master's answer agrees with its own
rows.

Prints an executed-check count; exits non-zero if nothing was checked.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

import discopt.solvers.lp_backend as lp_backend
import discopt.solvers.nlp_pounce as nlp_pounce
from discopt.decomposition.benders import solve_benders

sys.path.insert(0, "scratchpad/issue946")
from differential_panel import build  # noqa: E402

CHECKS = 0


def main() -> int:
    global CHECKS
    calls = []
    real_get = lp_backend.get_milp_solver

    def tapped_get(*a, **kw):
        milp = real_get(*a, **kw)

        def wrapper(c, **kwargs):
            res = milp(c, **kwargs)
            calls.append((np.asarray(c, float).copy(), kwargs, res))
            return res

        return wrapper

    real_nlp = nlp_pounce.solve_nlp

    def patched_nlp(problem, x0, options=None):
        opts = dict(options or {})
        opts["bound_relax_factor"] = 0.0
        return real_nlp(problem, x0, options=opts)

    lp_backend.get_milp_solver = tapped_get  # type: ignore[assignment]
    nlp_pounce.solve_nlp = patched_nlp  # type: ignore[assignment]
    try:
        r = solve_benders(build(16), time_limit=60)
    finally:
        lp_backend.get_milp_solver = real_get  # type: ignore[assignment]
        nlp_pounce.solve_nlp = real_nlp  # type: ignore[assignment]

    print("GBD:", r.status, "obj=", r.objective, "bound=", r.bound)
    CHECKS += 1

    c, kwargs, res = calls[-1]
    A = kwargs.get("A_ub")
    b = kwargs.get("b_ub")
    print(f"\nfinal master: ncols={len(c)} nrows={0 if A is None else A.shape[0]}")
    print("  status=", res.status, " objective=", res.objective, " bound=", res.bound, " x=", res.x)
    CHECKS += 1
    n_master = len(c) - 1
    print("\ncut rows (x-coeffs | eta-coeff | rhs):")
    for k in range(A.shape[0]):
        print(f"  [{k}] {np.array2string(A[k, :n_master], precision=4)} | {A[k, -1]:+.4g} | {b[k]:+.6g}")
        CHECKS += 1

    print("\nbrute force over 0/1 first-stage points (eta = max lower bound implied by the rows):")
    best = None
    for z in itertools.product([0.0, 1.0], repeat=n_master):
        z = np.array(z)
        lo = -1e12
        feasible = True
        for k in range(A.shape[0]):
            ax = float(A[k, :n_master] @ z)
            ce = float(A[k, -1])
            if abs(ce) < 1e-12:
                if ax > b[k] + 1e-6:
                    feasible = False
                continue
            # ax + ce*eta <= b  with ce = -1  ->  eta >= ax - b
            lo = max(lo, (ax - b[k]) / (-ce))
        print(f"  y={z} feasible={feasible} eta_min={lo:.10g}")
        CHECKS += 1
        if feasible and (best is None or lo < best[1]):
            best = (z, lo)
    print(f"\nhand-computed master optimum: y={best[0]} eta={best[1]:.10g}")
    print(f"master solver reported bound: {res.bound!r}")
    CHECKS += 1
    print(f"\nexecuted checks: {CHECKS}")
    return 0 if CHECKS else 2


if __name__ == "__main__":
    raise SystemExit(main())
