"""Seed 16 of the #946 panel: the post-fix exact arm certifies 128.7460 while the
monolithic optimum is 128.7322. Find out which ingredient is wrong — the global
objective floor L, the Lagrangian anchor, or the model's convexity gate.

Prints an executed-check count; exits non-zero if nothing was checked.
"""

from __future__ import annotations

import sys

import numpy as np

import discopt.solvers.nlp_pounce as nlp_pounce
from discopt._jax.convexity import classify_oa_cut_convexity
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.decomposition.benders import solve_benders
from discopt.decomposition.benders.gbd import _box_min_linear
from discopt.decomposition.structure import flat_bounds

sys.path.insert(0, "scratchpad/issue946")
from differential_panel import build  # noqa: E402

CHECKS = 0


def main() -> int:
    global CHECKS
    m = build(16)
    print("constraints:")
    for c in m._constraints:
        print(f"  {c.sense}  {c.body}")
    print("objective:", m._objective.expression, m._objective.sense)
    print("variables:", [(v.name, v.var_type, v.size, v.lb, v.ub) for v in m._variables])

    conv = classify_oa_cut_convexity(m)
    print("\nconvexity: objective_is_convex=", conv.objective_is_convex, " mask=", conv.constraint_mask)
    CHECKS += 1

    ev = NLPEvaluator(m)
    lb, ub = flat_bounds(m)
    print("box lb=", lb, " ub=", ub)

    mono = build(16).solve(time_limit=60)
    print("\nmonolithic:", mono.status, mono.objective, mono.x)
    CHECKS += 1

    # Replay the recourse solves the exact arm performs, and recompute the floor
    # candidates the same way gbd does.
    real = nlp_pounce.solve_nlp
    seen = []

    def patched(problem, x0, options=None):
        opts = dict(options or {})
        opts["bound_relax_factor"] = 0.0
        res = real(problem, x0, options=opts)
        if res.x is not None:
            seen.append((np.asarray(res.x, float).copy(),
                         None if res.multipliers is None else np.asarray(res.multipliers, float).copy()))
        return res

    nlp_pounce.solve_nlp = patched  # type: ignore[assignment]
    try:
        r = solve_benders(build(16), time_limit=60)
    finally:
        nlp_pounce.solve_nlp = real  # type: ignore[assignment]
    print("\nGBD(exact):", r.status, "obj=", r.objective, "bound=", r.bound)
    CHECKS += 1

    print("\nfloor candidates at each recourse point:")
    for k, (x, mu) in enumerate(seen):
        f = float(ev.evaluate_objective(x))
        grad = np.asarray(ev.evaluate_gradient(x), float)
        val, fin = _box_min_linear(grad, x, range(ev.n_variables), lb, ub)
        floor_f = f + val if fin else None
        line = f"  [{k}] x={np.array2string(x, precision=5)} f={f:.6f} floor_f={floor_f}"
        if mu is not None and mu.size == ev.n_constraints and ev.n_constraints:
            jac = np.asarray(ev.evaluate_jacobian(x), float)
            g = np.asarray(ev.evaluate_constraints(x), float)
            gl = grad + jac.T @ mu
            l0 = f + float(mu @ g)
            vl, finl = _box_min_linear(gl, x, range(ev.n_variables), lb, ub)
            line += f" mu={np.array2string(mu, precision=3)} l0={l0:.6f} floor_lag={(l0 + vl) if finl else None}"
        print(line)
        CHECKS += 1
        if floor_f is not None and mono.objective is not None:
            assert floor_f <= mono.objective + 1e-6 * (1 + abs(mono.objective)), (
                f"INVALID FLOOR: {floor_f} > monolithic optimum {mono.objective}"
            )

    print(f"\nexecuted checks: {CHECKS}")
    return 0 if CHECKS else 2


if __name__ == "__main__":
    raise SystemExit(main())
