"""#801 Stage 1 — gap attribution + structural audit (steers Stage 2).

Measures, on the REAL tanksize instance:

1. **Integers-fixed root LP.** Fix the 9 integer split vars (x18..x26) to their
   certified-optimal leaf and re-measure the root McCormick LP. #764 found this
   still 0.838 → the entire gap is continuous×continuous bilinear. Re-confirm on
   HEAD and name the loose products/stars.
2. **Ceiling C.** The continuous-relaxation global optimum satisfies
   0.955 (BARON root) ≤ C ≤ 1.2686 (MINLP incumbent). Multistart local NLP over
   the integer-relaxed box gives an *upper* estimate of C — bounding the maximum
   gain ANY root relaxation can deliver (0.838 → C).
3. **Structural / PQ audit.** Reconstruct the linear rows; classify the split
   group; determine whether a pooling-PQ formulation could add a product family
   OUTSIDE the falsified level-1 RLT closure (gates Stage 2c).
4. **Loose-star map.** The continuous×continuous products and their
   shared-variable stars — the SDP cliques for Stage 2b.

Run: ``python discopt_benchmarks/scripts/issue801_stage1_attribution.py``
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
from discopt._jax.term_classifier import classify_nonlinear_terms  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402
from issue801_root_probe import (  # noqa: E402
    OBJ_VAR,
    ORACLE,
    RESULTS,
    baseline_root_lp,
    load,
    root_box,
)
from scipy.optimize import minimize  # noqa: E402

BARON_ROOT = 0.955
INT_VARS = list(range(18, 27))  # x18..x26, the 9 integer split vars


def _flat(xr, n):
    v = np.zeros(n)
    for k, val in xr.items():
        v[int(k[1:])] = float(np.asarray(val).ravel()[0])
    return v


def certified_optimum(model, time_limit=60.0):
    res = model.solve(time_limit=time_limit, gap_tolerance=1e-4)
    return res, _flat(res.x, model_to_repr(model).n_vars)


def integers_fixed_root(model, xopt, lb, ub):
    """Root McCormick LP with x18..x26 pinned to their optimal leaf."""
    lb2, ub2 = lb.copy(), ub.copy()
    for j in INT_VARS:
        lb2[j] = ub2[j] = float(round(xopt[j]))
    probe = baseline_root_lp(model, lb2, ub2)
    return probe.bound, [float(round(xopt[j])) for j in INT_VARS]


def linear_rows(repr_):
    """Reconstruct (a, const, sense, rhs) for every linear constraint by exact FD."""
    n = repr_.n_vars
    e0 = np.zeros(n)
    rows = []
    for i in range(repr_.n_constraints):
        if not repr_.is_constraint_linear(i):
            continue
        c0 = repr_.evaluate_constraint(i, e0)
        a = np.zeros(n)
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            a[j] = repr_.evaluate_constraint(i, ej) - c0
        rows.append(
            {
                "i": i,
                "a": a,
                "const": float(c0),
                "sense": str(repr_.constraint_sense(i)),
                "rhs": float(repr_.constraint_rhs(i)),
            }
        )
    return rows


def pq_audit(rows):
    """Find sum-structure rows touching the split group; classify for PQ coverage."""
    split = set(INT_VARS)
    hits = []
    for r in rows:
        support = {j for j in range(r["a"].size) if abs(r["a"][j]) > 1e-9}
        inter = support & split
        if not inter:
            continue
        coeffs = {int(j): float(r["a"][j]) for j in support}
        # sum-to-one / simplex signature: all-ones over a subset with rhs const
        vals = np.array([r["a"][j] for j in sorted(inter)])
        is_simplex = np.allclose(vals, vals[0]) and abs(vals[0]) > 1e-9
        hits.append(
            {
                "i": r["i"],
                "support_size": len(support),
                "split_support": sorted(int(j) for j in inter),
                "sense": r["sense"],
                "rhs": r["rhs"],
                "const": r["const"],
                "simplex_like": bool(is_simplex and support == inter),
                "coeffs": coeffs,
            }
        )
    return hits


def loose_stars(model):
    """Continuous×continuous products (integers relaxed away) and their stars."""
    terms = classify_nonlinear_terms(model)
    ints = set(INT_VARS)
    cc = [(i, j) for (i, j) in terms.bilinear if i not in ints and j not in ints]
    # shared-variable stars: group products by each shared endpoint
    from collections import defaultdict

    star = defaultdict(set)
    for i, j in cc:
        star[i].add(j)
        star[j].add(i)
    stars = {int(k): sorted(int(v) for v in vs) for k, vs in star.items() if len(vs) >= 2}
    return [list(p) for p in cc], stars


def ceiling_C(repr_, lb, ub, xopt=None, n_starts=8, seed=0, time_budget=90.0):
    """Upper estimate of the continuous-relaxation optimum via multistart SLSQP.

    Integers relaxed to their continuous box. Objective = x17. Any feasible point
    found gives C ≤ its objective, bounding the max root-relaxation gain. Bounded
    budget (this only bounds the max gain — it is not the GO/KILL decider).
    """
    import time as _time

    n = repr_.n_vars
    cons = []
    for i in range(repr_.n_constraints):
        s = str(repr_.constraint_sense(i))
        rhs = float(repr_.constraint_rhs(i))

        def body(x, i=i):
            return repr_.evaluate_constraint(i, np.asarray(x, dtype=np.float64))

        if s == "<=":
            cons.append({"type": "ineq", "fun": (lambda x, b=body, r=rhs: r - b(x))})
        elif s == ">=":
            cons.append({"type": "ineq", "fun": (lambda x, b=body, r=rhs: b(x) - r)})
        else:  # equality
            cons.append({"type": "eq", "fun": (lambda x, b=body, r=rhs: b(x) - r)})
    bounds = list(zip(lb, ub))

    def obj(x):
        return x[OBJ_VAR]

    g = np.zeros(n)
    g[OBJ_VAR] = 1.0

    def jac(x):
        return g

    rng = np.random.default_rng(seed)
    best = np.inf
    best_x = None
    span = np.where(np.isfinite(ub - lb), ub - lb, 1.0)
    lo = np.where(np.isfinite(lb), lb, -1.0)
    # Seed the first start from the known optimum (integers relaxed): a local NLP
    # from there reveals whether the continuous relaxation dips below the incumbent.
    starts = []
    if xopt is not None:
        starts.append(np.asarray(xopt, dtype=np.float64).copy())
    t0 = _time.perf_counter()
    for k in range(n_starts):
        if _time.perf_counter() - t0 > time_budget:
            break
        x0 = starts[k] if k < len(starts) else lo + rng.random(n) * span
        try:
            r = minimize(
                obj, x0, method="SLSQP", jac=jac, bounds=bounds, constraints=cons,
                options={"maxiter": 80, "ftol": 1e-8},
            )
        except Exception:
            continue
        if r.success:
            feas = all(
                (c["fun"](r.x) >= -1e-5) if c["type"] == "ineq" else (abs(c["fun"](r.x)) <= 1e-5)
                for c in cons
            )
            if feas and r.fun < best:
                best = float(r.fun)
                best_x = r.x.copy()
    return best, best_x


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = load()
    repr_ = model_to_repr(model)
    lb, ub = root_box(model)

    base = baseline_root_lp(model, lb, ub)
    res, xopt = certified_optimum(model)
    np.save(os.path.join(RESULTS, "tanksize_opt.npy"), xopt)

    int_fixed_bound, leaf = integers_fixed_root(model, xopt, lb, ub)
    cc_products, stars = loose_stars(model)

    rows = linear_rows(repr_)
    pq = pq_audit(rows)

    # Save the fast (structural) results before the bounded ceiling-C solve.
    fast = {
        "root_mccormick_lp": base.bound,
        "certified_optimum": float(res.objective),
        "integers_fixed_leaf": leaf,
        "integers_fixed_root_lp": int_fixed_bound,
        "integers_fixed_still_loose": bool(int_fixed_bound < 0.90),
        "n_cont_x_cont_products": len(cc_products),
        "cont_x_cont_products": cc_products,
        "shared_variable_stars": stars,
    }
    print("FAST:", json.dumps(fast, default=str))
    with open(os.path.join(RESULTS, "stage1_fast.json"), "w") as f:
        json.dump(fast, f, indent=2, default=str)

    C_est, C_x = ceiling_C(repr_, lb, ub, xopt=xopt)

    out = {
        "root_mccormick_lp": base.bound,
        "certified_optimum": float(res.objective),
        "oracle": ORACLE,
        "integers_fixed_leaf": leaf,
        "integers_fixed_root_lp": int_fixed_bound,
        "integers_fixed_still_loose": bool(int_fixed_bound < 0.90),
        "continuous_ceiling_C_upper_est": None if not np.isfinite(C_est) else C_est,
        "baron_root": BARON_ROOT,
        "max_root_gain_available": (
            None if not np.isfinite(C_est) else C_est - base.bound
        ),
        "cont_x_cont_products": cc_products,
        "n_cont_x_cont_products": len(cc_products),
        "shared_variable_stars": stars,
        "pq_split_group_rows": pq,
    }
    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join(RESULTS, "stage1_attribution.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
