"""#2062 graduation panel: reactor-choice GDPs carrying the PFR ``c``/``v`` rows.

Measures ``DISCOPT_OA_CONVEXITY_TAYLOR`` (Taylor-form curvature test inside
``certify_convex_subdivided``) on the class #2062 reports: a PFR whose by-product row
``NC >= 50 w c(theta)`` and volume row ``VR >= 20 w v(theta)`` are convex but were left
without OA cuts by #2054's interval-Hessian bisection. The in-repo MINLPLib corpus does
not contain the class (see ``corpus_scan.py``).

Adapted from ``docs/dev/data/2054-reactor-panel/reactor_panel.py`` (same oracle and
checks; that script is left unchanged so its recorded run stays reproducible). Each
model is ``S`` reactor stages sharing a residence-time budget; each stage is a
disjunction between two reactor types whose rows are univariate in the stage's
``theta``. The by-product and volume columns are priced in the objective, so their
rows are active and their OA cuts matter. Negative controls put a NONCONVEX row
(``(t-2)**3``, ``exp(-exp(t-2))``) or a CONCAVE row (``NC >= 50 w (1.3 - c)``) in a
disjunct, so a false convexity certificate shows as a wrong answer.

Oracle (independent of discopt): a min-plus dynamic program over the shared budget on
a fine ``theta`` grid, polished by SLSQP; the point is re-verified from the numpy row
functions, so its objective is an UPPER bound on the optimum. A certified ``optimal``
objective or any dual bound above it (beyond tolerance) is a false certificate; every
incumbent is re-verified against the spec.

Arms are interleaved per model: ``TAYLOR in {0, 1}`` under ``gdp_method="loa"`` and
``"big-m"``, with ``DISCOPT_OA_CONVEXITY_SUBDIVISION`` at its default (ON). Prints one
line per solve, a summary and executed-check counts; exits non-zero if any check fired
zero times or any soundness check failed.

Usage: python -u pfr_rows_panel.py OUT.jsonl [--time-limit 30] [--quick] [--shard k/n]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

TOL_ABS, TOL_REL, FEAS = 1e-6, 1e-4, 1e-5


# ── reactor rows: f(theta, ops, kappa) ──
def u_pfr(t, o, k):
    return (1 - k) / (o.exp((1 - k) * t) - 1)


def u_cstr2(t, o, k):
    h = t / 2
    return (1 + k * h) ** 2 / (h * (2 + (1 + k) * h))


def u_cstr1(t, o, k):
    return (1 + k * t) / (t * (1 + k))


def c_pfr(t, o, k):
    """#2062's ``c``: C made per B in a PFR."""
    return (1 - k) * (o.exp(t) - 1) / (o.exp((1 - k) * t) - 1) - 1


def v_pfr(t, o, k):
    """#2062's ``v``: theta times A fed per B in a PFR."""
    return (1 - k) * t * o.exp(t) / (o.exp((1 - k) * t) - 1)


def neg_c(t, o, k):
    """Concave: a row ``NC >= 50 w (1.3 - c)`` must get no OA cut."""
    return 1.3 - c_pfr(t, o, k)


def bad_cube(t, o, k):
    return 4 * (t - 2) ** 3 + 40


def bad_expexp(t, o, k):
    return 80 * o.exp(-o.exp(t - 2))


#: Extra priced columns of a stage, each ``col >= w * coef * fn(theta)``.
COLS = ("NC", "VR")
TYPES = {
    # name: (unconverted-feed row, capital c0, c1, extras[(col, coef, fn)])
    "pfr_cv": (u_pfr, 22.0, 11.0, [("NC", 50.0, c_pfr), ("VR", 20.0, v_pfr)]),
    "pfr": (u_pfr, 30.0, 12.0, []),
    "cstr2": (u_cstr2, 10.0, 9.0, []),
    "cstr1": (u_cstr1, 5.0, 6.0, []),
    "pfr_negc": (u_pfr, 22.0, 11.0, [("NC", 50.0, neg_c)]),
    "bad_cube": (bad_cube, 8.0, 7.0, []),
    "bad_expexp": (bad_expexp, 8.0, 7.0, []),
}
LB, UB = 0.25, 4.0
COL_UB = 2000.0


def family(quick: bool):
    """The panel's model specs: (name, stages[(type_a, type_b, kappa, weight)], budget)."""
    rng = np.random.default_rng(2062)
    pos_pairs = [
        ("pfr_cv", "cstr2"),
        ("pfr_cv", "cstr1"),
        ("pfr_cv", "pfr"),
        ("pfr", "cstr2"),
    ]
    neg_pairs = [("bad_cube", "pfr_cv"), ("bad_expexp", "pfr_cv"), ("pfr_negc", "cstr2")]
    specs = []
    n_pos = 6 if quick else 30
    n_neg = 2 if quick else 10
    for i in range(n_pos + n_neg):
        neg = i >= n_pos
        S = int(rng.integers(2, 5))
        pairs = neg_pairs if neg else pos_pairs
        stages = []
        for s in range(S):
            a, b = (
                pairs[int(rng.integers(len(pairs)))]
                if (not neg or s == 0)
                else pos_pairs[int(rng.integers(len(pos_pairs)))]
            )
            stages.append(
                (a, b, float(rng.choice([0.15, 0.25, 0.4])), float(rng.uniform(0.8, 1.8)))
            )
        budget = float(rng.uniform(0.45, 0.8) * S * UB)
        specs.append((f"{'neg' if neg else 'pos'}{i:02d}_S{S}", stages, budget))
    return specs


def build(spec, config=None):
    """The GDP model, or (``config`` given) the continuous model of one configuration."""
    import discopt.modeling as dm

    name, stages, budget = spec
    m = dm.Model(name)
    obj = 0
    thetas = []
    for s, (a, b, k, w) in enumerate(stages):
        th = m.continuous(f"theta{s}", lb=LB, ub=UB)
        nu = m.continuous(f"NU{s}", lb=0, ub=COL_UB)
        cap = m.continuous(f"cap{s}", lb=0, ub=COL_UB)
        cols = {c: m.continuous(f"{c}{s}", lb=0, ub=COL_UB) for c in COLS}
        thetas.append(th)
        groups = []
        for typ in (a, b):
            f, c0, c1, extras = TYPES[typ]
            rows = [w * 50 * f(th, dm, k) - nu <= 0, cap >= c0 + c1 * th]
            rows += [coef * w * fn(th, dm, k) - cols[c] <= 0 for c, coef, fn in extras]
            groups.append(rows)
        if config is None:
            m.either_or(groups, name=f"stage{s}")
        else:
            for r in groups[config[s]]:
                m.subject_to(r)
        obj = obj + nu + cap + sum(cols.values())
    m.subject_to(sum(thetas) <= budget)
    m.minimize(obj)
    return m


def _columns(theta, stage, typ):
    """The least columns of one disjunct at ``theta``; ``ok`` false where one leaves its box."""
    _, _, k, w = stage
    f, c0, c1, extras = TYPES[typ]
    theta = np.asarray(theta, dtype=float)
    with np.errstate(all="ignore"):
        cols = {
            "NU": np.maximum(0.0, w * 50 * f(theta, np, k)),
            "cap": np.maximum(0.0, c0 + c1 * theta),
        }
        for c in COLS:
            cols[c] = np.zeros(theta.shape)
        for c, coef, fn in extras:
            cols[c] = np.maximum(0.0, coef * w * fn(theta, np, k))
    ok = np.ones(theta.shape, dtype=bool)
    for v in cols.values():
        ok &= v <= COL_UB
    return cols, ok


def objective_and_violation(spec, x):
    """Objective and the max violation of ``x`` (dict name -> array) against the spec."""
    _, stages, budget = spec
    v = lambda n: float(np.asarray(x[n]).reshape(()))  # noqa: E731
    obj, worst, tsum = 0.0, 0.0, 0.0
    for s, (a, b, k, w) in enumerate(stages):
        th = v(f"theta{s}")
        names = ["NU", "cap", *COLS]
        vals = {n: v(f"{n}{s}") for n in names}
        tsum += th
        worst = max(
            worst,
            LB - th,
            th - UB,
            *(-q for q in vals.values()),
            *(q - COL_UB for q in vals.values()),
        )
        best_disj = np.inf
        for typ in (a, b):
            f, c0, c1, extras = TYPES[typ]
            viol = max(w * 50 * f(th, np, k) - vals["NU"], c0 + c1 * th - vals["cap"])
            for c, coef, fn in extras:
                viol = max(viol, coef * w * fn(th, np, k) - vals[c])
            best_disj = min(best_disj, viol)
        worst = max(worst, best_disj)
        obj += sum(vals.values())
    worst = max(worst, tsum - budget)
    return obj, worst


def _stage_cost(theta, stage, choice=None):
    """Least cost of one stage at ``theta`` (array), over its disjuncts or one ``choice``."""
    a, b, _, _ = stage
    theta = np.asarray(theta, dtype=float)
    best = np.full(theta.shape, np.inf)
    for idx, typ in enumerate((a, b)):
        if choice is not None and idx != choice:
            continue
        cols, ok = _columns(theta, stage, typ)
        best = np.minimum(best, np.where(ok, sum(cols.values()), np.inf))
    return best


def point_from_theta(spec, theta, choice):
    """The full point implied by ``theta`` and the disjunct choices."""
    _, stages, _ = spec
    x = {}
    for s, st in enumerate(stages):
        t = float(theta[s])
        cols, _ = _columns(t, st, st[choice[s]])
        x[f"theta{s}"] = np.array(t)
        for n, val in cols.items():
            x[f"{n}{s}"] = np.array(float(val))
    return x


def oracle(spec, n_grid=4001):
    """A verified feasible objective at (or within grid-and-polish error of) the optimum.

    Independent of discopt: a min-plus dynamic program over the shared budget on a
    uniform ``theta`` grid, then SLSQP on the continuous problem with the grid's
    disjunct choices fixed. The point is re-verified with
    :func:`objective_and_violation`, so the value is an UPPER bound on the true
    optimum -- what a false certificate (an ``optimal`` objective or a dual bound
    above the optimum) is checked against.
    """
    from scipy.optimize import minimize

    _, stages, budget = spec
    S = len(stages)
    grid = np.linspace(LB, UB, n_grid)
    h = grid[1] - grid[0]
    K = min(int(np.floor((budget - S * LB) / h + 1e-9)), S * (n_grid - 1))
    costs = [_stage_cost(grid, st) for st in stages]
    # F_s[k]: least cost of stages 0..s using exactly k budget steps; back[s][k]: stage s's index.
    F = np.full(K + 1, np.inf)
    F[: min(n_grid, K + 1)] = costs[0][: min(n_grid, K + 1)]
    back = [None]
    for s in range(1, S):
        G = np.full(K + 1, np.inf)
        A = np.zeros(K + 1, dtype=np.int64)
        for i in range(min(n_grid, K + 1)):
            cand = costs[s][i] + F[: K + 1 - i]
            better = cand < G[i:]
            G[i:] = np.where(better, cand, G[i:])
            A[i:] = np.where(better, i, A[i:])
        back.append(A)
        F = G
    k = int(np.argmin(F))
    idx = [0] * S
    for s in range(S - 1, 0, -1):
        idx[s] = int(back[s][k])
        k -= idx[s]
    idx[0] = k
    theta = grid[idx]
    choice = [
        int(np.argmin([float(_stage_cost(theta[s], st, c)) for c in (0, 1)]))
        for s, st in enumerate(stages)
    ]

    def total(t):
        return float(sum(_stage_cost(t[s], st, choice[s]) for s, st in enumerate(stages)))

    cands = [theta]
    res = minimize(
        total,
        theta,
        method="SLSQP",
        bounds=[(LB, UB)] * S,
        constraints=[{"type": "ineq", "fun": lambda t: budget - np.sum(t)}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    cands.append(np.clip(res.x, LB, UB))
    best_val, best_x = np.inf, None
    for t in cands:
        x = point_from_theta(spec, t, choice)
        obj, viol = objective_and_violation(spec, x)
        if viol <= 1e-9 and obj < best_val:
            best_val, best_x = obj, x
    assert best_x is not None, "oracle produced no verified point"
    return best_val, best_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/n: run models with index %% n == k")
    args = ap.parse_args()
    shard_k, shard_n = (int(v) for v in args.shard.split("/"))

    import discopt._relax.convexity.certificate as cert

    # CLAUDE.md §8: assert the code under test is the code loaded.
    assert "_certify_univariate_taylor" in open(cert.__file__).read(), cert.__file__
    print("loaded", cert.__file__, flush=True)

    checks = {
        "oracle_points": 0,
        "oracle_objective": 0,
        "oracle_bound": 0,
        "feasibility": 0,
        "solves": 0,
    }
    failures: list[str] = []
    out = open(args.out, "w")
    os.environ.pop("DISCOPT_OA_CONVEXITY_SUBDIVISION", None)  # its default, ON
    arms = [("0", "0"), ("1", "0")]  # (TAYLOR, unused second field kept for summarize.py)
    for i_spec, spec in enumerate(family(args.quick)):
        if i_spec % shard_n != shard_k:
            continue
        name = spec[0]
        t0 = time.perf_counter()
        opt, _ = oracle(spec)
        checks["oracle_points"] += 1
        print(
            f"{name}: oracle (verified feasible) = {opt:.10g} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        for method in ("loa", "big-m"):
            for sub, bigm in arms:
                os.environ["DISCOPT_OA_CONVEXITY_TAYLOR"] = sub
                t1 = time.perf_counter()
                err = None
                try:
                    r = build(spec).solve(gdp_method=method, time_limit=args.time_limit)
                except Exception as e:  # recorded, never swallowed: it fails the panel
                    err = f"{type(e).__name__}: {e}"
                    r = None
                wall = time.perf_counter() - t1
                checks["solves"] += 1
                row = {
                    "model": name,
                    "method": method,
                    "sub": sub,
                    "bigm": bigm,
                    "wall": wall,
                    "oracle": opt,
                    "error": err,
                }
                if r is not None:
                    st = r.solver_stats or {}
                    row.update(
                        status=r.status,
                        objective=r.objective,
                        bound=r.bound,
                        gap_certified=bool(r.gap_certified),
                        nodes=int(r.node_count),
                        iterations=st.get("iterations"),
                        mip_count=int(r.mip_count),
                    )
                    tol = TOL_ABS + TOL_REL * max(1.0, abs(opt)) if opt is not None else None
                    if opt is not None and r.status == "optimal":
                        checks["oracle_objective"] += 1
                        # A certified optimum above a verified feasible point is false.
                        row["obj_minus_oracle"] = float(r.objective) - opt
                        if float(r.objective) > opt + tol:
                            failures.append(
                                f"{name} {method} {sub}{bigm}: optimal {r.objective} > feasible {opt}"
                            )
                    if opt is not None and r.bound is not None and np.isfinite(r.bound):
                        checks["oracle_bound"] += 1
                        if float(r.bound) > opt + tol:
                            failures.append(
                                f"{name} {method} {sub}{bigm}: bound {r.bound} > oracle {opt}"
                            )
                    if r.x:
                        checks["feasibility"] += 1
                        obj, viol = objective_and_violation(spec, r.x)
                        row["violation"] = viol
                        if viol > 1e-4 or abs(obj - float(r.objective)) > 1e-4 * max(1, abs(obj)):
                            failures.append(
                                f"{name} {method} {sub}{bigm}: incumbent viol {viol:.3g}, obj {obj} vs {r.objective}"
                            )
                else:
                    failures.append(f"{name} {method} {sub}{bigm}: raised {err}")
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    f"  {method:5s} sub={sub} bigm={bigm} {row.get('status')} obj={row.get('objective')} "
                    f"bound={row.get('bound')} nodes={row.get('nodes')} it={row.get('iterations')} "
                    f"wall={wall:.2f}",
                    flush=True,
                )
    print("checks executed:", checks, flush=True)
    print("failures:", len(failures), flush=True)
    for f in failures:
        print("  FAIL", f, flush=True)
    if min(checks.values()) == 0 or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
