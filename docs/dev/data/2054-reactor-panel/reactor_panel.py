"""#2054 graduation panel: synthetic reactor-choice GDP family, flags OFF vs ON.

Measures ``DISCOPT_OA_CONVEXITY_SUBDIVISION`` (OA cut mask by box subdivision) on the
class #2054 reports, which the in-repo MINLPLib corpus does not contain (0 rows
changed over its 74 instances). Graduated default-ON on this panel (2026-10-30).

The recorded run (commit 9b6a3c3) also had arms for ``DISCOPT_GDP_BIGM_SUBDIVISION``
(big-M enclosure by box subdivision). That flag was cert-clean but not net-positive
(identical B&B node totals) and was retired with its implementation; its arms are
gone from this script and the ``bigm`` field is always ``"0"``. See
``docs/dev/flag-retirement-audit.md``.

Each model is ``S`` reactor stages sharing a residence-time budget; each stage is a
disjunction between two reactor types whose unconverted-feed row (and, for the
by-product PFR, a by-product equality) is univariate in the stage's ``theta``.
Negative controls put a NONCONVEX row in a disjunct (``(t-2)**3``, ``exp(-exp(t-2))``)
so a false convexity certificate or a too-small M would show as a wrong answer.

Oracle (independent of discopt): every stage is univariate and the stages couple
only through a linear budget, so a min-plus dynamic program on a fine ``theta`` grid,
polished by SLSQP, gives a verified feasible point near the optimum. A certified
``optimal`` objective or any dual bound ABOVE that value (beyond tolerance) is a false
certificate. Each incumbent is re-verified from the spec's own numpy row functions,
which catches the other direction (an objective below the optimum is infeasible).

Arms are interleaved per model: ``SUB in {0, 1}`` under
``gdp_method="loa"`` and ``gdp_method="big-m"``. Prints one line per solve, a
summary, and the executed-check counts; exits non-zero if any check fired zero times
or any soundness check failed.

Usage: python -u reactor_panel.py OUT.jsonl [--time-limit 60] [--quick] [--shard k/n]
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


def byprod(t, o, k):
    return 60 * k * t / (o.exp((1 - k) * t) - 1)


def bad_cube(t, o, k):
    return 4 * (t - 2) ** 3 + 40


def bad_expexp(t, o, k):
    return 80 * o.exp(-o.exp(t - 2))


TYPES = {
    # name: (unconverted-feed row, has by-product, capital c0, c1)
    "pfr": (u_pfr, False, 30.0, 12.0),
    "cstr2": (u_cstr2, False, 10.0, 9.0),
    "cstr1": (u_cstr1, False, 5.0, 6.0),
    "pfr_bp": (u_pfr, True, 22.0, 11.0),
    "bad_cube": (bad_cube, False, 8.0, 7.0),
    "bad_expexp": (bad_expexp, False, 8.0, 7.0),
}
LB, UB = 0.25, 4.0
P_PRICE = 1.5


def family(quick: bool):
    """The panel's model specs: (name, stages[(type_a, type_b, kappa, weight)], budget)."""
    rng = np.random.default_rng(2054)
    pos_pairs = [
        ("pfr", "cstr2"),
        ("pfr_bp", "cstr2"),
        ("pfr", "cstr1"),
        ("pfr_bp", "cstr1"),
        ("cstr2", "cstr1"),
    ]
    neg_pairs = [("bad_cube", "cstr2"), ("bad_expexp", "pfr"), ("bad_cube", "pfr_bp")]
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
        nu = m.continuous(f"NU{s}", lb=0, ub=2000)
        cap = m.continuous(f"cap{s}", lb=0, ub=2000)
        p = m.continuous(f"P{s}", lb=0, ub=200)
        thetas.append(th)
        groups = []
        for typ in (a, b):
            f, bp, c0, c1 = TYPES[typ]
            rows = [w * 50 * f(th, dm, k) - nu <= 0, cap >= c0 + c1 * th]
            if bp:
                rows.append(p == byprod(th, dm, k))
            groups.append(rows)
        if config is None:
            m.either_or(groups, name=f"stage{s}")
        else:
            for r in groups[config[s]]:
                m.subject_to(r)
        obj = obj + nu + cap + P_PRICE * p
    m.subject_to(sum(thetas) <= budget)
    m.minimize(obj)
    return m


def objective_and_violation(spec, x):
    """Objective and the max violation of ``x`` (dict name -> array) against the spec."""
    _, stages, budget = spec
    v = lambda n: float(np.asarray(x[n]).reshape(()))  # noqa: E731
    obj, worst = 0.0, 0.0
    tsum = 0.0
    for s, (a, b, k, w) in enumerate(stages):
        th, nu, cap, p = v(f"theta{s}"), v(f"NU{s}"), v(f"cap{s}"), v(f"P{s}")
        tsum += th
        worst = max(worst, LB - th, th - UB, -nu, nu - 2000, -cap, cap - 2000, -p, p - 200)
        best_disj = np.inf
        for typ in (a, b):
            f, bp, c0, c1 = TYPES[typ]
            viol = max(w * 50 * f(th, np, k) - nu, c0 + c1 * th - cap)
            if bp:
                viol = max(viol, abs(p - byprod(th, np, k)))
            best_disj = min(best_disj, viol)
        worst = max(worst, best_disj)
        obj += nu + cap + P_PRICE * p
    worst = max(worst, tsum - budget)
    return obj, worst


def _stage_cost(theta, stage, choice=None):
    """Least cost of one stage at ``theta`` (array), over its disjuncts or one ``choice``.

    With ``theta`` fixed the stage's other columns sit at their lower limits:
    ``NU = max(0, w*50*f)``, ``cap = max(0, c0 + c1*theta)``, ``P`` = the by-product
    (0 without one). A disjunct whose implied column leaves its box is infeasible.
    """
    a, b, k, w = stage
    theta = np.asarray(theta, dtype=float)
    best = np.full(theta.shape, np.inf)
    for idx, typ in enumerate((a, b)):
        if choice is not None and idx != choice:
            continue
        f, bp, c0, c1 = TYPES[typ]
        with np.errstate(all="ignore"):
            nu = np.maximum(0.0, w * 50 * f(theta, np, k))
            cap = np.maximum(0.0, c0 + c1 * theta)
            p = byprod(theta, np, k) if bp else np.zeros(theta.shape)
        ok = (nu <= 2000) & (cap <= 2000) & (p >= 0) & (p <= 200)
        best = np.minimum(best, np.where(ok, nu + cap + P_PRICE * p, np.inf))
    return best


def point_from_theta(spec, theta, choice):
    """The full point implied by ``theta`` and the disjunct choices."""
    _, stages, _ = spec
    x = {}
    for s, (a, b, k, w) in enumerate(stages):
        f, bp, c0, c1 = TYPES[(a, b)[choice[s]]]
        t = float(theta[s])
        x[f"theta{s}"] = np.array(t)
        x[f"NU{s}"] = np.array(max(0.0, w * 50 * f(t, np, k)))
        x[f"cap{s}"] = np.array(max(0.0, c0 + c1 * t))
        x[f"P{s}"] = np.array(byprod(t, np, k) if bp else 0.0)
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
    choice = [int(np.argmin([float(_stage_cost(theta[s], st, c)) for c in (0, 1)]))
              for s, st in enumerate(stages)]

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
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/n: run models with index %% n == k")
    args = ap.parse_args()
    shard_k, shard_n = (int(v) for v in args.shard.split("/"))

    import discopt._relax.convexity.certificate as cert
    # CLAUDE.md §8: assert the code under test is the code loaded.
    assert "certify_convex_subdivided" in open(cert.__file__).read(), cert.__file__
    print("loaded", cert.__file__, flush=True)

    checks = {"oracle_points": 0, "oracle_objective": 0, "oracle_bound": 0, "feasibility": 0,
              "solves": 0}
    failures: list[str] = []
    out = open(args.out, "w")
    arms = [("0", "0"), ("1", "0")]  # (SUB, retired BIGM arm held at "0")
    for i_spec, spec in enumerate(family(args.quick)):
        if i_spec % shard_n != shard_k:
            continue
        name = spec[0]
        t0 = time.perf_counter()
        opt, _ = oracle(spec)
        checks["oracle_points"] += 1
        print(f"{name}: oracle (verified feasible) = {opt:.10g} in {time.perf_counter() - t0:.1f}s",
              flush=True)
        for method in ("loa", "big-m"):
            for sub, bigm in arms:
                os.environ["DISCOPT_OA_CONVEXITY_SUBDIVISION"] = sub
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
