"""#1154 panel, arm 3 (v2) — CORRECTED SCORING. Supersedes panel_capability.py.

**Retraction of v1's verdict (CLAUDE.md §11).** v1 scored the ``Σ[...]`` arm
against the *folded-chain arm's own result on the same route*, i.e. it treated a
solver output as an oracle. On the ``hull`` route with a nonlinear (``exp``) body
that oracle is not trustworthy: the chain arm frequently stops at ``feasible``
short of the optimum the ``auto``/``big-m`` routes certify, so v1's 5 "objective
mismatches" and its 1 "bound violation" scored the Σ arm against a *worse*
answer and flagged it for being *better*. All six are re-scored here.

v2 uses an oracle-free soundness gate that cannot have that failure mode:

  * every incumbent, in EVERY arm and route, is feasibility-verified in numpy
    against the ORIGINAL disjunction (box + at least one disjunct satisfied);
  * the best verified-feasible objective over all 6 (route, arm) pairs is a
    valid upper bound on the true minimum -- call it ``ref``;
  * therefore **any reported dual bound above ``ref`` is invalid**, full stop,
    and that is the soundness gate. No reliance on any solver being right.

Reported alongside it, as the net-positive evidence: per-(route, arm)
certification counts, and the primal gap of each arm against ``ref``.

Prints per-case progress (§10) and an executed-comparison count (§6).
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter

import discopt
import discopt.modeling as dm
import numpy as np

print("sources:", discopt.__file__, flush=True)

ROUTES = ("auto", "big-m", "hull")
ARMS = ("chain", "sumover")
FEAS_TOL = 1e-5
BOUND_TOL = 1e-6


def _chain(terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def parts_of(m, x, n_terms, coefs, nonlinear, scale):
    if nonlinear:
        return [dm.exp(coefs[i % len(coefs)] * scale * x[i] / 10.0) for i in range(n_terms)]
    return [coefs[i % len(coefs)] * scale * x[i] - 1.0 for i in range(n_terms)]


def rhs_of(n_terms, nonlinear):
    return float(n_terms) + 1.0 if nonlinear else 2.0


def build(n_terms, n_disj, sense, coefs, nonlinear, *, arm):
    m = dm.Model(f"case_{n_terms}_{n_disj}_{arm}")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_terms)]
    disjuncts = []
    for k in range(n_disj):
        parts = parts_of(m, x, n_terms, coefs, nonlinear, 1.0 + k)
        body = dm.sum(p for p in parts) if arm == "sumover" else _chain(parts)
        rhs = rhs_of(n_terms, nonlinear)
        if sense == "<=":
            disjuncts.append([body <= rhs])
        elif sense == ">=":
            disjuncts.append([body >= -rhs])
        else:
            disjuncts.append([body == rhs])
    m.either_or(disjuncts)
    m.minimize(-sum(x[i] for i in range(n_terms)))
    return m, [f"x{i}" for i in range(n_terms)]


def satisfies(xs, n_terms, n_disj, sense, coefs, nonlinear):
    """Pure-numpy check of the ORIGINAL disjunction. No solver involved."""
    if not all(-FEAS_TOL <= v <= 10.0 + FEAS_TOL for v in xs):
        return False
    rhs = rhs_of(n_terms, nonlinear)
    for k in range(n_disj):
        scale = 1.0 + k
        if nonlinear:
            val = sum(np.exp(coefs[i % len(coefs)] * scale * xs[i] / 10.0) for i in range(n_terms))
        else:
            val = sum(coefs[i % len(coefs)] * scale * xs[i] - 1.0 for i in range(n_terms))
        if sense == "<=" and val <= rhs + FEAS_TOL:
            return True
        if sense == ">=" and val >= -rhs - FEAS_TOL:
            return True
        if sense == "==" and abs(val - rhs) <= FEAS_TOL:
            return True
    return False


def solve(model, route):
    try:
        r = model.solve(gdp_method=route, time_limit=15)
        return {
            "status": str(r.status),
            "objective": None if r.objective is None else float(r.objective),
            "bound": None if r.bound is None else float(r.bound),
            "x": None if r.x is None else dict(r.x),
        }
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed (§7)
        return {"refused": f"{type(exc).__name__}: {exc}"[:160]}


CASES = list(
    itertools.product(
        (2, 3, 5), (2, 3), ("<=", ">=", "=="),
        ((1.0,), (1.0, -1.0), (0.5, 2.0, -1.5)), (False, True),
    )
)

compared = 0
invalid_bounds: list[str] = []
infeasible_incumbents: list[str] = []
refusals: list[str] = []
certified = Counter()          # (route, arm) -> #optimal
attempted = Counter()          # (route, arm) -> #cases where the arm answered
primal_gap_wins = Counter()    # which arm reaches the better verified incumbent

for n_terms, n_disj, sense, coefs, nonlinear in CASES:
    args = (n_terms, n_disj, sense, coefs, nonlinear)
    tag = f"n={n_terms} d={n_disj} s={sense!r} c={coefs} nl={int(nonlinear)}"
    results: dict[tuple[str, str], dict] = {}
    verified: dict[tuple[str, str], float] = {}

    for route in ROUTES:
        for arm in ARMS:
            model, names = build(*args, arm=arm)
            res = solve(model, route)
            results[(route, arm)] = res
            if "refused" in res:
                refusals.append(f"{tag} [{route}/{arm}]: {res['refused']}")
                continue
            attempted[(route, arm)] += 1
            if res["status"] == "optimal":
                certified[(route, arm)] += 1
            if res["x"] is not None:
                xs = [float(res["x"][nm]) for nm in names]
                if satisfies(xs, *args):
                    verified[(route, arm)] = -sum(xs)
                elif res["objective"] is not None:
                    infeasible_incumbents.append(f"{tag} [{route}/{arm}]: {xs} satisfies nothing")

    if not verified:
        print(f"  {tag}: no verified incumbent in any arm -> unscored", flush=True)
        continue
    ref = min(verified.values())          # a valid UPPER bound on the true minimum
    compared += 1

    for key, res in results.items():
        if "refused" in res or res["bound"] is None:
            continue
        if res["bound"] > ref + BOUND_TOL * max(1.0, abs(ref)):
            invalid_bounds.append(
                f"{tag} [{key[0]}/{key[1]}]: dual bound {res['bound']} > verified feasible "
                f"objective {ref} (from {[k for k, v in verified.items() if v == ref]})"
            )

    best_arms = [k for k, v in verified.items() if v <= ref + 1e-9]
    for _route, arm in best_arms:
        primal_gap_wins[arm] += 1

    line = "  ".join(
        f"{r}/{a}:{results[(r, a)].get('status', 'REFUSED')}"
        f"/{results[(r, a)].get('objective')}"
        for r in ROUTES for a in ARMS
    )
    print(f"  {tag}: ref={ref}  {line}", flush=True)

print()
print(f"cases_scored={compared}")
print(f"invalid_bounds={len(invalid_bounds)}")
for s in invalid_bounds:
    print("  INVALID BOUND", s)
print(f"infeasible_incumbents={len(infeasible_incumbents)}")
for s in infeasible_incumbents:
    print("  INFEASIBLE", s)
print(f"refusals={len(refusals)}")
for s in refusals[:20]:
    print("  REFUSED", s)
print()
print("certification rate (optimal / answered), per route and arm:")
for route in ROUTES:
    for arm in ARMS:
        print(f"  {route:6s} {arm:8s}: {certified[(route, arm)]:3d} / {attempted[(route, arm)]:3d}")
print()
print("cases where the arm reaches the best verified-feasible incumbent:")
for arm in ARMS:
    print(f"  {arm:8s}: {primal_gap_wins[arm]}")
print(f"executed_comparisons={compared}")
if compared == 0:
    print("PROBE DID NOT FIRE", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if (invalid_bounds or infeasible_incumbents) else 0)
