"""#1154 panel, arm 3 (v1) — SUPERSEDED by panel_capability_v2.py.

Kept unmodified for the retraction record (CLAUDE.md §11): its verdict -- 1 bound
violation, 2 certification regressions, 5 objective mismatches -- is WITHDRAWN.
It scored the ``Σ`` arm against the folded-chain arm's own result on the same
route, i.e. it used a solver output as an oracle, and on the ``hull`` route with a
nonlinear body that arm is frequently the one that has not converged. See
``docs/dev/issue-1154-gdp-sumover-panel-2026-09-04.md`` §3b. Two locals
(``recovered``, ``off_answers``) are declared and never read; they are left as
run rather than tidied, so this file is exactly what produced its log.

Original docstring follows.

#1154 panel, arm 3 — the class where the mechanism actually FIRES.

The .nl corpus contains zero ``SumOverExpression`` nodes (arm 1), so it can
measure *no risk* but not *benefit*. The class this change is for is Python-API
GDP models whose disjunct body is written with ``dm.sum(...)``. This arm sweeps
a generated family of them -- varying term count, disjunct count, constraint
sense, coefficient signs and linear/nonlinear bodies -- and checks each against
an **independent oracle built from the same model**: the identical disjunction
with the body written as the explicit left-folded chain ``t1 + t2 + ...``, which
is a plain ``BinaryOp`` tree the solver has always handled.

That is the whole contract of #1154 (``Σ[t1..tn]`` == ``t1 + .. + tn``) measured
end to end, on certificates rather than on walker return values. Per CLAUDE.md §5
it also checks, for every (model, route) pair:

  * cert-clean -- the dual bound never exceeds the chain oracle's certified
    objective, and no ``optimal`` certificate is lost relative to the chain; and
  * the returned incumbent is feasibility-verified against the ORIGINAL
    disjunction (evaluated in numpy here, independent of the solver).

Prints per-case progress (§10) and an executed-comparison count (§6).
"""

from __future__ import annotations

import itertools
import sys

import discopt
import discopt.modeling as dm
import numpy as np

print("sources:", discopt.__file__, flush=True)

TOL = 1e-4
ROUTES = ("auto", "big-m", "hull")


def _chain(terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def build(n_terms, n_disj, sense, coefs, nonlinear, *, sumover):
    """One GDP model. ``sumover=True`` writes the body with dm.sum, else a chain.

    Both arms are the SAME mathematical model; only the node type differs.
    """
    m = dm.Model(f"case_{n_terms}_{n_disj}_{sense}_{nonlinear}_{sumover}")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_terms)]

    def body(scale):
        if nonlinear:
            parts = [dm.exp(coefs[i % len(coefs)] * scale * x[i] / 10.0) for i in range(n_terms)]
        else:
            parts = [coefs[i % len(coefs)] * scale * x[i] - 1.0 for i in range(n_terms)]
        return dm.sum(p for p in parts) if sumover else _chain(parts)

    disjuncts = []
    for k in range(n_disj):
        scale = 1.0 + k
        rhs = 2.0 if not nonlinear else float(n_terms) + 1.0
        if sense == "<=":
            disjuncts.append([body(scale) <= rhs])
        elif sense == ">=":
            disjuncts.append([body(scale) >= -rhs])
        else:
            disjuncts.append([body(scale) == rhs])
    m.either_or(disjuncts)
    m.minimize(-sum(x[i] for i in range(n_terms)))
    return m, [f"x{i}" for i in range(n_terms)]


def evaluate_disjunction(xs, n_terms, n_disj, sense, coefs, nonlinear):
    """Does the point satisfy AT LEAST ONE disjunct? Pure numpy, no solver."""
    for k in range(n_disj):
        scale = 1.0 + k
        if nonlinear:
            val = sum(np.exp(coefs[i % len(coefs)] * scale * xs[i] / 10.0) for i in range(n_terms))
            rhs = float(n_terms) + 1.0
        else:
            val = sum(coefs[i % len(coefs)] * scale * xs[i] - 1.0 for i in range(n_terms))
            rhs = 2.0
        if sense == "<=" and val <= rhs + 1e-5:
            return True
        if sense == ">=" and val >= -rhs - 1e-5:
            return True
        if sense == "==" and abs(val - rhs) <= 1e-5:
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
        (2, 3, 5),                     # n_terms
        (2, 3),                        # n_disjuncts
        ("<=", ">=", "=="),            # sense
        ((1.0,), (1.0, -1.0), (0.5, 2.0, -1.5)),  # coefficient patterns
        (False, True),                 # nonlinear body
    )
)

compared = 0
bound_violations: list[str] = []
cert_regressions: list[str] = []
objective_mismatches: list[str] = []
infeasible_points: list[str] = []
recovered = 0   # (case, route) pairs: refusal with the flag OFF -> answer with it ON
off_answers = 0

for n_terms, n_disj, sense, coefs, nonlinear in CASES:
    args = (n_terms, n_disj, sense, coefs, nonlinear)
    tag = f"n={n_terms} d={n_disj} s={sense!r} c={coefs} nl={int(nonlinear)}"
    for route in ROUTES:
        chain_model, names = build(*args, sumover=False)
        chain = solve(chain_model, route)          # the oracle: BinaryOp body
        sum_model, _ = build(*args, sumover=True)
        sumover = solve(sum_model, route)          # the node under test, flag ON

        if "refused" in chain:
            # The chain arm itself declines (e.g. an unbounded big-M on an
            # equality body). Nothing to compare against; not a #1154 case.
            print(f"  [{route}] {tag}: chain arm declines -> skipped", flush=True)
            continue
        compared += 1

        if "refused" in sumover:
            cert_regressions.append(f"{tag} [{route}]: Σ refused but chain answered: {sumover}")
            print(f"  [{route}] {tag}: SUMOVER REFUSED (chain answered)", flush=True)
            continue

        # 1. same status class
        if chain["status"] == "optimal" and sumover["status"] != "optimal":
            cert_regressions.append(f"{tag} [{route}]: chain optimal, Σ {sumover['status']}")

        # 2. same certified objective
        if chain["objective"] is not None and sumover["objective"] is not None:
            if abs(chain["objective"] - sumover["objective"]) > TOL * max(
                1.0, abs(chain["objective"])
            ):
                objective_mismatches.append(
                    f"{tag} [{route}]: chain {chain['objective']} vs Σ {sumover['objective']}"
                )

        # 3. cert-clean: the dual bound may never exceed the oracle's optimum
        ref = chain["objective"]
        for label, res in (("chain", chain), ("sumover", sumover)):
            if res["bound"] is not None and ref is not None:
                if res["bound"] > ref + 1e-5 * max(1.0, abs(ref)):
                    bound_violations.append(
                        f"{tag} [{route}/{label}]: bound {res['bound']} > oracle optimum {ref}"
                    )

        # 4. feasible-point verification of the Σ incumbent, in numpy
        if sumover["x"] is not None and sumover["objective"] is not None:
            xs = [float(sumover["x"][nm]) for nm in names]
            if not all(-1e-6 <= v <= 10.0 + 1e-6 for v in xs):
                infeasible_points.append(f"{tag} [{route}]: {xs} outside the box")
            elif not evaluate_disjunction(xs, n_terms, n_disj, sense, coefs, nonlinear):
                infeasible_points.append(f"{tag} [{route}]: {xs} satisfies no disjunct")
            elif abs(-sum(xs) - sumover["objective"]) > 1e-4 * max(1.0, abs(sumover["objective"])):
                objective_mismatches.append(
                    f"{tag} [{route}]: reported {sumover['objective']} != attained {-sum(xs)}"
                )

        print(
            f"  [{route}] {tag}: chain {chain['status']}/{chain['objective']}/{chain['bound']}"
            f"  Σ {sumover['status']}/{sumover['objective']}/{sumover['bound']}",
            flush=True,
        )

print()
print(f"cases_compared={compared}")
print(f"bound_violations={len(bound_violations)}")
for line in bound_violations:
    print("  BOUND VIOLATION", line)
print(f"certification_regressions={len(cert_regressions)}")
for line in cert_regressions:
    print("  CERT REGRESSION", line)
print(f"objective_mismatches={len(objective_mismatches)}")
for line in objective_mismatches:
    print("  OBJ MISMATCH", line)
print(f"infeasible_incumbents={len(infeasible_points)}")
for line in infeasible_points:
    print("  INFEASIBLE", line)
print(f"executed_comparisons={compared}")

if compared == 0:
    print("PROBE DID NOT FIRE", file=sys.stderr)
    sys.exit(1)
bad = bound_violations or cert_regressions or objective_mismatches or infeasible_points
sys.exit(1 if bad else 0)
