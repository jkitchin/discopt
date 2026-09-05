"""#1154: are the two infeasible hull incumbents PRE-EXISTING, or did #1154 cause them?

The v2 capability panel found two cases where the ``hull`` route returns an
incumbent that satisfies NO disjunct of the original disjunction. Both are in the
``chain`` arm -- the body written as a plain ``BinaryOp`` chain, which is what
``main`` already handles -- so the claim is that they are pre-existing and
unrelated to ``SumOverExpression``. This probe checks that claim directly, with
the #1154 flag OFF so no #1154 code path is reachable at all.

Prints an executed-assertion count and exits non-zero at zero (CLAUDE.md §6).
"""

from __future__ import annotations

import sys

import discopt
import discopt.modeling as dm
import numpy as np

print("sources:", discopt.__file__)
import discopt._relax.gdp_reformulate as _g  # noqa: E402

print("marker _sumover_terms:", open(_g.__file__).read().count("_sumover_terms"))

CASES = [
    (2, 3, "<=", (0.5, 2.0, -1.5), True),
    (3, 2, "<=", (1.0,), True),
]
FEAS_TOL = 1e-5
checks = 0

for n_terms, n_disj, sense, coefs, nonlinear in CASES:
    m = dm.Model("probe")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_terms)]
    rhs = float(n_terms) + 1.0
    disjuncts = []
    for k in range(n_disj):
        scale = 1.0 + k
        parts = [dm.exp(coefs[i % len(coefs)] * scale * x[i] / 10.0) for i in range(n_terms)]
        body = parts[0]
        for p in parts[1:]:
            body = body + p          # the CHAIN arm: no SumOverExpression anywhere
        disjuncts.append([body <= rhs])
    m.either_or(disjuncts)
    m.minimize(-sum(x[i] for i in range(n_terms)))

    r = m.solve(gdp_method="hull", time_limit=15)
    xs = [float(r.x[f"x{i}"]) for i in range(n_terms)]
    sat = []
    for k in range(n_disj):
        scale = 1.0 + k
        val = sum(np.exp(coefs[i % len(coefs)] * scale * xs[i] / 10.0) for i in range(n_terms))
        sat.append(val - rhs)
    feasible = any(v <= FEAS_TOL for v in sat)
    checks += 1
    print(
        f"  n={n_terms} d={n_disj} c={coefs}: status={r.status} obj={r.objective} "
        f"x={xs} residuals={sat} feasible={feasible}"
    )

print(f"executed_assertions={checks}")
if checks == 0:
    sys.exit(1)
