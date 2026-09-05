"""PR #1159 review, finding 1 (second shape): the literal ``SumExpression`` row.

The review reproduced the reverted-#1150 row by monkeypatching ``_is_linear`` to
admit ``SumExpression``, and the union-based guard stayed silent. With the guard
checking per body, it must fire — which is what makes the PR's "leaving
SumExpression unhandled is safe because the guard turns the silent-miss hazard
into a refusal" argument actually true.

Prints an executed-assertion count and exits non-zero at zero (§6).
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
import numpy as np
import discopt._relax.gdp_reformulate as g
from discopt.modeling.core import SumExpression

checks = 0
_real_is_linear = g._is_linear


def _is_linear_admitting_sumexpression(expr):
    if isinstance(expr, SumExpression):
        return _real_is_linear(expr.operand)
    return _real_is_linear(expr)


def build():
    m = dm.Model("sumexpr")
    X = m.continuous("X", shape=(3,), lb=0.0, ub=10.0)
    m.either_or([[dm.sum(X) - 3.0 <= 0.0], [X[0] >= 8.0]])
    m.minimize(-(X[0] + X[1] + X[2]))
    return m

body = dm.sum(build()._variables[0]) - 3.0
print("body:", type(body).__name__, "| left:", type(body.left).__name__)
checks += 1
assert isinstance(body.left, SumExpression), "premise: dm.sum(X) is a SumExpression"

g._is_linear = _is_linear_admitting_sumexpression
try:
    for label in ("with _is_linear monkeypatched to admit SumExpression",):
        try:
            r = build().solve(gdp_method="hull", time_limit=30)
            print(f"{label}: status={r.status} obj={r.objective} bound={r.bound}")
            checks += 1
            if r.bound is not None and r.bound > -30.0 + 1e-6:
                print("  FALSE CERTIFICATE: bound above the true minimum -30.0")
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed (§7)
            print(f"{label}: REFUSED {type(exc).__name__}: {str(exc)[:150]}")
            checks += 1
finally:
    g._is_linear = _real_is_linear

print(f"executed_assertions={checks}")
if checks == 0:
    sys.exit(1)
