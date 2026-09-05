"""PR #1159 review, finding 1: does the union-based coverage guard miss a
PER-BODY collector miss?

The claim: ``_assert_hull_saw_every_variable`` compares the independent walker's
leaves over ALL disjunct bodies against ``all_vars``, itself the union of
``_collect_variables`` over ALL bodies. A variable missed in one body is then
invisible whenever any other body in the same disjunction collects it.

Repro from the review: ``(A @ X)[0]`` is an ``IndexExpression`` whose base is a
``MatMulExpression``. ``_is_linear`` returns True for IndexExpression without
inspecting its base; ``_collect_variables`` returns {} for it. ``X`` is still in
``all_vars`` via the other arm's ``X[0] >= 8.0``, so the guard stays silent.

Prints an executed-assertion count and exits non-zero at zero (CLAUDE.md §6).
"""

from __future__ import annotations

import sys

import discopt
import discopt.modeling as dm
import numpy as np
from discopt._relax.gdp_reformulate import _collect_variables, _is_linear

print("sources:", discopt.__file__)
import discopt._relax.gdp_reformulate as _g  # noqa: E402

src = open(_g.__file__).read()
print("marker _sumover_terms:", src.count("_sumover_terms"))
print("marker HullVariableCoverageError:", src.count("HullVariableCoverageError"))

TRUE_OPT = -30.0
checks = 0


def build():
    m = dm.Model("ix")
    X = m.continuous("X", shape=(3,), lb=0.0, ub=10.0)
    A = np.array([[1.0, 1.0, 1.0]])
    m.either_or([[(A @ X)[0] - 3.0 <= 0.0], [X[0] >= 8.0]])
    m.minimize(-(X[0] + X[1] + X[2]))
    return m


# --- the two walkers disagree on this body, which is the whole defect ---------
m0 = build()
X0 = m0._variables[0]
A0 = np.array([[1.0, 1.0, 1.0]])
body = (A0 @ X0)[0] - 3.0
print("body node:", type(body).__name__)
print("  _is_linear      ->", _is_linear(body))
print("  _collect_variables ->", sorted(_collect_variables(body)))
checks += 1
assert _is_linear(body) is True, "premise: the linear route is taken for this body"
checks += 1
assert not _collect_variables(body), "premise: the collector sees no variable here"
checks += 1

# --- and hull answers instead of refusing ------------------------------------
try:
    r = build().solve(gdp_method="hull", time_limit=30)
    print(f"hull -> status={r.status} obj={r.objective} bound={r.bound}")
    checks += 1
    if r.bound is not None and r.bound > TRUE_OPT + 1e-6:
        print(f"  CONFIRMED: dual bound {r.bound} is ABOVE the true minimum {TRUE_OPT}")
    else:
        print("  NOT reproduced: the bound is sound")
except Exception as exc:  # noqa: BLE001 - reported, never swallowed (§7)
    print(f"hull REFUSED: {type(exc).__name__}: {exc}")
    checks += 1

print(f"executed_assertions={checks}")
if checks == 0:
    sys.exit(1)
