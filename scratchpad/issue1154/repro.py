"""#1154 entry experiment: what the three GDP routes do with a SumOverExpression
disjunct body TODAY (before any change).

Prints an executed-assertion count (CLAUDE.md §6) and exits non-zero at zero.
"""
import sys
import traceback

import discopt
import discopt.modeling as dm
from discopt import Model
from discopt._relax import gdp_reformulate

print("sources:", discopt.__file__)
print("gdp_reformulate:", gdp_reformulate.__file__)
# §8 marker: is the widening present in the loaded sources?
src = open(gdp_reformulate.__file__).read()
print("marker _is_linear/SumOverExpression:", src.count("SumOverExpression"))

TRUE_OPT = -30.0
checks = 0


def build():
    m = Model("sumover_disjunct")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
    m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]])
    m.minimize(-(x[0] + x[1] + x[2]))
    return m


for method in ("auto", "big-m", "hull"):
    m = build()
    try:
        r = m.solve(gdp_method=method, time_limit=30)
        print(f"[{method}] status={r.status} obj={r.objective} bound={r.bound}")
        checks += 1
        if r.bound is not None:
            # a dual bound above the true minimum is invalid
            assert r.bound <= TRUE_OPT + 1e-6, (
                f"[{method}] INVALID BOUND {r.bound} > true optimum {TRUE_OPT}"
            )
            checks += 1
    except Exception as exc:  # noqa: BLE001 - we *report*, never swallow (§7)
        print(f"[{method}] REFUSED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        checks += 1

print(f"executed_assertions={checks}")
if checks == 0:
    sys.exit(1)
