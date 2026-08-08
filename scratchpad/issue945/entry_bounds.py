"""Entry experiment for #945(a): does the NLP path return points outside the box?

Reproduces the issue's evidence table directly. Prints an executed-assertion
count and exits non-zero if nothing was actually measured (CLAUDE.md §6).
"""

from __future__ import annotations

import inspect
import sys

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
import numpy as np
from discopt import Model

# §8: prove which code is loaded, and which arm it is.
SEEDED = "opts = pounce_option_defaults()" in inspect.getsource(nlp_pounce.solve_nlp)
print(f"# discopt      : {dm.__file__}")
print(f"# nlp_pounce   : {nlp_pounce.__file__}")
print(f"# NLP arm      : {'SEEDED (bound_relax_factor=0)' if SEEDED else 'UNSEEDED (Ipopt defaults)'}")

checks = 0
violations = 0


def build(flat: bool) -> Model:
    m = Model()
    y = m.add_variable("y", shape=(3,), lb=1.0, ub=5.0)
    m.minimize(dm.sum(y.flat) if flat else dm.sum(y))
    return m


for flat in (True, False):
    label = "dm.sum(y.flat)" if flat else "dm.sum(y)"
    m = build(flat)
    res = m.solve(verbose=False)
    x = np.asarray(res.x["y"], dtype=float).ravel()
    below = float(np.max(1.0 - x))  # >0 means below lb=1
    above = float(np.max(x - 5.0))
    checks += 2
    worst = max(below, above)
    bad = worst > 0.0
    violations += int(bad)
    # Super-optimality: the true optimum is exactly 3.0.
    checks += 1
    super_opt = 3.0 - float(res.objective)
    if super_opt > 0.0:
        violations += 1
    print(
        f"{label:16s} obj={res.objective!r:22s} "
        f"max_outside={worst:.3e} super_optimal_by={super_opt:.3e} "
        f"{'BAD' if bad or super_opt > 0 else 'ok'}"
    )

print(f"\nexecuted_assertions={checks} violations={violations}")
if checks == 0:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(2)
