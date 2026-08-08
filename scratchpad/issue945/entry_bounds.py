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
from discopt import Model, solver

# §8: prove which code is loaded, and which arm it is.
# The seed lives at the CALL SITES, not in the backend — #945 settled that the
# split is what the caller consumes, and `solve_nlp` also serves dual consumers.
# An earlier version of this marker tested `solve_nlp`'s own source and so was
# permanently False on the post tree, mislabelling every post run as UNSEEDED.
SEEDED = "pounce_incumbent_options()" in inspect.getsource(solver._solve_continuous)
print(f"# discopt      : {dm.__file__}")
print(f"# nlp_pounce   : {nlp_pounce.__file__}")
print(
    f"# NLP arm      : {'SEEDED (bound_relax_factor=0)' if SEEDED else 'UNSEEDED (Ipopt defaults)'}"
)

checks = 0
violations = 0


def build(flat: bool):
    """Same construction as ``test_940...::test_returned_point_stays_inside_its_declared_box``.

    Two API mistakes made this harness unrunnable as vendored — ``Model.add_variable``
    does not exist and ``solve()`` rejects ``verbose`` (it refuses unknown options
    rather than swallowing them). Mirroring the test is what keeps the probe and the
    thing it claims to measure from drifting apart again.
    """
    m = Model()
    s = m.set("S", [10, 20, 30])
    y = m.continuous("y", lb=1.0, ub=5.0, over=s)
    m.minimize(dm.sum(y.flat) if flat else dm.sum(y))
    return m, y


for flat in (True, False):
    label = "dm.sum(y.flat)" if flat else "dm.sum(y)"
    m, y = build(flat)
    res = m.solve()
    x = np.asarray(res.value(y), dtype=float).ravel()
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
