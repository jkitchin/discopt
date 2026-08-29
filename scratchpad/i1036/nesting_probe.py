"""Is a larger-budget run a continuation of a smaller-budget one? (issue #1036)

The panel test's stated rationale -- "the incumbent first reached the tolerance at
evaluation k, so a budget of B > k has headroom" -- is only valid if the run at
budget B evaluates the same points, in the same order, as the run at budget k.
This probe checks exactly that, and counts its comparisons (CLAUDE.md §6).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "tests"))
import numpy as np
import discopt.solvers.surrogate as S
from support import direct_testfuncs as tfs
print("loaded:", S.__file__)

BUDGETS = [40, 46, 60, 80, 100]
compared = violations = 0
for name in ("branin", "hartman_3", "hartman_6"):
    tf = tfs.get(name)
    traces = {}
    for b in BUDGETS:
        tr: list[float] = []
        model, _ = tfs.build_model(tf)
        S.solve_surrogate(model, max_evals=b, time_limit=600.0, seed=0,
                          acquisition_optimizer="multistart",
                          on_evaluation=lambda k, v, _t=tr: _t.append(v))
        assert tr, f"on_evaluation never fired for {name}@{b}"
        traces[b] = tr
    for i, b1 in enumerate(BUDGETS):
        for b2 in BUDGETS[i + 1:]:
            k = min(len(traces[b1]), len(traces[b2]))
            a = [v for v in traces[b1][:k]]
            c = [v for v in traces[b2][:k]]
            compared += 1
            same = all(
                (x is None and y is None) or (x is not None and y is not None and x == y)
                for x, y in zip(a, c)
            )
            if not same:
                violations += 1
                j = next(i for i, (x, y) in enumerate(zip(a, c)) if x != y)
                print(f"  {name}: budget {b1} vs {b2} DIVERGE at evaluation {j + 1} "
                      f"({a[j]} vs {c[j]})")
print(f"compared {compared} budget pairs; {violations} diverged")
if compared == 0:
    sys.exit(1)
sys.exit(2 if violations else 0)
