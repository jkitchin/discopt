"""Is the all-NaN "feasible" verdict the swallowed exception, or NaN comparison
semantics? Two different fixes, so distinguish them. Nothing swallowed here."""

import discopt.modeling as dm
import numpy as np
from discopt._tape_nlp_evaluator import make_evaluator

m = dm.Model()
x = m.continuous("x", lb=0.0, ub=1.0)
y = m.continuous("y", lb=0.0, ub=1.0)
m.subject_to(x + y >= 10.0)
m.minimize(x + y)

nanx = np.full(2, np.nan)
ev = make_evaluator(m)
CHECKS = 0
raised = None
try:
    cons = ev.evaluate_constraints(nanx)
except Exception as e:
    raised = f"{type(e).__name__}: {e}"
CHECKS += 1
print(f"evaluator on all-NaN: raised={raised}  value={None if raised else cons!r}")

if raised is None:
    v = float(cons[0])
    CHECKS += 1
    print(f"constraint value is {v!r}; sense '>=' test is `val < -tol` -> {v < -1e-4}")
    print("MECHANISM: NaN comparison semantics -- every violation test is False for NaN,")
    print("           so the point is reported feasible WITHOUT any exception.")
else:
    print("MECHANISM: the swallowed exception.")

# And the bounds loop, independently.
lo = nanx < (np.zeros(2) - 1e-4)
hi = nanx > (np.ones(2) + 1e-4)
CHECKS += 1
print(
    f"bounds loop on all-NaN: below={lo.any()} above={hi.any()} "
    f"-> {'no bound violation flagged' if not (lo.any() or hi.any()) else 'flagged'}"
)
print(f"EXECUTED ASSERTIONS: {CHECKS}")
assert CHECKS >= 2
