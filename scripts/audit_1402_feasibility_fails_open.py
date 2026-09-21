"""Does warm_start.check_feasibility report "feasible" for a point it never checked?

`check_feasibility` (python/discopt/warm_start.py:215) wraps the entire constraint
arm in `try: ... except Exception as e: logger.debug(...)` and then returns
`len(violations) == 0`. If the evaluator raises, `violations` is whatever the
BOUNDS loop found -- so a point that satisfies its bounds but violates every
constraint returns `(True, [])`.

Reach: this is the independent verifier in the benchmark correctness gate,
discopt_benchmarks/tests/test_correctness.py:348, whose own comment says "the
point of the test is to catch a solver that believes an infeasible point".

No exception is swallowed HERE (CLAUDE.md section 7): every probe arm lets its own
errors crash. Prints an executed-assertion count; exits 2 if it graded nothing,
1 if the defect is confirmed.
AFTER THE FIX THIS PROBE EXITS 0. Before it, on ``main`` @ 7910df37, arms 2 and 3
confirmed the defect (2 of 4 arms). It is kept as the entry measurement, so the
verdict it recorded stays reproducible; the permanent guard is
``python/tests/test_1402_feasibility_fails_closed.py``.
"""

import sys

import discopt.modeling as dm
import numpy as np
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.warm_start import check_feasibility

CHECKS = 0
CONFIRMED = []


def build():
    """x + y >= 10 with x,y in [0,1] -- INFEASIBLE at any point in the box."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + y >= 10.0)
    m.minimize(x + y)
    return m


m = build()
n = sum(v.size for v in m._variables)
print(f"model: {n} variables, {len(m._constraints)} constraints")

# --- arm 1: control. A correct-length point must be reported INFEASIBLE. ---
good = np.zeros(n)
ok, viols = check_feasibility(m, good)
CHECKS += 1
print(f"  arm1 correct-length x=zeros: ok={ok} viols={viols}")
if ok:
    CONFIRMED.append("arm1: an x violating `x + y >= 10` by 10.0 was reported feasible")

# --- arm 2: the swallow. A short vector makes evaluate_constraints raise. ---
short = np.zeros(n - 1)
ok2, viols2 = check_feasibility(m, short)
CHECKS += 1
print(f"  arm2 short vector (len {n - 1} for {n} vars): ok={ok2} viols={viols2}")
if ok2:
    CONFIRMED.append(
        f"arm2: a length-{n - 1} vector for a {n}-variable model was reported "
        "FEASIBLE -- the constraint arm raised and the exception was swallowed"
    )

# --- arm 3: NaN. A point that cannot be judged at all. ---
nanx = np.full(n, np.nan)
ok3, viols3 = check_feasibility(m, nanx)
CHECKS += 1
print(f"  arm3 all-NaN point: ok={ok3} viols={viols3}")
if ok3:
    CONFIRMED.append("arm3: an all-NaN point was reported FEASIBLE")

# --- arm 4: prove the raise is real, by calling the evaluator directly. ---
ev = make_evaluator(m)
raised = None
try:
    ev.evaluate_constraints(short)
except Exception as e:  # noqa: BLE001 - deliberately reporting, then re-raising nothing
    raised = f"{type(e).__name__}: {e}"
CHECKS += 1
print(f"  arm4 evaluator on the short vector raised: {raised}")
if raised is None:
    print("  arm4 NOTE: the evaluator did NOT raise; arm2's verdict needs another trigger")

print(f"\nexecuted assertions: {CHECKS}")
if CHECKS == 0:
    print("FAIL: probe graded nothing")
    sys.exit(2)
for c in CONFIRMED:
    print(f"CONFIRMED  {c}")
print(f"\n{len(CONFIRMED)} of {CHECKS} arms confirm the defect")
sys.exit(1 if CONFIRMED else 0)
