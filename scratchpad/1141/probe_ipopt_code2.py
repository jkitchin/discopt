"""Entry experiment for #1141 item 2: is Ipopt code 2 (`Infeasible_Problem_Detected`)
a GLOBAL infeasibility proof?

If it is, mapping it to `SolveStatus.INFEASIBLE` is free pruning power. If a
NONCONVEX but FEASIBLE model can produce it, the map must stay conservative on the
general path and the verdict may only be trusted where convexity is certified.

Prints an executed-check count (§6).
"""
import sys
import numpy as np
import discopt.modeling as dm
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solvers.nlp_pounce import solve_nlp
from discopt.solvers import SolveStatus

CASES = []

# 1. x² >= 1 on [-2, 2] started at 0: feasible (x = ±1), a local minimum of the
#    constraint violation sits exactly at the start.
m = dm.Model("disconnected")
x = m.continuous("x", lb=-2.0, ub=2.0)
m.subject_to(x * x >= 1.0)
m.minimize(x)
CASES.append(("x^2>=1 from x=0", m, np.array([0.0]), True))

# 2. Two disconnected rings.
m2 = dm.Model("annulus")
y = m2.continuous("y", lb=-3.0, ub=3.0)
z = m2.continuous("z", lb=-3.0, ub=3.0)
m2.subject_to(y * y + z * z >= 4.0)
m2.subject_to(y * y + z * z <= 9.0)
m2.minimize(y + z)
CASES.append(("annulus from origin", m2, np.array([0.0, 0.0]), True))

# 3. Genuinely infeasible, as a control: the probe must be able to see code 2 at all.
m3 = dm.Model("really_infeasible")
w = m3.continuous("w", lb=0.0, ub=1.0)
m3.subject_to(w >= 2.0)
m3.minimize(w)
CASES.append(("w>=2 with w<=1", m3, np.array([0.5]), False))

checks = 0
false_infeasible = 0
for name, model, x0, feasible in CASES:
    ev = make_evaluator(model)
    r = solve_nlp(ev, x0, options={"max_iter": 500, "print_level": 0})
    checks += 1
    print(f"{name:28s} feasible={feasible!s:5s} status={r.status} obj={r.objective!r} x={r.x}")
    if feasible and r.status in (SolveStatus.ERROR, SolveStatus.INFEASIBLE):
        false_infeasible += 1
        print("   ^ a FEASIBLE nonconvex model produced a local infeasibility verdict")

print(f"\nEXECUTED CHECKS: {checks}   FEASIBLE-BUT-DECLARED-INFEASIBLE: {false_infeasible}")
if checks == 0:
    sys.exit(1)

# --- a constructed strict local minimum of the constraint violation ---------
# g(x) = (x²−1)(x²−4) = x⁴ − 5x² + 4 has a strict local MINIMUM at x = 0 with
# g(0) = 4 > 0, while {1 ≤ |x| ≤ 2} is feasible. Restoration started at 0 has
# nowhere downhill to go, which is precisely the state Ipopt reports as code 2 —
# on a model that is feasible.
print("\n--- constructed local-violation-minimum case ---")
m4 = dm.Model("quartic_valley")
v = m4.continuous("v", lb=-3.0, ub=3.0)
m4.subject_to(v**4 - 5.0 * v**2 + 4.0 <= 0.0)
m4.minimize(v)
ev4 = make_evaluator(m4)
extra = 0
for x0 in (0.0, 1e-3, -1e-3, 0.05):
    r = solve_nlp(ev4, np.array([x0]), options={"max_iter": 500, "print_level": 0})
    ok = r.x is not None and (float(r.x[0]) ** 4 - 5 * float(r.x[0]) ** 2 + 4.0) <= 1e-6
    print(f"  start={x0:<8g} status={r.status} x={r.x} primal_feasible={ok}")
    if not ok:
        extra += 1
print(f"CONSTRUCTED CASES DECLARED NON-OPTIMAL ON A FEASIBLE MODEL: {extra}/4")
