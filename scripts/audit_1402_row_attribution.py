"""Does ``warm_start.check_feasibility`` check every constraint ROW (#1402)?

#1404 closed the two mechanisms the issue names (a swallowed evaluator exception
and NaN failing every strict comparison). It left a third, of the same class --
a verdict of "feasible" reached without examining the rows -- untouched, because
it is not an exception and not a NaN. It is *row attribution*:

``check_feasibility`` walks ``model._constraints`` with ``idx += 1`` per
``Constraint`` OBJECT, while the evaluator emits one row per flat ELEMENT.

  1. An array-valued body is ONE ``Constraint`` and MANY rows -- ``x <= 1`` on a
     3-vector is one object and three rows -- so rows 1..k-1 of every vector
     constraint are never read, and every constraint after the first vector one
     reads the WRONG row.
  2. The evaluator's row set is ``model._constraints`` PLUS
     ``model._builder_linear_constraints()`` (#840). A walk over ``_constraints``
     alone never examines the builder-resident rows at all.

Both were diagnosed and fixed elsewhere by #908, which built
``evaluator.constraint_row_map()`` for exactly this and migrated the two in-solver
incumbent verifiers onto it. ``check_feasibility`` -- the verifier the BENCHMARK
correctness gate uses (``discopt_benchmarks/tests/test_correctness.py:348``) --
was not migrated.

Nothing is swallowed here (CLAUDE.md section 7): every arm lets its own errors
crash. Prints an executed-assertion count; exits 2 if it graded nothing, 1 if the
defect is confirmed, 0 once it is fixed.

Measured on #1404's head (62a1aa03), with #1404's own fix in place: arms 1, 2 and
3 confirmed the defect -- 3 of 7 executed assertions -- and arm 4, the control,
correctly reported the scalar row infeasible. After the fix: 0 of 7, exit 0. Kept
as the entry measurement so that verdict stays reproducible; the permanent guard
is ``python/tests/test_1402_feasibility_row_attribution.py``.
"""

import sys

import discopt.modeling as dm
import numpy as np
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.warm_start import check_feasibility

CHECKS = 0
CONFIRMED = []


# --- arm 1: an array-valued body. One Constraint, three rows. ---------------
m = dm.Model("f1402_vector")
x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
m.subject_to(x <= 1.0)
m.minimize(dm.sum(x))

# row 0 satisfied, rows 1 and 2 violated by 4.0 each. A verifier that reads only
# cons[0] sees the one satisfied row and stops.
pt = np.array([0.5, 5.0, 5.0])
ev = make_evaluator(m)
rows = np.asarray(ev.evaluate_constraints(pt), dtype=float)
CHECKS += 1
print(
    f"arm1 evaluator emits {ev.n_constraints} rows for {len(m._constraints)} Constraint "
    f"object(s); values at the point = {rows!r}"
)
assert ev.n_constraints == 3, ev.n_constraints

ok, viols = check_feasibility(m, pt)
CHECKS += 1
print(f"arm1 check_feasibility(x=[0.5, 5, 5]) -> ok={ok} viols={viols}")
if ok:
    CONFIRMED.append(
        "arm1: a point violating rows 1 and 2 of a size-3 vector constraint BY 4.0 "
        "was reported FEASIBLE -- only row 0 was ever read"
    )

# --- arm 2: desynchronisation. A vector row shifts every later constraint. ---
m2 = dm.Model("f1402_desync")
y = m2.continuous("y", shape=(3,), lb=0.0, ub=10.0)
z = m2.continuous("z", lb=0.0, ub=10.0)
m2.subject_to(y <= 9.0)  # 3 rows, satisfied at the point below
m2.subject_to(z <= 1.0)  # row 3 -- VIOLATED at the point below
m2.minimize(z)

pt2 = np.array([0.0, 0.0, 0.0, 5.0])  # z = 5 > 1
ev2 = make_evaluator(m2)
rows2 = np.asarray(ev2.evaluate_constraints(pt2), dtype=float)
CHECKS += 1
print(
    f"\narm2 evaluator emits {ev2.n_constraints} rows for {len(m2._constraints)} Constraint "
    f"objects; values = {rows2!r}"
)
ok2, viols2 = check_feasibility(m2, pt2)
CHECKS += 1
print(f"arm2 check_feasibility(z=5 against z <= 1) -> ok={ok2} viols={viols2}")
if ok2:
    CONFIRMED.append(
        "arm2: `z <= 1` violated by 4.0 was reported FEASIBLE -- it is evaluator row "
        "3, but the per-object walk read row 1 (a satisfied row of the vector "
        "constraint ahead of it)"
    )

# --- arm 3: builder-resident rows (#840) are not in model._constraints. -----
m3 = dm.Model("f1402_builder")
w = m3.continuous("w", shape=(2,), lb=0.0, ub=10.0)
m3.add_linear_constraints(np.eye(2), w, "<=", np.array([1.0, 1.0]))
m3.minimize(dm.sum(w))

pt3 = np.array([5.0, 5.0])  # both builder rows violated by 4.0
ev3 = make_evaluator(m3)
CHECKS += 1
print(
    f"\narm3 evaluator emits {ev3.n_constraints} row(s); model._constraints holds "
    f"{len(m3._constraints)}; _builder_linear_constraints() holds "
    f"{len(m3._builder_linear_constraints())}"
)
ok3, viols3 = check_feasibility(m3, pt3)
CHECKS += 1
print(f"arm3 check_feasibility(w=[5, 5] against w <= 1) -> ok={ok3} viols={viols3}")
if ok3:
    CONFIRMED.append(
        "arm3: two builder-resident rows each violated by 4.0 were reported FEASIBLE "
        "-- the walk over model._constraints never saw them"
    )

# --- arm 4: control. The scalar case the current walk does handle. ----------
m4 = dm.Model("f1402_scalar_control")
a = m4.continuous("a", lb=0.0, ub=10.0)
m4.subject_to(a <= 1.0)
m4.minimize(a)
ok4, viols4 = check_feasibility(m4, np.array([5.0]))
CHECKS += 1
print(f"\narm4 control (scalar `a <= 1` at a=5) -> ok={ok4} viols={viols4}")
if ok4:
    CONFIRMED.append("arm4: the CONTROL failed -- even the scalar row is not checked")

print(f"\nexecuted assertions: {CHECKS}")
if CHECKS == 0:
    print("FAIL: probe graded nothing")
    sys.exit(2)
for c in CONFIRMED:
    print(f"CONFIRMED  {c}")
print(f"\n{len(CONFIRMED)} of {CHECKS} arms confirm the defect")
sys.exit(1 if CONFIRMED else 0)
