"""Entry experiment for validation/feasibility.py's FEASIBLE_DISTANCE_TOL (#1397).

This probe now EXITS 0: the fix landed with this commit, and the pytest
regression that pins the class is python/tests/test_1397_feasible_distance_tol_scale.py.
Run it against an older tree (pass that tree's root as argv[1]) to reproduce the
pre-fix measurement, which was 15 of 15 inflated verdicts and exit 1.

Claim under test: in ``improving_gradient_norms`` the round-off allowance
``slack = 16*_EPS`` is RELATIVE (it multiplies ``|ub| + |x|``) but the divisor
``FEASIBLE_DISTANCE_TOL`` is ABSOLUTE, so a column pinned exactly ON the bound
that blocks its improving direction -- true room 0 -- is credited with phantom
room that grows linearly with |bound|.

FIRST ATTEMPT WAS A NO-OP, RETRACTED (CLAUDE.md section 6): the first version of
this probe used |J| = 1 for both columns, so ``plain = max_j |J_ij|`` = 1 equalled
the correct answer and the final ``np.minimum(contrib.sum(), plain)`` clamped both
arms to 1.0 at every scale. It printed "NOT CONFIRMED" and graded nothing.

The clip is the structural fact that decides what the defect IS. ``contrib`` can
never exceed ``plain``, so phantom room cannot invent a cap out of nothing -- it
can only walk the result back UP to the plain sup-norm, i.e. silently undo the
#1284 tightening. #1284's docstring records that the untightened cap certified a
point 0.87 away in ``y`` at ``z = -20`` against a true optimum of -6.699. So
"the tightening degrades to a no-op" IS the unsound path, and that is what is
graded here.

Construction, oracle exact by hand:
  one row, body must DECREASE (direction = +1)
  col 0: partial -W (large), sitting EXACTLY on its upper bound B.
         improving move is UP (step = -1*sign(-W) = +1), which is blocked.
         TRUE room = 0, so its true contribution is 0.
  col 1: partial +1, interior at 0.5 with lb 0, room 0.5 >> tol, contributes 1.0.
  => the true improving gradient norm is exactly 1.0, at every B.
     plain = W. So any returned value above 1.0 is phantom, and W is the
     ceiling at which #1284 has been fully erased.

Prints an executed-comparison count and exits 2 if it graded nothing. Exits 1 if
the scale-dependence is confirmed: this is an entry experiment whose "failure" is
the defect being real.
"""

import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, os.path.join(TREE, "python"))

import numpy as np  # noqa: E402
from discopt.validation import feasibility as F  # noqa: E402

assert F.__file__.startswith(TREE), f"loaded {F.__file__}, expected under {TREE}"
print(f"LOAD GATE ok: {F.__file__}")
TOL = F.FEASIBLE_DISTANCE_TOL
print(f"FEASIBLE_DISTANCE_TOL = {TOL:.3e}   SMALL_ROW_ABS_FLOOR = {F.SMALL_ROW_ABS_FLOOR:.3e}")

EPS = float(np.finfo(np.float64).eps)
W = 1000.0  # |J| of the BLOCKED column; also `plain`, the ceiling on the result
TRUE_NORM = 1.0  # exact: only the interior column can help, and it saturates

CHECKS = 0
rows = []
INFLATED = []

for exponent in range(0, 15):
    B = 10.0**exponent
    J = np.array([[-W, 1.0]])
    x = np.array([B, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([B, 1.0])
    got = float(F.improving_gradient_norms(J, x, lb, ub, np.array([1.0]))[0])

    plain = float(F.jacobian_row_gradient_norms(J)[0])
    phantom_room = 16.0 * EPS * (abs(B) + abs(B))
    # What the function CLAIMS a best move of size TOL can remove from the body,
    # vs what it actually can (only the interior column moves).
    claimed_reduction = TOL * got
    true_reduction = TOL * TRUE_NORM

    CHECKS += 1
    rows.append((B, phantom_room, got, plain, claimed_reduction, true_reduction))
    if got > TRUE_NORM * (1.0 + 1e-9):
        INFLATED.append((B, got, got / TRUE_NORM))

print(f"\nrow: [-{W:.0f}, +1] * x,  body must decrease,  col0 pinned on ub=B (blocked),")
print(f"     col1 interior.  TRUE improving grad norm = {TRUE_NORM:.1f} at every B;")
print(f"     plain sup-norm = {W:.0f} is the ceiling the clip imposes.\n")
print(
    f"{'B (=ub0)':>10} {'phantom room':>14} {'returned':>12} {'true':>8} "
    f"{'inflation':>11} {'claimed dv':>12} {'true dv':>10}"
)
for B, ph, got, plain, cr, tr in rows:
    print(
        f"{B:10.0e} {ph:14.3e} {got:12.4f} {TRUE_NORM:8.1f} "
        f"{got / TRUE_NORM:10.1f}x {cr:12.3e} {tr:10.3e}"
    )

print(f"\nexecuted comparisons: {CHECKS}")
if CHECKS == 0:
    print("FAIL: probe graded nothing")
    sys.exit(2)

sat = TOL / (32.0 * EPS)
print("\nphantom room reaches FEASIBLE_DISTANCE_TOL (frac saturates at 1, blocked")
print(f"column credited with its FULL |J|) at B >= {sat:.3e}")
print(f"inflated verdicts: {len(INFLATED)} of {CHECKS}")
if INFLATED:
    worst = max(INFLATED, key=lambda r: r[2])
    print(
        f"WORST: at B={worst[0]:.0e} the returned norm is {worst[1]:.4f} vs a true "
        f"{TRUE_NORM:.1f} ({worst[2]:.1f}x)"
    )
    print(
        f"       => cap {TOL * worst[1]:.3e} instead of {TOL * TRUE_NORM:.3e}: a violation "
        f"up to {TOL * worst[1]:.3e} is certified feasible"
    )
    full = [r for r in INFLATED if r[1] >= W * (1.0 - 1e-9)]
    if full:
        print(
            f"       => at B >= {min(r[0] for r in full):.0e} the returned value EQUALS the "
            f"plain sup-norm {W:.0f}: #1284's tightening is a silent no-op"
        )

confirmed = bool(INFLATED)
print(
    f"\n{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}: a relative allowance over an "
    f"absolute divisor makes the #1284 tightening scale-dependent"
)
sys.exit(1 if confirmed else 0)
