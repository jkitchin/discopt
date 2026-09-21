#!/usr/bin/env python3
"""#1397 entry experiment: is ``factorable_reform._ZERO_MARGIN`` scale-exposed?

``_find_clearable_denominator`` licenses multiplying a whole constraint through by a
denominator ``D`` once it believes ``D`` is sign-definite over the box, and it decides
that with ``lo > _ZERO_MARGIN`` where ``_ZERO_MARGIN = 1e-9`` and ``lo`` comes from
``_bound_expression`` -- plain float interval arithmetic with **no outward rounding**.
So ``lo`` is not a rigorous under-estimate of the true infimum: it carries an error of
order ``n * u * sum|terms|``. At ``sum|terms| = 1e17`` one ulp is 16.0, ten orders of
magnitude past the whole margin.

Multiplying through by a ``D`` that can be negative is not a conservative mistake: it
**flips the inequality** on the part of the box where ``D < 0``, so the rewritten model
has a different feasible set than the one the user wrote. That is a false-optimal /
false-infeasible generator, not a loss of tightening strength.

The probe builds denominators whose float-folded lower bound clears 1e-9 while their
*exact* infimum is negative, and asks the gate. Arm B is the no-weakening arm: a
genuinely sign-definite denominator must still be cleared, or the fix has simply
switched the pass off.

Per CLAUDE.md §6 this prints an executed-assertion count and exits non-zero when it is
zero. Per §8 it prints the loaded file and whether the fix marker is present.
"""

from __future__ import annotations

import sys

import numpy as np

import discopt.modeling as dm
from discopt._relax import factorable_reform as fr

MARKER = "_denominator_sign_slack"


def _load_gate() -> None:
    print(f"file   : {fr.__file__}")
    src = open(fr.__file__).read()
    print(f"marker : {'PRESENT' if MARKER in src else 'ABSENT'} ({MARKER})")
    print(f"_ZERO_MARGIN = {fr._ZERO_MARGIN!r}")


def _cancelling_denominator(y, magnitude: float, residue: float):
    """``y + M - M + residue``: exact infimum ``y.lb + residue``, float-folded
    lower bound ``residue`` once ``y.lb`` is absorbed by ``M``."""
    return y + magnitude - magnitude + residue


def main() -> int:
    _load_gate()
    executed = 0
    arm_a_bad = 0
    arm_b_bad = 0

    print("\n-- Arm A: exact infimum NEGATIVE, float-folded lower bound above the margin --")
    for magnitude in (1e15, 1e16, 1e17, 1e18):
        for ylb in (-0.5, -8.0):
            m = dm.Model()
            y = m.continuous("y", lb=ylb, ub=1.0)
            x = m.continuous("x", lb=1.0, ub=2.0)
            d = _cancelling_denominator(y, magnitude, 1e-8)
            lo, hi = fr._bound_expression(d, m)
            exact_inf = ylb + 1e-8
            if not (lo > fr._ZERO_MARGIN and exact_inf < 0.0):
                # The construction did not land; do not count it as a pass.
                print(f"   M={magnitude:.0e} ylb={ylb:<6} SKIP (lo={lo:.3e} exact={exact_inf:.3e})")
                continue
            body = x / d - 1.0
            found = fr._find_clearable_denominator(body, m)
            executed += 1
            cleared = found is not None
            if cleared:
                arm_a_bad += 1
            print(
                f"   M={magnitude:.0e} ylb={ylb:<6} lo={lo:.3e} exact_inf={exact_inf:.3e} "
                f"-> cleared={cleared}{'  <-- UNSOUND' if cleared else ''}"
            )

    print("\n-- Arm B (no weakening): genuinely sign-definite denominator must still clear --")
    for dlb, dub in ((1.0, 5.0), (0.5, 100.0), (-9.0, -0.25), (1e-3, 1.0)):
        m = dm.Model()
        d_var = m.continuous("d", lb=dlb, ub=dub)
        x = m.continuous("x", lb=1.0, ub=2.0)
        body = x / d_var - 1.0
        found = fr._find_clearable_denominator(body, m)
        executed += 1
        if found is None:
            arm_b_bad += 1
        print(
            f"   D in [{dlb:>8}, {dub:>8}] -> cleared={found is not None}"
            f"{'  <-- LOST STRENGTH' if found is None else ''}"
        )

    print("\n-- Arm B2: a large-magnitude but genuinely definite denominator still clears --")
    # RETRACTION (CLAUDE.md §11): an earlier revision of this arm used ``y in [1, 2]``
    # with the comment "1.0 is far above 8*u*(2*M) even at M = 1e18". That is false --
    # the slack at M = 1e15 is already 1.78 -- and the arm reported a spurious
    # "LOST STRENGTH". Refusing there is the *correct* verdict: a lower bound of 1.0
    # read off a 1e15-magnitude float fold genuinely cannot be certified positive.
    # The arm now uses a margin that really does dominate its own round-off.
    for magnitude in (1e15, 1e17):
        m = dm.Model()
        y = m.continuous("y", lb=1e6, ub=2e6)
        x = m.continuous("x", lb=1.0, ub=2.0)
        # 1e6 vs. a slack of ~8*u*2*M = 355 at M = 1e17: definite with room to spare.
        d = _cancelling_denominator(y, magnitude, 0.0)
        body = x / d - 1.0
        found = fr._find_clearable_denominator(body, m)
        executed += 1
        if found is None:
            arm_b_bad += 1
        print(
            f"   M={magnitude:.0e} D in [1e6, 2e6] -> cleared={found is not None}"
            f"{'  <-- LOST STRENGTH' if found is None else ''}"
        )

    print(f"\nEXECUTED ASSERTIONS: {executed}")
    if executed == 0:
        print("FAIL: the probe asserted nothing (CLAUDE.md §6)")
        return 2
    print(f"Arm A  unsound clears        : {arm_a_bad}")
    print(f"Arm B  lost clears           : {arm_b_bad}")
    ok = arm_a_bad == 0 and arm_b_bad == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    assert np.isfinite(fr._ZERO_MARGIN)
    sys.exit(main())
