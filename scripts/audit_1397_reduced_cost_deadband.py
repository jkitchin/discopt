#!/usr/bin/env python3
"""#1397 entry experiment: are the reduced-cost deadbands scale-exposed?

Both reduced-cost-fixing sites gate on an absolute deadband
(``node_reduce._RC_TOL = 1e-7``, ``solver._RCF_RC_TOL = 1e-7``) and then **divide**
the optimality gap by ``|d_j|``:

    x_j <= lb_j + floor(gap / d_j)

A reduced cost is a *difference* -- ``c_j - (A^T y)_j`` at the simplex site,
``mult_x_L - mult_x_U`` at the POUNCE site -- so its own magnitude says nothing about
how much of it is the round-off of the arithmetic that produced it. Two terms of
magnitude ``M`` differ by one ulp at ``2**-52 * M``: at ``M = 1e9`` that is 2.2e-7,
already past the deadband. The unsound direction is *not* conservative: an over-stated
``|d_j|`` makes ``gap / |d_j|`` too small, so the fix is too tight and can remove the
true optimum from the box -- a false ``optimal`` certificate.

Arm A: a reduced cost that is pure round-off at its own scale must yield no fix.
Arm B (no weakening): an honest, well-scaled reduced cost must still fix exactly as
far as it does today -- deflating by a slack of ~1e-16 relative must not move a
``floor()`` that today lands well inside its integer.

Per CLAUDE.md §6 this prints an executed-assertion count and exits non-zero when it is
zero. Per §8 it prints the loaded files and whether the fix marker is present.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np
from discopt import solver as sol
from discopt._relax import node_reduce as nr

# Mirrored here, not imported, so this probe also runs against a baseline tree whose
# ``_numeric`` predates them (CLAUDE.md §8: a baseline arm that dies on an ImportError
# is a baseline that was never measured). The load gate below reports whether the
# loaded ``_numeric`` exports them, which is the fix marker.
FLOAT_EPS = float(np.finfo(np.float64).eps)
ROUNDOFF_OPS = 8.0


def _call(fn, *args, **kwargs):
    """Call *fn*, dropping any kwarg its signature does not accept.

    The pre-fix versions of both entry points take no ``rc_absum``; dropping it lets
    the SAME probe run on the baseline tree and report what today's absolute deadband
    does, instead of dying with a TypeError and leaving the baseline unmeasured.
    Not an exception swallow (CLAUDE.md §7): the decision is made from the signature,
    every other error still propagates, and the load gate reports which arm ran.
    """
    accepted = inspect.signature(fn).parameters
    return fn(*args, **{k: v for k, v in kwargs.items() if k in accepted})


MARKERS = {
    "discopt/solver.py": (sol.__file__, "rc_absum"),
    "discopt/_relax/node_reduce.py": (nr.__file__, "rc_absum"),
}


def _load_gate() -> bool:
    from discopt._relax import _numeric

    print(f"file   : {_numeric.__file__}")
    exports = hasattr(_numeric, "roundoff_slack") and hasattr(_numeric, "FLOAT_EPS")
    print(f"marker : {'PRESENT' if exports else 'ABSENT'} ('roundoff_slack' in _numeric)")
    ok = exports
    for label, (path, marker) in MARKERS.items():
        src = open(path).read()
        present = marker in src
        print(f"file   : {path}")
        print(f"marker : {'PRESENT' if present else 'ABSENT'} ({marker!r} in {label})")
        ok = ok and present
    print(f"_RCF_RC_TOL = {sol._RCF_RC_TOL!r}   _RC_TOL = {nr._RC_TOL!r}")
    print(f"ROUNDOFF_OPS * FLOAT_EPS = {ROUNDOFF_OPS * FLOAT_EPS:.3e}")
    return ok


def main() -> int:
    marker_ok = _load_gate()
    executed = 0
    unsound = 0
    weakened = 0

    # ---------------------------------------------------------------- Arm A (root)
    # One integer column, gap = 1.0. Its reduced cost is the difference of two bound
    # multipliers of magnitude M: entirely round-off once M is large enough.
    print("\n-- Arm A (root RCF): a reduced cost that is round-off at its own scale --")
    for mag in (1e8, 1e9, 1e11, 1e14):
        d = np.array([4.0 * FLOAT_EPS * mag])  # ~ a few ulps of M
        absum = np.array([2.0 * mag])
        lb = np.array([0.0])
        ub = np.array([1e12])
        gap_z_lp, gap_z_inc = 0.0, 1.0
        new_lb, new_ub, n = _call(
            sol._reduced_cost_fixing, lb, ub, [0], d, gap_z_lp, gap_z_inc, rc_absum=absum
        )
        executed += 1
        # today's (absolute-deadband) verdict, for contrast
        bad = n != 0
        if bad:
            unsound += 1
        print(
            f"   M={mag:.0e}  d={d[0]:.3e} (absum={absum[0]:.0e})  "
            f"-> changes={n} ub={new_ub[0]:g}{'  <-- UNSOUND FIX' if bad else ''}"
        )

    # ---------------------------------------------------------------- Arm A (node)
    print("\n-- Arm A (node DBBT): same construction through _dbbt_from_reduced_costs --")
    for mag in (1e8, 1e9, 1e11, 1e14):
        rc = np.array([4.0 * FLOAT_EPS * mag])
        absum = np.array([2.0 * mag])
        lb = np.array([0.0])
        ub = np.array([1e12])
        is_int = np.array([True])
        nlb, nub, _nt, _inf = _call(
            nr._dbbt_from_reduced_costs, lb, ub, rc, 0.0, 1.0, is_int, rc_absum=absum
        )
        executed += 1
        bad = float(nub[0]) < float(ub[0]) - 0.5 or float(nlb[0]) > float(lb[0]) + 0.5
        if bad:
            unsound += 1
        print(
            f"   M={mag:.0e}  d={rc[0]:.3e}  -> box=[{nlb[0]:g}, {nub[0]:g}]"
            f"{'  <-- UNSOUND FIX' if bad else ''}"
        )

    # ------------------------------------------------------------- Arm B (no weakening)
    print("\n-- Arm B (no weakening): an honest reduced cost still fixes as far as today --")
    for d_val, absum_val, gap in ((0.5, 1.5, 10.0), (2.0, 5.0, 7.0), (1.0, 3.0, 100.0)):
        d = np.array([d_val])
        absum = np.array([absum_val])
        lb = np.array([0.0])
        ub = np.array([10000.0])
        _, ub_new, n = _call(sol._reduced_cost_fixing, lb, ub, [0], d, 0.0, gap, rc_absum=absum)
        # reference: the same formula with no deflation at all
        g = gap + 1e-6 * (1.0 + abs(gap))
        ref = float(np.floor(g / d_val + 1e-9))
        executed += 1
        lost = float(ub_new[0]) != ref
        if lost:
            weakened += 1
        print(
            f"   d={d_val} absum={absum_val} gap={gap}  -> ub={ub_new[0]:g} "
            f"(undeflated {ref:g}) changes={n}{'  <-- WEAKENED' if lost else ''}"
        )

    print(f"\nEXECUTED ASSERTIONS: {executed}")
    if executed == 0:
        print("FAIL: the probe asserted nothing (CLAUDE.md §6)")
        return 2
    print(f"Arm A  unsound fixes         : {unsound}")
    print(f"Arm B  weakened fixes        : {weakened}")
    ok = marker_ok and unsound == 0 and weakened == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
